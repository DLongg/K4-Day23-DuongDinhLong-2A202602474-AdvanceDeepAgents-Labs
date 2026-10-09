"""test_tools.py - Unit tests for tools.py: with_retry, error handling, rate limiting, and 5 source tools."""
import json
import os
import unittest
from unittest.mock import MagicMock, patch

import httpx

import tools
from tools import (
    RetryableError,
    _check_http_response,
    _clean_error_message,
    arxiv_search,
    hf_daily_papers,
    hf_search_papers,
    web_fetch,
    web_search,
    with_retry,
)


class TestWithRetry(unittest.TestCase):
    """Test retry helper logic, backoff, jitter, Retry-After header, and failure behavior."""

    def test_immediate_success(self):
        fn = MagicMock(return_value="ok")
        result = with_retry(fn, attempts=3)
        self.assertEqual(result, "ok")
        self.assertEqual(fn.call_count, 1)

    @patch("tools.time.sleep")
    def test_retry_on_retryable_error_then_succeed(self, mock_sleep):
        fn = MagicMock(side_effect=[RetryableError("temporary"), "success"])
        result = with_retry(fn, attempts=3, base=0.1, cap=5.0)
        self.assertEqual(result, "success")
        self.assertEqual(fn.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    @patch("tools.time.sleep")
    def test_retry_exhausted_raises(self, mock_sleep):
        fn = MagicMock(side_effect=RetryableError("fail forever"))
        with self.assertRaises(RetryableError):
            with_retry(fn, attempts=3, base=0.1, cap=5.0)
        self.assertEqual(fn.call_count, 3)
        # Should not sleep after the final attempt
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("tools.time.sleep")
    def test_non_retryable_error_bubbles_immediately(self, mock_sleep):
        fn = MagicMock(side_effect=ValueError("programmer error"))
        with self.assertRaises(ValueError):
            with_retry(fn, attempts=3)
        self.assertEqual(fn.call_count, 1)
        self.assertEqual(mock_sleep.call_count, 0)

    @patch("tools.time.sleep")
    def test_respects_retry_after_header(self, mock_sleep):
        fn = MagicMock(side_effect=[RetryableError("rate limited", retry_after=7.5), "success"])
        result = with_retry(fn, attempts=3, base=1.0, cap=10.0)
        self.assertEqual(result, "success")
        mock_sleep.assert_called_once_with(7.5)


class TestCheckHttpResponse(unittest.TestCase):
    """Test status code classification for retryable errors."""

    def test_200_ok_does_not_raise(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        _check_http_response(resp)

    def test_429_raises_retryable_error_with_header(self):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 429
        resp.headers = {"Retry-After": "12.0"}
        resp.text = "Too Many Requests"
        with self.assertRaises(RetryableError) as ctx:
            _check_http_response(resp)
        self.assertEqual(ctx.exception.retry_after, 12.0)

    def test_500_series_raises_retryable_error(self):
        for code in (500, 502, 503, 504):
            with self.subTest(code=code):
                resp = MagicMock(spec=httpx.Response)
                resp.status_code = code
                resp.headers = {}
                resp.text = f"Server Error {code}"
                with self.assertRaises(RetryableError):
                    _check_http_response(resp)


class TestCleanErrorMessage(unittest.TestCase):
    """Test masking API keys from exception messages."""

    def test_masking_key_when_present(self):
        with patch.dict(os.environ, {"EXA_API_KEY": "secret_exa_key_12345"}):
            msg = "Error connecting to https://mcp.exa.ai/mcp?exaApiKey=secret_exa_key_12345"
            cleaned = _clean_error_message(msg)
            self.assertNotIn("secret_exa_key_12345", cleaned)
            self.assertIn("[REDACTED]", cleaned)


class TestArxivSearchTool(unittest.TestCase):
    """Test arxiv_search tool input cleaning, parsing, and error handling."""

    def test_empty_or_whitespace_query(self):
        self.assertEqual(arxiv_search.invoke({"query": ""}), "NO RESULTS")
        self.assertEqual(arxiv_search.invoke({"query": "   "}), "NO RESULTS")
        self.assertEqual(arxiv_search.invoke({"query": "@#$%^"}), "NO RESULTS")

    @patch("tools.httpx.Client")
    def test_successful_atom_xml_parsing(self, mock_client_cls):
        sample_xml = """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2501.00001v2</id>
            <published>2025-01-02T12:00:00Z</published>
            <title>World Models Survey</title>
            <summary>Comprehensive overview of world models in robotics.</summary>
          </entry>
        </feed>"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = sample_xml
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        output = arxiv_search.invoke({"query": "world models", "max_results": 2})
        data = json.loads(output)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        # Verify version suffix stripped
        self.assertEqual(data[0]["id"], "2501.00001")
        self.assertEqual(data[0]["url"], "https://arxiv.org/abs/2501.00001")
        self.assertEqual(data[0]["published"], "2025-01-02")
        self.assertEqual(data[0]["title"], "World Models Survey")

    @patch("tools.time.sleep")
    @patch("tools.httpx.Client")
    def test_network_failure_returns_error_string(self, mock_client_cls, mock_sleep):
        mock_client_cls.return_value.__enter__.side_effect = httpx.ConnectError("Network is down")
        output = arxiv_search.invoke({"query": "robotics"})
        self.assertTrue(output.startswith("ERROR:"))


class TestHuggingFaceTools(unittest.TestCase):
    """Test hf_daily_papers and hf_search_papers tools."""

    @patch("tools.httpx.Client")
    def test_hf_daily_papers_keyword_filter_and_sorting(self, mock_client_cls):
        sample_data = [
            {
                "paper": {"id": "p1", "title": "Paper One", "summary": "About LLMs", "upvotes": 10},
                "publishedAt": "2025-01-01",
            },
            {
                "paper": {"id": "p2", "title": "Paper Two", "summary": "About Vision", "upvotes": 50},
                "publishedAt": "2025-01-02",
            },
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_data
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        output = hf_daily_papers.invoke({"limit": 10, "keyword": "vision"})
        data = json.loads(output)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], "p2")
        self.assertEqual(data[0]["url"], "https://huggingface.co/papers/p2")

    @patch("tools.httpx.Client")
    def test_hf_search_papers_ai_summary_preference(self, mock_client_cls):
        sample_data = [
            {
                "paper": {
                    "id": "p3",
                    "title": "Reasoning Models",
                    "summary": "Long raw summary text...",
                    "ai_summary": "Concise AI summary.",
                    "upvotes": 15,
                },
                "publishedAt": "2025-02-01",
            }
        ]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_data
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        output = hf_search_papers.invoke({"query": "reasoning"})
        data = json.loads(output)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["summary"], "Concise AI summary.")


class TestExaTools(unittest.TestCase):
    """Test web_search and web_fetch Exa MCP tools."""

    def test_web_search_empty_query(self):
        self.assertEqual(web_search.invoke({"query": ""}), "NO RESULTS")

    def test_web_fetch_empty_url(self):
        self.assertEqual(web_fetch.invoke({"url": ""}), "NO RESULTS")

    @patch("tools._call_exa_mcp")
    def test_web_search_success(self, mock_call):
        mock_call.return_value = {
            "content": [
                {"type": "text", "text": "Result 1 from web"},
                {"type": "text", "text": "Result 2 from web"},
            ]
        }
        res = web_search.invoke({"query": "world models"})
        self.assertIn("Result 1 from web", res)
        self.assertIn("Result 2 from web", res)

    @patch("tools._call_exa_mcp")
    def test_web_fetch_truncation(self, mock_call):
        long_text = "A" * 15000
        mock_call.return_value = {"content": [{"type": "text", "text": long_text}]}
        res = web_fetch.invoke({"url": "https://example.com"})
        self.assertIn("[TRUNCATED]", res)
        self.assertTrue(len(res) < 13000)


if __name__ == "__main__":
    unittest.main()
