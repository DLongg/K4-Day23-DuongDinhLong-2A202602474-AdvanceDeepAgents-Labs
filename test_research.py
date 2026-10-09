"""test_research.py - Unit tests for research.py helper functions: slugify, summarize, save_outputs, and main."""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from research import build_prompt, main, save_outputs, slugify, summarize


class TestSlugify(unittest.TestCase):
    """Test safe file name generation from arbitrary topic strings."""

    def test_standard_topic(self):
        self.assertEqual(slugify("survey about world model"), "survey-about-world-model")

    def test_path_traversal_prevention(self):
        self.assertEqual(slugify("../../x"), "x")
        self.assertEqual(slugify("../../../etc/passwd"), "etc-passwd")

    def test_special_characters(self):
        self.assertEqual(slugify("AI & LLMs: 2026 Edition!"), "ai-llms-2026-edition")

    def test_empty_or_whitespace_fallback(self):
        self.assertEqual(slugify(""), "topic")
        self.assertEqual(slugify("    "), "topic")
        self.assertEqual(slugify(None), "topic")

    def test_length_capping(self):
        long_topic = "a" * 100
        slug = slugify(long_topic)
        self.assertTrue(len(slug) <= 60)
        self.assertFalse(slug.endswith("-"))


class TestSummarize(unittest.TestCase):
    """Test lead message analysis: tool calls, subagent calls, token counts, elapsed time."""

    def test_summarize_metrics(self):
        msg1 = MagicMock()
        msg1.tool_calls = [
            {"name": "write_todos"},
            {"name": "task"},
            {"name": "task"},
        ]
        msg1.usage_metadata = {"input_tokens": 1200, "output_tokens": 350}

        msg2 = MagicMock()
        msg2.tool_calls = [{"name": "task"}, {"name": "execute"}]
        msg2.usage_metadata = {"input_tokens": 800, "output_tokens": 150}

        messages = [msg1, msg2]
        res = summarize(messages, elapsed=45.267, model_name="gpt-4o")

        self.assertEqual(res["model"], "gpt-4o")
        self.assertEqual(res["elapsed_s"], 45.3)
        self.assertEqual(res["subagent_calls"], 3)
        self.assertEqual(res["tool_calls"]["task"], 3)
        self.assertEqual(res["tool_calls"]["write_todos"], 1)
        self.assertEqual(res["tool_calls"]["execute"], 1)
        self.assertEqual(res["tokens"]["input"], 2000)
        self.assertEqual(res["tokens"]["output"], 500)


class TestSaveOutputs(unittest.TestCase):
    """Test saving reports and handling failure scenarios."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("research.download")
    def test_save_outputs_success(self, mock_download):
        mock_backend = MagicMock()
        mock_download.return_value = {
            "/tmp/work/report/report.md": b"# Final Report\nBody text [1].\n",
            "/tmp/work/research/sources.json": json.dumps([
                {"n": 1, "url": "https://arxiv.org/abs/1", "source": "arxiv"},
                {"n": 2, "url": "https://huggingface.co/papers/2", "source": "hf-search"},
                {"n": 3, "url": "https://example.com/3", "source": "web"},
            ]).encode("utf-8"),
        }

        report_file = save_outputs(
            backend=mock_backend,
            topic="World Models",
            messages=[],
            elapsed=10.0,
            model_name="test-model",
            reports_dir=self.temp_dir.name,
        )

        self.assertTrue(report_file.exists())
        self.assertEqual(report_file.name, "world-models.md")

        sources_file = report_file.with_name("world-models.sources.json")
        meta_file = report_file.with_name("world-models.meta.json")

        self.assertTrue(sources_file.exists())
        self.assertTrue(meta_file.exists())

        meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
        self.assertEqual(meta_data["n_sources"], 3)
        self.assertEqual(meta_data["source_families"], ["arxiv", "hf-search", "web"])

    @patch("research.download")
    def test_save_outputs_missing_report_raises_runtime_error(self, mock_download):
        mock_backend = MagicMock()
        mock_download.return_value = {
            "/tmp/work/report/report.md": b"",
            "/tmp/work/research/sources.json": b"[]",
        }

        with self.assertRaises(RuntimeError):
            save_outputs(
                backend=mock_backend,
                topic="World Models",
                messages=[],
                elapsed=10.0,
                model_name="test-model",
                reports_dir=self.temp_dir.name,
            )

        # Ensure nothing was written into reports directory
        files = list(os.listdir(self.temp_dir.name))
        self.assertEqual(files, [])

    @patch("research.download")
    def test_save_outputs_corrupt_sources_raises_runtime_error(self, mock_download):
        mock_backend = MagicMock()
        mock_download.return_value = {
            "/tmp/work/report/report.md": b"# Report Content",
            "/tmp/work/research/sources.json": b"corrupt-json",
        }

        with self.assertRaises(RuntimeError):
            save_outputs(
                backend=mock_backend,
                topic="World Models",
                messages=[],
                elapsed=10.0,
                model_name="test-model",
                reports_dir=self.temp_dir.name,
            )


class TestMainFunction(unittest.TestCase):
    """Test CLI main entry point in research.py."""

    def test_empty_topic_returns_2(self):
        self.assertEqual(main(""), 2)
        self.assertEqual(main("   "), 2)


if __name__ == "__main__":
    unittest.main()
