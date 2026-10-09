"""tools.py - Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urlunparse

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

# State for rate limiting
_LAST_ARXIV_CALL = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as e:
            if attempt == attempts - 1:
                raise
            if e.retry_after is not None:
                delay = float(e.retry_after)
            else:
                delay = min(cap, base * (2 ** attempt)) + random.uniform(0, 1.0)
            delay = min(delay, cap)
            time.sleep(delay)


def _check_http_response(response: httpx.Response):
    """Check response status and raise RetryableError for retryable status codes."""
    if response.status_code in (429, 500, 502, 503, 504):
        retry_after = response.headers.get("Retry-After")
        retry_after_val = None
        if retry_after:
            try:
                retry_after_val = float(retry_after)
            except ValueError:
                retry_after_val = None
        raise RetryableError(
            f"HTTP {response.status_code}: {response.text[:200]}",
            retry_after=retry_after_val,
        )


def _clean_error_message(err_msg: str) -> str:
    """Mask any API keys from error message string."""
    key = os.getenv("EXA_API_KEY")
    if key and key in err_msg:
        err_msg = err_msg.replace(key, "[REDACTED]")
    return err_msg


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _LAST_ARXIV_CALL
    try:
        # Keep only word characters (alphanumeric, underscores, hyphens)
        words = re.findall(r"[\w-]+", query)
        if not words:
            return "NO RESULTS"

        # Rate limiting: at least 3 seconds between arXiv calls
        now = time.time()
        elapsed = now - _LAST_ARXIV_CALL
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)

        max_results = max(1, min(int(max_results), 30))
        search_query = " AND ".join(f"all:{w}" for w in words)
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        }

        def do_request():
            global _LAST_ARXIV_CALL
            _LAST_ARXIV_CALL = time.time()
            try:
                with httpx.Client(timeout=20.0) as client:
                    resp = client.get(ARXIV_URL, params=params)
                    _check_http_response(resp)
                    resp.raise_for_status()
                    return resp.text
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                raise RetryableError(f"arXiv network error: {exc}") from exc

        xml_text = with_retry(do_request, attempts=5, base=2.0, cap=60.0)

        # Parse Atom XML
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)
        if not entries:
            return "NO RESULTS"

        records = []
        for entry in entries:
            id_elem = entry.find("atom:id", ns)
            pub_elem = entry.find("atom:published", ns)
            title_elem = entry.find("atom:title", ns)
            summary_elem = entry.find("atom:summary", ns)

            if id_elem is None or not id_elem.text:
                continue

            raw_id = id_elem.text.strip()
            # Extract id: after /abs/
            if "/abs/" in raw_id:
                paper_id = raw_id.split("/abs/")[-1]
            else:
                paper_id = raw_id.split("/")[-1]
            # Strip version suffix (e.g., 2501.00001v1 -> 2501.00001)
            paper_id = re.sub(r"v\d+$", "", paper_id)

            url = f"https://arxiv.org/abs/{paper_id}"
            published = pub_elem.text.strip()[:10] if pub_elem is not None and pub_elem.text else ""

            title = " ".join((title_elem.text or "").split()) if title_elem is not None else ""
            summary = " ".join((summary_elem.text or "").split()) if summary_elem is not None else ""
            if len(summary) > 600:
                summary = summary[:600] + "..."

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_clean_error_message(str(exc))}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        limit = max(1, min(int(limit), 100))
        params = {"limit": limit}
        if date and date.strip():
            params["date"] = date.strip()

        def do_request():
            try:
                with httpx.Client(timeout=20.0) as client:
                    resp = client.get(HF_DAILY_URL, params=params)
                    _check_http_response(resp)
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                raise RetryableError(f"HF network error: {exc}") from exc

        data = with_retry(do_request, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list) or not data:
            return "NO RESULTS"

        records = []
        kw = keyword.strip().lower() if keyword else ""
        for item in data:
            paper = item.get("paper", {})
            if not paper or not paper.get("id"):
                continue

            paper_id = paper.get("id")
            title = " ".join((paper.get("title") or item.get("title") or "").split())
            summary = " ".join((paper.get("summary") or item.get("summary") or "").split())
            if len(summary) > 600:
                summary = summary[:600] + "..."

            pub = paper.get("publishedAt") or item.get("publishedAt") or ""
            pub = pub[:10] if pub else ""

            upvotes = paper.get("upvotes", 0) or 0
            github = paper.get("githubRepo") or ""
            stars = paper.get("githubStars", 0) or 0
            url = f"https://huggingface.co/papers/{paper_id}"

            if kw and (kw not in title.lower() and kw not in summary.lower()):
                continue

            records.append({
                "id": paper_id,
                "url": url,
                "published": pub,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"

        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)
        return json.dumps(records, ensure_ascii=False)

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_clean_error_message(str(exc))}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        limit = max(1, min(int(limit), 50))
        params = {"q": query, "limit": limit}

        def do_request():
            try:
                with httpx.Client(timeout=20.0) as client:
                    resp = client.get(HF_SEARCH_URL, params=params)
                    _check_http_response(resp)
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                raise RetryableError(f"HF network error: {exc}") from exc

        data = with_retry(do_request, attempts=5, base=1.0, cap=30.0)
        if not isinstance(data, list) or not data:
            return "NO RESULTS"

        records = []
        for item in data:
            paper = item.get("paper", {})
            if not paper or not paper.get("id"):
                continue

            paper_id = paper.get("id")
            title = " ".join((paper.get("title") or item.get("title") or "").split())
            # Prefer ai_summary over summary when present
            summary_raw = paper.get("ai_summary") or paper.get("summary") or item.get("summary") or ""
            summary = " ".join(summary_raw.split())
            if len(summary) > 600:
                summary = summary[:600] + "..."

            pub = paper.get("publishedAt") or item.get("publishedAt") or ""
            pub = pub[:10] if pub else ""

            upvotes = paper.get("upvotes", 0) or 0
            github = paper.get("githubRepo") or ""
            stars = paper.get("githubStars", 0) or 0
            url = f"https://huggingface.co/papers/{paper_id}"

            records.append({
                "id": paper_id,
                "url": url,
                "published": pub,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_clean_error_message(str(exc))}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa_mcp(tool_name: str, arguments: dict) -> dict:
    """Call Exa MCP JSON-RPC over HTTP and return result dict."""
    api_key = os.getenv("EXA_API_KEY", "").strip()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Build URL with exaApiKey query param if provided
    url = f"{EXA_URL}?exaApiKey={api_key}" if api_key else EXA_URL

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    def do_request():
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                _check_http_response(resp)
                resp.raise_for_status()

                # Parse SSE or JSON
                raw_text = resp.text
                parsed_json = None
                for line in raw_text.splitlines():
                    if line.startswith("data:"):
                        data_part = line[len("data:"):].strip()
                        if data_part:
                            try:
                                parsed_json = json.loads(data_part)
                                break
                            except json.JSONDecodeError:
                                pass

                if parsed_json is None:
                    parsed_json = resp.json()

                # Check JSON-RPC error
                if "error" in parsed_json:
                    err = parsed_json["error"]
                    err_msg = str(err.get("message", err))
                    if "rate limit" in err_msg.lower() or "too many requests" in err_msg.lower():
                        raise RetryableError(f"Exa rate limit error: {err_msg}")
                    raise RuntimeError(f"Exa RPC error: {err_msg}")

                result = parsed_json.get("result", {})
                # Detect free-tier rate limit in result._meta or content
                meta = result.get("_meta", {})
                if meta.get("rate_limited") or meta.get("is_rate_limited"):
                    raise RetryableError("Exa rate limit flag in _meta")

                # Check text content for rate limit message
                content_list = result.get("content", [])
                for c in content_list:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text_val = c.get("text", "")
                        if "rate limit" in text_val.lower() and "dashboard.exa.ai" in text_val.lower():
                            raise RetryableError(f"Exa rate limit in response body: {text_val[:200]}")

                return result

        except (httpx.TransportError, httpx.TimeoutException) as exc:
            raise RetryableError(f"Exa network transport error: {exc}") from exc

    return with_retry(do_request, attempts=5, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        if not query or not query.strip():
            return "NO RESULTS"

        if not objective or not objective.strip():
            objective = f"Find comprehensive information and reliable sources about: {query}"

        args = {
            "query": query,
            "objective": objective,
            "numResults": max(1, min(int(num_results), 10)),
        }
        result = _call_exa_mcp("web_search_exa", args)
        contents = result.get("content", [])
        texts = [c["text"] for c in contents if isinstance(c, dict) and c.get("type") == "text" and c.get("text")]
        combined = "\n\n".join(texts).strip()

        if not combined:
            return "NO RESULTS"
        return combined

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_clean_error_message(str(exc))}"


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        if not url or not url.strip():
            return "NO RESULTS"

        args = {"urls": [url.strip()]}
        result = _call_exa_mcp("web_fetch_exa", args)
        contents = result.get("content", [])
        texts = [c["text"] for c in contents if isinstance(c, dict) and c.get("type") == "text" and c.get("text")]
        combined = "\n\n".join(texts).strip()

        if not combined:
            return "NO RESULTS"

        if len(combined) > 12000:
            combined = combined[:12000] + "\n\n...[TRUNCATED]"
        return combined

    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {_clean_error_message(str(exc))}"


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
