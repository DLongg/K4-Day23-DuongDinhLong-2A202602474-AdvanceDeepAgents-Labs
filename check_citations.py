"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_GROUP = re.compile(r"\[\s*(\d+(?:\s*[,–—\-]\s*\d+)*)\s*\](?!\()")
_URL_RE = re.compile(r"https?://\S+")
_BUNDLED_REF_LINE = re.compile(r"^\[\s*\d+(?:\s*[,–—\-]\s*\d+)+\s*\]")


def _group_numbers(group):
    """Extract individual citation numbers from a grouped string like '1, 2' or '1-3'."""
    numbers = []
    for part in re.split(r"\s*,\s*", group.strip()):
        span = re.fullmatch(r"(\d+)\s*[–—-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def _extract_citations(body):
    """Extract all cited numbers from body text, ignoring code blocks and markdown links."""
    segments = _CODE.split(body)
    cited = set()
    for i, segment in enumerate(segments):
        if i % 2:  # odd indexes are code blocks or inline code
            continue
        for match in _GROUP.finditer(segment):
            for n in _group_numbers(match.group(1)):
                cited.add(n)
    return cited


def _extract_urls(line, expected_url=None):
    """Extract http(s) URLs from a line, cleaning common trailing punctuation."""
    raw_urls = _URL_RE.findall(line)
    cleaned = []
    for u in raw_urls:
        if expected_url and u == expected_url:
            cleaned.append(u)
            continue
        clean = u.strip("<>").rstrip(".,;)>]")
        if expected_url and clean == expected_url:
            cleaned.append(expected_url)
        else:
            cleaned.append(clean)
    return cleaned


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []

    # 1. Validate sources
    if not isinstance(sources, list):
        return ["sources must be a list of objects"]
    if len(sources) == 0:
        return ["no sources in sources.json"]

    valid_sources = {}
    seen_n = set()
    seen_urls = set()

    for entry in sources:
        if not isinstance(entry, dict):
            problems.append(f"source entry must be an object, got {type(entry).__name__}")
            continue

        n = entry.get("n")
        n_valid = False
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"source entry 'n' must be an int, got {n!r}")
        else:
            n_valid = True
            if n in seen_n:
                problems.append(f"duplicate source number in sources.json: [{n}]")
            seen_n.add(n)

        url = entry.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n if n_valid else '?'}] url must start with http:// or https://: {url!r}")
        elif url in seen_urls:
            problems.append(f"duplicate url in sources.json: '{url}'")
        else:
            seen_urls.add(url)

        if n_valid and n not in valid_sources:
            valid_sources[n] = entry

    # 2. Validate report_text and split at "## References"
    if not isinstance(report_text, str):
        problems.append("report_text must be a string")
        return problems

    # Normalize line endings for cross-platform compatibility (CRLF -> LF)
    report_text_norm = report_text.replace("\r\n", "\n").replace("\r", "\n")

    ref_matches = list(_REF_HEADING.finditer(report_text_norm))
    if not ref_matches:
        problems.append("missing '## References' heading")
        body = report_text_norm
        ref_text = ""
    else:
        if len(ref_matches) > 1:
            problems.append(f"multiple '## References' headings found ({len(ref_matches)})")
        last_match = ref_matches[-1]
        body = report_text_norm[:last_match.start()]
        ref_text = report_text_norm[last_match.end():]

    # 3. Check citations in body
    cited = _extract_citations(body)

    for n in sorted(cited):
        if n not in valid_sources:
            problems.append(f"[{n}] cited but missing from sources.json")

    for n in sorted(valid_sources):
        if n not in cited:
            problems.append(f"source [{n}] never cited")

    # 4. Check References section lines
    ref_lines_by_n = {}
    ref_seen_n = set()

    for line in ref_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Check for bundled ranges like [1, 2] or [1-3] at start of reference lines
        if _BUNDLED_REF_LINE.match(stripped):
            problems.append(f"reference line bundles multiple sources: {stripped[:60]}")
            continue

        m = re.match(r"^\[(\d+)\]\s*(.*)$", stripped)
        if not m:
            continue

        ref_n = int(m.group(1))
        if ref_n in ref_seen_n:
            problems.append(f"duplicate reference line for [{ref_n}]")
            continue

        ref_seen_n.add(ref_n)
        ref_lines_by_n[ref_n] = stripped

        if ref_n not in valid_sources:
            problems.append(f"reference line [{ref_n}] is not in sources.json")

        expected_url = valid_sources[ref_n]["url"] if ref_n in valid_sources else None
        urls = _extract_urls(stripped, expected_url=expected_url)

        if len(urls) == 0:
            problems.append(f"reference line [{ref_n}] has no http(s) URL")
        elif len(urls) > 1:
            problems.append(f"reference line [{ref_n}] holds {len(urls)} URLs (bundling several sources under one number is not allowed)")
        elif expected_url and urls[0] != expected_url:
            problems.append(f"reference line [{ref_n}] URL does not match sources.json: expected '{expected_url}', got '{urls[0]}'")

    # Ensure every source in sources.json has exactly one reference line
    for n in sorted(valid_sources):
        if n not in ref_lines_by_n:
            problems.append(f"missing reference line for source [{n}]")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
