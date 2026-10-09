"""test_check_citations.py - Comprehensive Unit Tests for check_citations.py.

Tests cover:
  1. Happy path (single source, multiple sources, various source families)
  2. Grouped citations in body ([1, 2], [1-3], en-dash, em-dash, mixed ranges)
  3. False-positive citation avoidance (code blocks, inline code, markdown links [n](url))
  4. URL handling in reference lines (trailing punctuation, angle brackets, parentheses, query params)
  5. Cross-platform line endings (CRLF vs LF)
  6. Source validation edge cases (empty list, non-list, non-dict, non-int n, bool n, invalid url, duplicate url, duplicate n)
  7. Report structure edge cases (missing heading, wrong heading level, multiple headings, non-string input)
  8. Citation consistency edge cases (cited but not in sources, source never cited, empty body)
  9. Reference lines edge cases (missing line, duplicate line, unknown line, no url, multiple urls, url mismatch, bundled line)
  10. CLI main() function (exit codes 0 and 1, file I/O, error messages)
"""
import io
import json
import os
import sys
import tempfile
import unittest

from check_citations import check, main, _extract_citations, _extract_urls, _group_numbers


class TestGroupNumbers(unittest.TestCase):
    """Unit tests for _group_numbers helper."""

    def test_single_number(self):
        self.assertEqual(_group_numbers("1"), [1])
        self.assertEqual(_group_numbers(" 42 "), [42])

    def test_comma_separated(self):
        self.assertEqual(_group_numbers("1, 2, 3"), [1, 2, 3])
        self.assertEqual(_group_numbers("1,2,3"), [1, 2, 3])

    def test_hyphen_range(self):
        self.assertEqual(_group_numbers("1-4"), [1, 2, 3, 4])
        self.assertEqual(_group_numbers("2 - 5"), [2, 3, 4, 5])

    def test_en_dash_and_em_dash_range(self):
        self.assertEqual(_group_numbers("1–3"), [1, 2, 3])  # en-dash \u2013
        self.assertEqual(_group_numbers("4—6"), [4, 5, 6])  # em-dash \u2014

    def test_mixed_group(self):
        self.assertEqual(_group_numbers("1, 3-5, 8"), [1, 3, 4, 5, 8])

    def test_reverse_or_large_range(self):
        self.assertEqual(_group_numbers("5-2"), [5, 2])
        self.assertEqual(_group_numbers("1-500"), [1, 500])


class TestExtractCitations(unittest.TestCase):
    """Unit tests for _extract_citations helper."""

    def test_basic_citations(self):
        body = "Claim [1] and claim [2]."
        self.assertEqual(_extract_citations(body), {1, 2})

    def test_adjacent_citations(self):
        body = "Multiple references [1][2][3]."
        self.assertEqual(_extract_citations(body), {1, 2, 3})

    def test_grouped_citations(self):
        body = "Survey covers [1, 2] and [3-5]."
        self.assertEqual(_extract_citations(body), {1, 2, 3, 4, 5})

    def test_ignore_fenced_code_blocks(self):
        body = """
        Real citation [1].
        ```python
        # Code comment [2]
        x = [3, 4]
        ```
        Another citation [5].
        """
        self.assertEqual(_extract_citations(body), {1, 5})

    def test_ignore_inline_code(self):
        body = "Claim [1] with `x = array[2]` and claim [3]."
        self.assertEqual(_extract_citations(body), {1, 3})

    def test_ignore_markdown_links(self):
        body = "Citation [1] but link [2](https://example.com) should not be counted."
        self.assertEqual(_extract_citations(body), {1})

    def test_citations_with_internal_spaces(self):
        body = "Claim [ 1 ] and [ 2 , 3 ]."
        self.assertEqual(_extract_citations(body), {1, 2, 3})


class TestExtractUrls(unittest.TestCase):
    """Unit tests for _extract_urls helper."""

    def test_plain_url(self):
        line = "[1] Title. https://arxiv.org/abs/2501.00001 (2025)"
        self.assertEqual(_extract_urls(line), ["https://arxiv.org/abs/2501.00001"])

    def test_url_with_trailing_punctuation(self):
        line = "[1] Title. https://arxiv.org/abs/2501.00001."
        self.assertEqual(_extract_urls(line), ["https://arxiv.org/abs/2501.00001"])

    def test_url_with_angle_brackets(self):
        line = "[1] Title. <https://arxiv.org/abs/2501.00001>"
        self.assertEqual(_extract_urls(line), ["https://arxiv.org/abs/2501.00001"])

    def test_url_with_parentheses(self):
        line = "[1] Title. (https://arxiv.org/abs/2501.00001)"
        self.assertEqual(_extract_urls(line), ["https://arxiv.org/abs/2501.00001"])

    def test_url_containing_parens_like_wikipedia(self):
        url = "https://en.wikipedia.org/wiki/Foo_(bar)"
        line = f"[1] Title. {url} (2025)"
        self.assertEqual(_extract_urls(line, expected_url=url), [url])

    def test_multiple_urls(self):
        line = "[1] Paper A https://a.com and Paper B https://b.com"
        urls = _extract_urls(line)
        self.assertEqual(len(urls), 2)
        self.assertEqual(urls, ["https://a.com", "https://b.com"])


class TestHappyPaths(unittest.TestCase):
    """Happy path unit tests where reports are valid."""

    def test_valid_single_source(self):
        report = """# Title

## Background
According to foundational research [1], world models are important.

## References
[1] World Models. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)
"""
        sources = [
            {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "World Models", "source": "arxiv"}
        ]
        problems = check(report, sources)
        self.assertEqual(problems, [])

    def test_valid_multiple_sources_and_grouped_citations(self):
        report = """# Deep Agents Survey

## TL;DR
- Key findings cite [1, 2].
- Open problems cite [3-4].

## Body
Details on topic [2] and foundational work [1].
Recent benchmarks show [3][4].

## References
[1] Paper 1. arxiv. https://arxiv.org/abs/2501.00001 (2025)
[2] Paper 2. hf-daily. https://huggingface.co/papers/2501.00002 (2025)
[3] Paper 3. hf-search. https://huggingface.co/papers/2501.00003 (2025)
[4] Paper 4. web. https://example.org/agent-eval (2025)
"""
        sources = [
            {"n": 1, "url": "https://arxiv.org/abs/2501.00001"},
            {"n": 2, "url": "https://huggingface.co/papers/2501.00002"},
            {"n": 3, "url": "https://huggingface.co/papers/2501.00003"},
            {"n": 4, "url": "https://example.org/agent-eval"},
        ]
        self.assertEqual(check(report, sources), [])

    def test_windows_crlf_line_endings(self):
        report = "# Title\r\n\r\n## Body\r\nFact [1].\r\n\r\n## References\r\n[1] P. https://example.com/p1\r\n"
        sources = [{"n": 1, "url": "https://example.com/p1"}]
        self.assertEqual(check(report, sources), [])


class TestSourcesValidation(unittest.TestCase):
    """Edge cases for sources input validation."""

    def test_empty_sources(self):
        report = "# Title\n\n## References\n"
        self.assertEqual(check(report, []), ["no sources in sources.json"])

    def test_sources_not_a_list(self):
        report = "# Title\n\n## References\n"
        self.assertEqual(check(report, None), ["sources must be a list of objects"])
        self.assertEqual(check(report, {}), ["sources must be a list of objects"])
        self.assertEqual(check(report, "not-a-list"), ["sources must be a list of objects"])

    def test_source_entry_not_a_dict(self):
        report = "# Title\nFact [1].\n\n## References\n[1] https://example.com"
        problems = check(report, ["invalid"])
        self.assertTrue(any("must be an object" in p for p in problems))

    def test_source_n_not_an_int(self):
        report = "# Title\nFact [1].\n\n## References\n[1] https://example.com"
        for bad_n in ["1", 1.5, None]:
            with self.subTest(bad_n=bad_n):
                sources = [{"n": bad_n, "url": "https://example.com"}]
                problems = check(report, sources)
                self.assertTrue(any("'n' must be an int" in p for p in problems))

    def test_source_n_boolean_is_not_allowed(self):
        # In Python, isinstance(True, int) is True, but booleans must be rejected!
        report = "# Title\nFact [1].\n\n## References\n[1] https://example.com"
        sources = [{"n": True, "url": "https://example.com"}]
        problems = check(report, sources)
        self.assertTrue(any("'n' must be an int" in p for p in problems))

    def test_source_url_missing_or_invalid_protocol(self):
        report = "# Title\nFact [1].\n\n## References\n[1] ftp://example.com"
        for bad_url in [None, "", "ftp://example.com", "arxiv.org/abs/2501.00001", 123]:
            with self.subTest(bad_url=bad_url):
                sources = [{"n": 1, "url": bad_url}]
                problems = check(report, sources)
                self.assertTrue(any("url must start with http:// or https://" in p for p in problems))

    def test_duplicate_url_in_sources(self):
        report = "# Title\nFact [1] and [2].\n\n## References\n[1] https://example.com\n[2] https://example.com"
        sources = [
            {"n": 1, "url": "https://example.com"},
            {"n": 2, "url": "https://example.com"},
        ]
        problems = check(report, sources)
        self.assertTrue(any("duplicate url in sources.json" in p for p in problems))

    def test_duplicate_n_in_sources(self):
        report = "# Title\nFact [1].\n\n## References\n[1] https://example.com/1"
        sources = [
            {"n": 1, "url": "https://example.com/1"},
            {"n": 1, "url": "https://example.com/2"},
        ]
        problems = check(report, sources)
        self.assertTrue(any("duplicate source number in sources.json" in p for p in problems))


class TestReportStructure(unittest.TestCase):
    """Edge cases for report text structure."""

    def test_report_text_not_a_string(self):
        sources = [{"n": 1, "url": "https://example.com"}]
        self.assertEqual(check(None, sources), ["report_text must be a string"])
        self.assertEqual(check(12345, sources), ["report_text must be a string"])

    def test_missing_references_heading(self):
        report = "# Title\n\nBody fact [1].\n[1] https://example.com"
        sources = [{"n": 1, "url": "https://example.com"}]
        problems = check(report, sources)
        self.assertTrue(any("missing '## References' heading" in p for p in problems))

    def test_wrong_heading_level_or_spelling(self):
        sources = [{"n": 1, "url": "https://example.com"}]
        for bad_heading in ["# References", "### References", "## Reference", "## Bibliography"]:
            with self.subTest(bad_heading=bad_heading):
                report = f"# Title\nFact [1].\n\n{bad_heading}\n[1] https://example.com"
                problems = check(report, sources)
                self.assertTrue(any("missing '## References' heading" in p for p in problems))

    def test_multiple_references_headings(self):
        report = """# Title
## References
Extra heading
## References
[1] https://example.com
Fact [1].
"""
        sources = [{"n": 1, "url": "https://example.com"}]
        problems = check(report, sources)
        self.assertTrue(any("multiple '## References' headings" in p for p in problems))


class TestCitedVsSourcesConsistency(unittest.TestCase):
    """Edge cases for citations in body vs sources.json consistency."""

    def test_number_cited_in_body_missing_from_sources(self):
        report = """# Title
Fact [1] and ghost fact [2].

## References
[1] https://example.com/1
"""
        sources = [{"n": 1, "url": "https://example.com/1"}]
        problems = check(report, sources)
        self.assertTrue(any("[2] cited but missing from sources.json" in p for p in problems))

    def test_source_never_cited_in_body(self):
        report = """# Title
Only cited [1].

## References
[1] https://example.com/1
[2] https://example.com/2
"""
        sources = [
            {"n": 1, "url": "https://example.com/1"},
            {"n": 2, "url": "https://example.com/2"},
        ]
        problems = check(report, sources)
        self.assertTrue(any("source [2] never cited" in p for p in problems))

    def test_empty_body_no_citations(self):
        report = """# Title
No citations anywhere in the text.

## References
[1] https://example.com/1
"""
        sources = [{"n": 1, "url": "https://example.com/1"}]
        problems = check(report, sources)
        self.assertTrue(any("source [1] never cited" in p for p in problems))

    def test_citation_only_in_code_block_fails(self):
        report = """# Title
```python
# Citation [1] in code only
```

## References
[1] https://example.com/1
"""
        sources = [{"n": 1, "url": "https://example.com/1"}]
        problems = check(report, sources)
        self.assertTrue(any("source [1] never cited" in p for p in problems))

    def test_citation_only_in_markdown_link_fails(self):
        report = """# Title
See [1](https://example.com/1) for details.

## References
[1] https://example.com/1
"""
        sources = [{"n": 1, "url": "https://example.com/1"}]
        problems = check(report, sources)
        self.assertTrue(any("source [1] never cited" in p for p in problems))


class TestReferenceLines(unittest.TestCase):
    """Edge cases for the ## References section lines."""

    def test_missing_reference_line(self):
        report = """# Title
Fact [1] and [2].

## References
[1] https://example.com/1
"""
        sources = [
            {"n": 1, "url": "https://example.com/1"},
            {"n": 2, "url": "https://example.com/2"},
        ]
        problems = check(report, sources)
        self.assertTrue(any("missing reference line for source [2]" in p for p in problems))

    def test_duplicate_reference_line(self):
        report = """# Title
Fact [1].

## References
[1] Title A. https://example.com/1
[1] Title B. https://example.com/1
"""
        sources = [{"n": 1, "url": "https://example.com/1"}]
        problems = check(report, sources)
        self.assertTrue(any("duplicate reference line for [1]" in p for p in problems))

    def test_reference_line_not_in_sources(self):
        report = """# Title
Fact [1].

## References
[1] Title 1. https://example.com/1
[99] Title 99. https://example.com/99
"""
        sources = [{"n": 1, "url": "https://example.com/1"}]
        problems = check(report, sources)
        self.assertTrue(any("reference line [99] is not in sources.json" in p for p in problems))

    def test_reference_line_missing_url(self):
        report = """# Title
Fact [1].

## References
[1] Title 1 without any URL
"""
        sources = [{"n": 1, "url": "https://example.com/1"}]
        problems = check(report, sources)
        self.assertTrue(any("reference line [1] has no http(s) URL" in p for p in problems))

    def test_reference_line_multiple_urls_bundling(self):
        report = """# Title
Fact [1].

## References
[1] Paper A https://example.com/1 and Paper B https://example.com/extra
"""
        sources = [{"n": 1, "url": "https://example.com/1"}]
        problems = check(report, sources)
        self.assertTrue(any("holds 2 URLs" in p or "bundling several sources" in p for p in problems))

    def test_reference_line_url_mismatch(self):
        report = """# Title
Fact [1].

## References
[1] Title 1. https://wrong-domain.com/1
"""
        sources = [{"n": 1, "url": "https://example.com/1"}]
        problems = check(report, sources)
        self.assertTrue(any("URL does not match sources.json" in p for p in problems))

    def test_reference_line_bundled_numbers_syntax(self):
        report = """# Title
Fact [1] and [2].

## References
[1, 2] Bundled Papers. https://example.com/1
"""
        sources = [
            {"n": 1, "url": "https://example.com/1"},
            {"n": 2, "url": "https://example.com/2"},
        ]
        problems = check(report, sources)
        self.assertTrue(any("bundles multiple sources" in p for p in problems))


class TestCommandLineInterface(unittest.TestCase):
    """Test CLI invocation and exit codes of check_citations.py."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.report_path = os.path.join(self.temp_dir.name, "report.md")
        self.sources_path = os.path.join(self.temp_dir.name, "sources.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cli_success_exit_0(self):
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("# Title\nFact [1].\n\n## References\n[1] Title. https://example.com/1\n")
        with open(self.sources_path, "w", encoding="utf-8") as f:
            json.dump([{"n": 1, "url": "https://example.com/1"}], f)

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = stdout_capture
            code = main(["check_citations.py", self.report_path, self.sources_path])
        finally:
            sys.stdout = old_stdout

        self.assertEqual(code, 0)
        self.assertIn("OK: 1 sources, all citations resolve", stdout_capture.getvalue())

    def test_cli_failure_exit_1(self):
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write("# Title\nFact [1] and uncited [2].\n\n## References\n[1] Title. https://example.com/1\n")
        with open(self.sources_path, "w", encoding="utf-8") as f:
            json.dump([{"n": 1, "url": "https://example.com/1"}], f)

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = stdout_capture
            code = main(["check_citations.py", self.report_path, self.sources_path])
        finally:
            sys.stdout = old_stdout

        self.assertEqual(code, 1)
        self.assertIn("[2] cited but missing from sources.json", stdout_capture.getvalue())

    def test_cli_nonexistent_files(self):
        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = stdout_capture
            code = main(["check_citations.py", "nonexistent_report.md", "nonexistent_sources.json"])
        finally:
            sys.stdout = old_stdout

        self.assertEqual(code, 1)
        self.assertIn("cannot read inputs", stdout_capture.getvalue())


if __name__ == "__main__":
    unittest.main()
