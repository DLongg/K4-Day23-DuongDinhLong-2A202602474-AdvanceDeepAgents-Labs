"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, NOTES_DIR, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    text = (topic or "").strip().lower()
    # Runs of non-word characters become one "-"
    slug = re.sub(r"[^\w]+", "-", text).strip("-")
    if not slug:
        return "topic"
    slug = slug[:60].strip("-")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Conduct a comprehensive research survey on the topic: '{topic}'.\n"
        f"Please execute the complete research workflow:\n"
        f"1. Plan and split the topic into 3-5 sub-questions using write_todos.\n"
        f"2. Delegate sub-questions to researcher subagents using the task tool (specifying subagent_type='researcher' and target source families).\n"
        f"3. Consolidate notes into {SOURCES_PATH} using the write_file tool.\n"
        f"4. Synthesize the final report into {REPORT_PATH} using the write_file tool (do NOT write ## References section).\n"
        f"5. Finalize references by executing python3 {FINALIZER_PATH} via the execute tool.\n"
        f"6. Validate references by executing python3 {VALIDATOR_PATH} via the execute tool until it passes.\n"
        f"CRITICAL: Always use write_file to save {SOURCES_PATH} and {REPORT_PATH} into the sandbox filesystem!"
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_counts = Counter()
    total_input_tokens = 0
    total_output_tokens = 0

    for msg in messages or []:
        # Extract tool calls
        calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            calls = msg.tool_calls
        elif isinstance(msg, dict) and msg.get("tool_calls"):
            calls = msg["tool_calls"]
        elif hasattr(msg, "additional_kwargs") and isinstance(msg.additional_kwargs, dict) and msg.additional_kwargs.get("tool_calls"):
            calls = msg.additional_kwargs["tool_calls"]

        for call in calls:
            name = None
            if isinstance(call, dict):
                name = call.get("name")
                if not name and isinstance(call.get("function"), dict):
                    name = call["function"].get("name")
            elif hasattr(call, "name"):
                name = call.name
            if name:
                tool_counts[name] += 1

        # Extract usage metadata
        meta = None
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            meta = msg.usage_metadata
        elif isinstance(msg, dict) and msg.get("usage_metadata"):
            meta = msg["usage_metadata"]
        elif hasattr(msg, "response_metadata") and isinstance(msg.response_metadata, dict):
            meta = msg.response_metadata.get("token_usage")

        if isinstance(meta, dict):
            inp = meta.get("input_tokens") or meta.get("prompt_tokens") or 0
            out = meta.get("output_tokens") or meta.get("completion_tokens") or 0
            total_input_tokens += int(inp)
            total_output_tokens += int(out)

    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": tool_counts.get("task", 0),
        "tool_calls": dict(tool_counts),
        "tokens": {
            "input": total_input_tokens,
            "output": total_output_tokens,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_content = files.get(REPORT_PATH)
    sources_content = files.get(SOURCES_PATH)

    # Recovery 1: If sources.json is missing but notes exist in sandbox, compile them
    if (not sources_content or not sources_content.strip()) and hasattr(backend, "execute") and hasattr(backend, "upload_files"):
        compile_script = (
            "import json, re, glob\n"
            "sources = []\n"
            "seen_urls = set()\n"
            "n = 1\n"
            "for path in sorted(glob.glob('/tmp/work/research/notes/*.md')):\n"
            "    with open(path, encoding='utf-8') as f:\n"
            "        text = f.read()\n"
            "    blocks = re.split(r'(?m)^###\\s+', text)\n"
            "    for block in blocks:\n"
            "        if not block.strip(): continue\n"
            "        title = block.splitlines()[0].strip().strip('[]')\n"
            "        id_m = re.search(r'- id:\\s*(.+)', block)\n"
            "        url_m = re.search(r'- url:\\s*(https?://\\S+)', block)\n"
            "        date_m = re.search(r'- date:\\s*(\\S+)', block)\n"
            "        src_m = re.search(r'- source:\\s*(\\S+)', block)\n"
            "        url = url_m.group(1).rstrip('.)') if url_m else ''\n"
            "        if url and url not in seen_urls:\n"
            "            seen_urls.add(url)\n"
            "            paper_id = id_m.group(1).strip() if id_m else url.split('/')[-1]\n"
            "            date = date_m.group(1).strip() if date_m else '2025-01-01'\n"
            "            source = src_m.group(1).strip() if src_m else ('arxiv' if 'arxiv' in url else 'web')\n"
            "            sources.append({'n': n, 'id': paper_id, 'url': url, 'title': title, 'date': date, 'source': source})\n"
            "            n += 1\n"
            "if sources:\n"
            "    with open('/tmp/work/research/sources.json', 'w', encoding='utf-8') as f:\n"
            "        json.dump(sources, f, indent=2, ensure_ascii=False)\n"
        )
        backend.execute(f"python3 -c \"{compile_script}\"")
        files = download(backend, [REPORT_PATH, SOURCES_PATH])
        sources_content = files.get(SOURCES_PATH)

    # Recovery 2: If report is missing on disk but present in the assistant messages
    if (not report_content or not report_content.strip()) and messages:
        for msg in reversed(messages):
            content = getattr(msg, "content", "") if not isinstance(msg, dict) else msg.get("content", "")
            if isinstance(content, str) and ("## TL;DR" in content or "## Background" in content):
                backend.upload_files([(REPORT_PATH, content.encode("utf-8"))])
                if hasattr(backend, "execute"):
                    backend.execute(f"python3 {FINALIZER_PATH}")
                    backend.execute(f"python3 {VALIDATOR_PATH}")
                files = download(backend, [REPORT_PATH, SOURCES_PATH])
                report_content = files.get(REPORT_PATH)
                sources_content = files.get(SOURCES_PATH)
                break

    if not report_content or not report_content.strip():
        raise RuntimeError(f"Report is missing or empty at {REPORT_PATH}")

    if not sources_content:
        raise RuntimeError(f"sources.json is missing at {SOURCES_PATH}")

    try:
        sources_raw = sources_content.decode("utf-8") if isinstance(sources_content, bytes) else str(sources_content)
        sources_data = json.loads(sources_raw)
        if not isinstance(sources_data, list):
            raise ValueError("sources.json must be a JSON list")
    except Exception as exc:
        raise RuntimeError(f"sources.json is missing/invalid JSON: {exc}") from exc

    slug = slugify(topic)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    summary = summarize(messages, elapsed, model_name)
    n_sources = len(sources_data)
    source_families = sorted(list({
        str(item["source"]).strip()
        for item in sources_data
        if isinstance(item, dict) and "source" in item and item["source"]
    }))

    meta = {
        "topic": topic,
        **summary,
        "n_sources": n_sources,
        "source_families": source_families,
    }

    report_file = reports_path / f"{slug}.md"
    sources_file = reports_path / f"{slug}.sources.json"
    meta_file = reports_path / f"{slug}.meta.json"

    sources_file.write_bytes(sources_content if isinstance(sources_content, bytes) else sources_content.encode("utf-8"))
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    report_file.write_bytes(report_content if isinstance(report_content, bytes) else report_content.encode("utf-8"))

    return report_file


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    if not topic or not str(topic).strip():
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2

    topic = str(topic).strip()
    try:
        model = make_model()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL") or "unknown"
    start = time.monotonic()

    try:
        with open_sandbox() as backend:
            backend.execute(f"mkdir -p {NOTES_DIR} {WORKDIR}/report")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            agent = build_lead_agent(backend, model)

            print(f"=== Researching topic: '{topic}' ===", flush=True)
            messages = [{"role": "user", "content": build_prompt(topic)}]
            max_turns = 4
            for turn in range(max_turns):
                try:
                    for chunk in agent.stream(
                        {"messages": messages},
                        config={"recursion_limit": 1000},
                        stream_mode="values",
                    ):
                        current_msgs = chunk.get("messages", [])
                        if current_msgs and len(current_msgs) > len(messages):
                            for new_msg in current_msgs[len(messages):]:
                                msg_type = type(new_msg).__name__
                                if hasattr(new_msg, "tool_calls") and new_msg.tool_calls:
                                    calls = [tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None) for tc in new_msg.tool_calls]
                                    print(f"  [{msg_type}] Calling tool(s): {', '.join(str(c) for c in calls)}", flush=True)
                                elif hasattr(new_msg, "name") and new_msg.name:
                                    print(f"  [ToolResult] {new_msg.name}: {str(new_msg.content)[:100]}...", flush=True)
                                elif hasattr(new_msg, "content") and new_msg.content:
                                    text_snip = str(new_msg.content).strip()
                                    if text_snip:
                                        print(f"  [{msg_type}] {text_snip[:120]}...", flush=True)
                            messages = current_msgs
                except Exception as stream_exc:
                    print(f"Notice during agent run: {stream_exc}", file=sys.stderr)

                # Check if report.md and sources.json exist in sandbox
                check_files = download(backend, [REPORT_PATH, SOURCES_PATH])
                if check_files.get(REPORT_PATH) and check_files.get(SOURCES_PATH):
                    print("Report and sources are verified in sandbox!", flush=True)
                    break

                if turn < max_turns - 1:
                    reminder = (
                        f"The research notes are available in {NOTES_DIR}.\n"
                        "Now proceed to complete the task:\n"
                        f"1. Use write_file to save all compiled sources into {SOURCES_PATH} (schema: [{{'n': 1, 'id': '...', 'url': '...', 'title': '...', 'date': '...', 'source': '...'}}]).\n"
                        f"2. Use write_file to save the complete survey report into {REPORT_PATH} (do NOT write ## References section).\n"
                        f"3. Run python3 {FINALIZER_PATH} using the execute tool.\n"
                        f"4. Run python3 {VALIDATOR_PATH} using the execute tool until it prints OK.\n"
                        "You MUST execute write_file now to create the files."
                    )
                    print(f"  -> Prompting lead agent to proceed to synthesis (turn {turn + 2}/{max_turns})...", flush=True)
                    messages.append({"role": "user", "content": reminder})

            elapsed = time.monotonic() - start
            report_path = save_outputs(backend, topic, messages, elapsed, model_name)
    except RuntimeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"Report saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
