"""Batch summary generation (plan layer 2, warmed proactively).

Generates English summaries for the most central files so even a first-time query
is cheap, instead of waiting for sessions to fill them in lazily. Uses headless
`claude -p` so it needs no API key — it rides the user's existing Claude Code auth.
A cheap model (Haiku by default) keeps the one-time cost low.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import shutil
import subprocess

from . import config as cfg
from . import db, graph, summaries

INSTRUCTION = (
    "You are summarizing a source file for a code-navigation index. In 1-3 dense "
    "sentences, in English, describe what this file does and its role in the system "
    "(key responsibilities, important types/functions, how it fits in). Output ONLY "
    "the summary text — no preamble, no markdown, no bullet points."
)
MAX_CHARS = 16000
DEFAULT_MODEL = "claude-haiku-4-5"


def _claude_bin() -> str:
    return os.environ.get("CEREBRO_CLAUDE") or shutil.which("claude") or "claude"


# Headless `claude -p` normally prints just the summary, but a misbehaving hook in
# the user's environment (e.g. a plugin's UserPromptSubmit hook failing) can make it
# emit a block/permission notice on stdout while still exiting 0 — which would then be
# recorded verbatim as the file's "summary", poisoning the brain. This actually
# happened: claude-mem's session-init hook wrote "operation blocked by hook … worker
# unreachable" into three real summaries. Reject anything that smells like a
# hook/permission notice instead of prose.
_HOOK_NOISE = (
    "blocked by hook",
    "operation blocked",
    "permission denied",
    "worker unreachable",
    "export path=",
)


def _is_valid_summary(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return not any(marker in low for marker in _HOOK_NOISE)


def summarize_one(config, rel: str, model: str) -> str | None:
    """Generate a summary for one file via `claude -p`. Returns None on failure."""
    abs_path = config.root / rel
    try:
        content = abs_path.read_text(encoding="utf-8", errors="ignore")[:MAX_CHARS]
    except OSError:
        return None
    prompt = f"{INSTRUCTION}\n\nFile path: {rel}\n\n```\n{content}\n```\n"
    try:
        out = subprocess.run(
            [_claude_bin(), "-p", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    summary = out.stdout.strip()
    return summary if _is_valid_summary(summary) else None


def select_central_missing(conn, limit: int, prefix: str | None = None) -> list[str]:
    """Top files by dependency centrality that have no summary yet."""
    have = {r["path"] for r in conn.execute("SELECT path FROM summaries")}
    out = []
    for path, _score in graph.rank(conn):
        if path in have or cfg.Config.lang_for(path) is None:
            continue
        if prefix and not path.startswith(prefix):
            continue
        out.append(path)
        if len(out) >= limit:
            break
    return out


def select_stale(conn, limit: int, prefix: str | None = None) -> list[str]:
    """Files whose cached summary went STALE (source changed since it was written).
    Re-warming these is the 'auto-record' path: edits during a session leave their
    summaries outdated, and select_central_missing skips them (they HAVE a summary).
    Bounded by `limit`."""
    out = []
    for path in summaries.stale_summaries(conn):
        if cfg.Config.lang_for(path) is None:
            continue
        if prefix and not path.startswith(prefix):
            continue
        out.append(path)
        if len(out) >= limit:
            break
    return out


def run(config, conn, rels: list[str], model: str = DEFAULT_MODEL, workers: int = 4) -> dict:
    """Summarize files in parallel (claude -p subprocesses), then record serially
    (one sqlite writer). Returns a count of what was produced."""
    produced: dict[str, str] = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(summarize_one, config, r, model): r for r in rels}
        for fut in cf.as_completed(futs):
            summary = fut.result()
            if summary:
                produced[futs[fut]] = summary
    for rel, summary in produced.items():
        summaries.record(conn, rel, summary, model=model)
    return {"requested": len(rels), "summarized": len(produced)}


def main():  # `cerebro-summarize` entry point
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Pre-generate Cerebro summaries via claude -p")
    ap.add_argument("--limit", type=int, default=20, help="max files to summarize")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prefix", default=None, help="only files under this path prefix")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stale", action="store_true",
                    help="re-summarize summaries gone stale (file changed), then fill with missing")
    args = ap.parse_args()

    config = cfg.Config.load()
    conn = db.connect(config.db_path)
    if args.stale:
        rels = select_stale(conn, args.limit, args.prefix)
        seen = set(rels)
        rels += [r for r in select_central_missing(conn, args.limit - len(rels), args.prefix)
                 if r not in seen]
    else:
        rels = select_central_missing(conn, args.limit, args.prefix)
    if not rels:
        print(json.dumps({"summarized": 0, "note": "nothing missing in scope"}))
        return
    result = run(config, conn, rels, model=args.model, workers=args.workers)
    result["model"] = args.model
    result["root"] = str(config.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
