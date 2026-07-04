#!/usr/bin/env python3
"""SessionEnd hook: auto-record.

When CEREBRO_AUTORECORD=N (a positive int) is set, re-summarize up to N files whose
cached summary went STALE during the session — so the NEXT session reuses fresh
traces instead of re-reading the files you changed. Detached (never blocks exit)
and OFF by default (it spends a little via `claude -p`; set the env var to opt in).
Mirrors the spawn pattern of session_start.py's autosummarize.

Crucially it re-summarizes in EVERY brain the session touched, not just the cwd's.
post_edit.py reindexes each edit into the brain nearest the edited file, so in a
cross-repo or multi-root workspace the stale summaries land in brains OTHER than the
cwd's — a cwd-only auto-record would silently skip exactly where the work happened.
We read the session transcript to find every edited file, resolve each to its brain,
union that with the cwd brain, and re-summarize stale in each distinct brain.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def find_brain_root(start: str):
    p = Path(start).resolve()
    p = p if p.is_dir() else p.parent
    for d in (p, *p.parents):
        if (d / ".cerebro" / "brain.db").exists():
            return d
    return None


def edited_files(transcript_path: str) -> list[str]:
    """Absolute paths of every file touched by Edit/Write/MultiEdit this session,
    parsed from the transcript JSONL. Best-effort: [] if the file is missing or a
    line is malformed. A cheap substring pre-filter avoids json.loads on the many
    lines that carry no tool call, keeping a big transcript fast to scan."""
    out: list[str] = []
    try:
        with open(transcript_path, encoding="utf-8") as fh:
            for line in fh:
                if '"tool_use"' not in line or "file_path" not in line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                content = (obj.get("message") or {}).get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "tool_use"
                        and item.get("name") in ("Edit", "Write", "MultiEdit")
                    ):
                        fp = (item.get("input") or {}).get("file_path")
                        if fp:
                            out.append(fp)
    except OSError:
        pass
    return out


def touched_roots(cwd: str, transcript_path: str | None) -> list[Path]:
    """Every distinct Cerebro brain root the session touched: the cwd's brain, plus
    the brain nearest each edited file. Deduped by resolved path, preserving the cwd
    brain first so the common single-repo case is unchanged."""
    roots: dict[str, Path] = {}
    cwd_root = find_brain_root(cwd)
    if cwd_root is not None:
        roots[str(cwd_root)] = cwd_root
    if transcript_path:
        for fp in edited_files(transcript_path):
            r = find_brain_root(fp)
            if r is not None:
                roots.setdefault(str(r), r)
    return list(roots.values())


def main() -> None:
    n = os.environ.get("CEREBRO_AUTORECORD", "").strip()
    if not (n.isdigit() and int(n) > 0):
        return  # opt-in: set CEREBRO_AUTORECORD=N to enable

    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}
    cwd = data.get("cwd") or os.getcwd()
    roots = touched_roots(cwd, data.get("transcript_path"))
    if not roots:
        return  # not a Cerebro project

    home = os.environ.get("CEREBRO_HOME")
    uv = os.environ.get("CEREBRO_UV", "uv")
    for root in roots:
        env = {**os.environ, "CEREBRO_ROOT": str(root)}
        cmd = (
            [uv, "run", "--directory", home, "cerebro", "summarize", "--stale", "--limit", n]
            if home
            else ["cerebro", "summarize", "--stale", "--limit", n]
        )
        try:
            subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            pass  # best-effort; never block session exit


if __name__ == "__main__":
    main()
