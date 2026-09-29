"""
Query cleanup for APPS problem statements (--query-clean).

APPS queries are full problem statements. Codeforces / CodeChef / AtCoder use
"-----Input-----" style headers, HackerRank uses "=====Input Format=====", and
LeetCode-style statements put bare "Example 1:" / "Note:" / "Constraints:" lines
between sections. Everything before the first header is the description.

Modes:
    none     original text
    desc     description only (text before the first section header)
    desc-io  description + Input / Output spec sections; examples, samples, notes,
             explanations, constraints, scoring etc. are dropped

Survey of 900 test-split queries (2026-09-28): 886 have Input/Output headers;
others are LeetCode-style (bare "Example:" lines), HackerRank ("=====...=====")
or a handful with no headers at all, which are left as-is. A few are in
Russian ("Входные данные" / "Выходные данные").

Pure Python: no torch / mteb imports, so it can be tested on the dev laptop.
"""
from __future__ import annotations

import re

QUERY_CLEAN_MODES = ("none", "desc", "desc-io")

# "-----Input-----", "-----Sample Input 1:-----", "=====Output Format====="
_DASHED_HEADER = re.compile(r"^[ \t]*(?:-{3,}|={3,})[ \t]*(.+?)[ \t]*(?:-{3,}|={3,})[ \t]*$", re.M)
# LeetCode style: a line that is only "Example 1:", "Examples", "Note:", "Constraints:", "Follow up:"
_BARE_HEADER = re.compile(
    r"^[ \t]*(examples?(?:[ \t]*\d+)?|notes?|constraints|follow[ \t-]?up|explanation)[ \t]*:?[ \t]*$",
    re.M | re.I,
)

_DESC_NAMES = {"problem statement", "statement", "problem", "legend"}
_IO_NAMES = {
    "input", "output", "inputs", "outputs", "input format", "output format",
    "входные данные", "выходные данные",
}


def _norm(name: str) -> str:
    name = re.sub(r"[\s:;.]+$", "", name.strip().lower())
    return re.sub(r"\s+", " ", name)


def _sections(text: str) -> list[tuple[str | None, str]]:
    """Split into (normalised header or None for the leading text, body)."""
    headers = sorted(
        [(m.start(), m.end(), _norm(m.group(1))) for m in _DASHED_HEADER.finditer(text)]
        + [(m.start(), m.end(), _norm(m.group(1))) for m in _BARE_HEADER.finditer(text)]
    )
    out: list[tuple[str | None, str]] = []
    pos, name = 0, None
    for start, end, hname in headers:
        if start < pos:  # overlapping match (a dashed line can't also be bare, but be safe)
            continue
        out.append((name, text[pos:start]))
        pos, name = end, hname
    out.append((name, text[pos:]))
    return out


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_query(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode not in QUERY_CLEAN_MODES:
        raise ValueError(f"Unknown query-clean mode {mode!r}; expected one of {QUERY_CLEAN_MODES}")

    parts: list[str] = []
    for name, body in _sections(text):
        body = _tidy(body)
        if not body:
            continue
        if name is None or name in _DESC_NAMES:
            parts.append(body)
        elif mode == "desc-io" and name in _IO_NAMES:
            parts.append(f"{name.capitalize()}:\n{body}")
    cleaned = "\n\n".join(parts)
    # Statement that starts with a dropped section: keep the original rather than
    # sending an empty query.
    return cleaned or text
