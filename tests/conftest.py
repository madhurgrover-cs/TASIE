"""pytest fixtures: a throwaway git repo with three tagged commits."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tests", ROOT / "space"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CORE_V1 = '''"""Core helpers."""
import os

MAX_RETRIES = 3


def parse_header(line):
    """Parse an HTTP header line into (key, value)."""
    key, _, value = line.partition(":")
    return key.strip().lower(), value.strip()


class Session:
    """Keeps cookies between requests."""

    timeout = 10

    def __init__(self):
        self.cookies = {}

    @property
    def cookie_count(self):
        return len(self.cookies)

    def send(self, request):
        def _prepare():
            return request
        return _prepare()


async def fetch_json(url):
    return {"url": url}


if __name__ == "__main__":
    print(parse_header("Accept: text/html"))
'''

# v2: parse_header changes; everything else in core.py is identical
CORE_V2 = CORE_V1.replace(
    '''    key, _, value = line.partition(":")
    return key.strip().lower(), value.strip()''',
    '''    if ":" not in line:
        raise ValueError("malformed header line")
    key, _, value = line.partition(":")
    return key.strip().lower(), value.strip()''')

RETRY = '''def backoff_delay(attempt, base=0.5):
    """Exponential backoff delay for a retry attempt."""
    return base * (2 ** attempt)
'''

SESSION_BLOCK = CORE_V2[CORE_V2.index("class Session:"):CORE_V2.index("async def fetch_json")]
# v3: Session moves, unchanged, into its own module
CORE_V3 = CORE_V2.replace(SESSION_BLOCK, "")
SESSION_V3 = '"""Session object."""\nfrom pkg.core import parse_header\n\n\n' + SESSION_BLOCK.rstrip() + "\n"


def _git(repo: Path, *args: str, date: str | None = None) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    if date:
        env.update(GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
    return subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
                          check=True, capture_output=True, text=True, env=env).stdout.strip()


def _write(repo: Path, rel: str, data: str | bytes) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())


@pytest.fixture(scope="session")
def git_repo(tmp_path_factory) -> dict:
    repo = tmp_path_factory.mktemp("repo")
    _git(repo, "init", "-q", "-b", "main")

    _write(repo, "pkg/__init__.py", "from pkg.core import parse_header\n")
    _write(repo, "pkg/core.py", CORE_V1)
    _write(repo, "legacy.py", 'print "python 2 only"\n')
    _write(repo, "README.md", "# demo\n\nA tiny HTTP helper library.\n")
    _write(repo, "logo.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00binary")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v1", date="2020-01-01T00:00:00+00:00")
    _git(repo, "tag", "v1")

    _write(repo, "pkg/core.py", CORE_V2)
    _write(repo, "pkg/retry.py", RETRY)
    (repo / "README.md").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v2", date="2021-01-01T00:00:00+00:00")
    _git(repo, "tag", "v2")

    _write(repo, "pkg/core.py", CORE_V3)
    _write(repo, "pkg/session.py", SESSION_V3)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v3", date="2022-01-01T00:00:00+00:00")
    _git(repo, "tag", "v3")

    shas = {t: _git(repo, "rev-parse", f"{t}^{{commit}}") for t in ("v1", "v2", "v3")}
    return {"path": repo, "shas": shas}
