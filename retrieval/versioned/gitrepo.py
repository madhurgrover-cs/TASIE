"""
Read-only git access through the `git` binary (no GitPython, no working-tree checkout).

Blobs are read with one long-lived `git cat-file --batch` process, so reading every
file of a commit costs one subprocess, not one per file.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class GitError(RuntimeError):
    pass


@dataclass(frozen=True)
class TreeEntry:
    path: str
    blob: str  # blob sha
    mode: str


@dataclass(frozen=True)
class FileChange:
    status: str  # A, M, D, T (renames are reported as D + A: --no-renames)
    path: str


class GitRepo:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._batch: subprocess.Popen | None = None

    # ---- plumbing
    def git(self, *args: str) -> str:
        return self._git_bytes(*args).decode("utf-8", errors="replace")

    def _git_bytes(self, *args: str) -> bytes:
        r = subprocess.run(["git", "-C", str(self.path), *args], capture_output=True)
        if r.returncode != 0:
            raise GitError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
        return r.stdout

    def resolve(self, ref: str) -> str:
        """Full commit sha for a ref / tag / short sha (annotated tags are peeled)."""
        return self.git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").strip()

    def commit_info(self, commit: str) -> dict[str, str | None]:
        out = self.git("show", "-s", "--format=%H%x00%P%x00%cI%x00%s", commit).strip()
        sha, parents, date, subject = out.split("\x00", 3)
        return {"commit": sha, "git_parent": parents.split()[0] if parents else None,
                "committed_at": date, "subject": subject}

    def ls_tree(self, commit: str) -> list[TreeEntry]:
        """Regular files at `commit` (symlinks and submodules skipped)."""
        entries = []
        for rec in self._git_bytes("ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
            if not rec:
                continue
            meta, path = rec.split(b"\t", 1)
            mode, typ, sha = meta.decode().split()
            if typ == "blob" and mode in ("100644", "100755"):
                entries.append(TreeEntry(path.decode("utf-8", errors="replace"), sha, mode))
        return entries

    def diff(self, base: str, commit: str) -> list[FileChange]:
        """Files that differ between two commits (any two, not only parent/child)."""
        out = self._git_bytes("diff", "--name-status", "--no-renames", "-z", base, commit)
        parts = [p.decode("utf-8", errors="replace") for p in out.split(b"\0") if p]
        return [FileChange(parts[i][0], parts[i + 1]) for i in range(0, len(parts), 2)]

    def commits_between(self, base: str, commit: str) -> int:
        return int(self.git("rev-list", "--count", f"{base}..{commit}").strip())

    def is_ancestor(self, a: str, b: str) -> bool:
        r = subprocess.run(["git", "-C", str(self.path), "merge-base", "--is-ancestor", a, b],
                           capture_output=True)
        return r.returncode == 0

    # ---- blobs
    def read_blob(self, blob: str) -> bytes:
        if self._batch is None:
            self._batch = subprocess.Popen(["git", "-C", str(self.path), "cat-file", "--batch"],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        assert self._batch.stdin and self._batch.stdout
        self._batch.stdin.write(blob.encode() + b"\n")
        self._batch.stdin.flush()
        header = self._batch.stdout.readline().split()
        if len(header) < 3 or header[1] == b"missing":
            raise GitError(f"missing blob {blob}")
        data = self._batch.stdout.read(int(header[2]))
        self._batch.stdout.read(1)  # trailing newline
        return data

    def close(self) -> None:
        if self._batch is not None:
            if self._batch.stdin:
                self._batch.stdin.close()
            self._batch.wait()
            self._batch = None

    def __enter__(self) -> "GitRepo":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
