import git
from pathlib import Path

# Git's well-known empty tree object — always resolvable in any repo, so it works
# for the first-commit case without shelling out to a platform path like /dev/null.
EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

EXCLUDED_DIRS = {
    "venv", ".venv", "env", ".env",
    "node_modules",
    ".git",
    "__pycache__",
    "dist", "build",
    "site-packages",
    ".tox", ".mypy_cache", ".pytest_cache",
}


def _is_excluded(file_path: str) -> bool:
    parts = Path(file_path).parts
    return any(part in EXCLUDED_DIRS for part in parts)


def get_diff_chunks(repo_path: str) -> list[dict]:
    """
    Returns a list of added/modified code chunks from the latest commit diff.
    Each chunk is a dict: { file_path, code, start_line }
    """
    repo = git.Repo(repo_path)
    chunks = []

    if len(repo.heads) == 0:
        return chunks  # No commits yet

    head_commit = repo.head.commit
    parents = head_commit.parents
    if parents:
        # Diff parent -> HEAD so lines *added* by the latest commit show up with
        # a '+' prefix in the patch (the direction the parser below expects).
        base = parents[0]
    else:
        # First commit: diff the empty tree -> HEAD (everything counts as added).
        base = repo.tree(EMPTY_TREE_SHA)

    diff = base.diff(head_commit, create_patch=True)

    for item in diff:
        if item.b_blob is None:
            continue  # skip pure deletions
        patch = item.diff.decode("utf-8", errors="ignore")
        file_path = item.b_path

        if _is_excluded(file_path):
            continue

        # Extract added/modified line blocks
        current_block = []
        start_line = 0
        line_number = 0

        for line in patch.splitlines():
            if line.startswith("@@"):
                # Flush previous block
                if current_block:
                    chunks.append({
                        "file_path": file_path,
                        "code": "\n".join(current_block),
                        "start_line": start_line,
                    })
                    current_block = []
                # Parse start line from hunk header e.g. @@ -1,4 +3,8 @@
                try:
                    after = line.split("+")[1].split(",")[0].split("@@")[0].strip()
                    start_line = int(after)
                    line_number = start_line
                except Exception:
                    start_line = 0
            elif line.startswith("+") and not line.startswith("+++"):
                current_block.append(line[1:])
                line_number += 1
            elif not line.startswith("-"):
                line_number += 1

        if current_block:
            chunks.append({
                "file_path": file_path,
                "code": "\n".join(current_block),
                "start_line": start_line,
            })

    return chunks


def get_all_files(repo_path: str, extensions: list[str] | None = None) -> list[dict]:
    """
    Returns all file contents in the repo (used for full scan mode).
    """
    extensions = extensions or [".py", ".js", ".ts", ".java", ".go", ".php", ".rb"]
    results = []
    for path in Path(repo_path).rglob("*"):
        if path.is_file() and path.suffix in extensions:
            rel = str(path.relative_to(repo_path))
            if _is_excluded(rel):
                continue
            try:
                code = path.read_text(errors="ignore")
                results.append({"file_path": rel, "code": code, "start_line": 1})
            except Exception:
                pass
    return results
