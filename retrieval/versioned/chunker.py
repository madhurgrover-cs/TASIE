"""
Split files into retrieval chunks.

Python (ast):
  - every def / async def outside a function body is a chunk: kind "function" at module
    level, "method" inside a class, with qualified names (`Session.request`,
    `Outer.Inner.method`). Nested functions stay inside their parent's chunk.
    Decorators are included in the line range.
  - a class is a chunk too: the whole class if it has no methods / nested classes,
    otherwise its header (decorators, signature, docstring and statements before the
    first def), since the methods are chunks of their own.
  - consecutive top-level statements that are not defs form "module" blocks (e.g.
    `if __name__ == "__main__":`, constants). Imports, docstrings and `pass` are left out
    and split blocks: otherwise a file's import header plus one constant becomes a large
    chunk that matches every query.
  - a file that doesn't parse (e.g. Python 2 syntax) falls back to one "file" chunk.
Other text files: one "file" chunk. Binary, empty and oversized files: no chunks.

Each chunk's `code` is normalised (newlines, trailing whitespace, common indentation,
surrounding blank lines) and `hash` is sha256 of it. The embedding cache is keyed by
that hash, so a method that is moved, re-indented or left untouched in an edited file
is never embedded twice.
"""
from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

MAX_FILE_BYTES = 512_000
# Text files that are noise for code search.
SKIP_SUFFIXES = {".svg", ".map", ".lock", ".csv", ".tsv", ".ipynb", ".pem", ".crt", ".key"}
LANGUAGES = {".py": "python", ".pyi": "python", ".md": "markdown", ".rst": "rst", ".txt": "text",
             ".toml": "toml", ".cfg": "ini", ".ini": "ini", ".yml": "yaml", ".yaml": "yaml",
             ".json": "json", ".sh": "shell", ".js": "javascript", ".ts": "typescript",
             ".html": "html", ".css": "css", ".c": "c", ".h": "c", ".cpp": "cpp", ".go": "go",
             ".rs": "rust", ".java": "java"}
_NEWLINES = re.compile(r"\r\n|\r|\n")  # what ast counts as a line break


@dataclass(frozen=True)
class Chunk:
    path: str
    name: str  # qualified name, "<module>" for top-level blocks, file name for "file" chunks
    kind: str  # function | method | class | module | file
    start_line: int  # 1-based, inclusive
    end_line: int
    code: str  # normalised
    hash: str
    language: str

    @property
    def id(self) -> str:
        return f"{self.path}#{self.name}@{self.start_line}"

    def to_dict(self) -> dict:
        return {"id": self.id, **asdict(self)}


def normalize_code(text: str) -> str:
    lines = [ln.rstrip() for ln in _NEWLINES.split(text)]
    return textwrap.dedent("\n".join(lines)).strip("\n")


def content_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def language_of(path: str) -> str:
    return LANGUAGES.get(PurePosixPath(path).suffix.lower(), "text")


def _make(path: str, name: str, kind: str, lines: list[str], start: int, end: int, language: str) -> Chunk | None:
    code = normalize_code("\n".join(lines[start - 1:end]))
    if not code.strip():
        return None
    return Chunk(path, name, kind, start, end, code, content_hash(code), language)


def _start(node: ast.AST) -> int:
    return min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])


_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _trivial(stmt: ast.stmt) -> bool:
    """Imports, docstrings and `pass` carry no search signal on their own."""
    if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass)):
        return True
    return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def chunk_python(path: str, source: str) -> list[Chunk] | None:
    """ast-based chunks, or None if the file doesn't parse."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    lines = _NEWLINES.split(source)
    lang = language_of(path)
    chunks: list[Chunk] = []

    def emit(name: str, kind: str, start: int, end: int) -> None:
        c = _make(path, name, kind, lines, start, end, lang)
        if c is not None:
            chunks.append(c)

    def visit_class(node: ast.ClassDef, prefix: str) -> None:
        qual = prefix + node.name
        inner = [n for n in node.body if isinstance(n, _DEFS)]
        header_end = _start(inner[0]) - 1 if inner else node.end_lineno
        emit(qual, "class", _start(node), max(header_end, node.lineno))
        for n in inner:
            if isinstance(n, ast.ClassDef):
                visit_class(n, qual + ".")
            else:
                emit(f"{qual}.{n.name}", "method", _start(n), n.end_lineno)

    run: list[ast.stmt] = []

    def flush_run() -> None:
        if run:
            emit("<module>", "module", _start(run[0]), run[-1].end_lineno)
        run.clear()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush_run()
            emit(node.name, "function", _start(node), node.end_lineno)
        elif isinstance(node, ast.ClassDef):
            flush_run()
            visit_class(node, "")
        elif _trivial(node):
            flush_run()
        else:
            run.append(node)
    flush_run()
    return chunks


def decode_text(data: bytes) -> str | None:
    """Text content, or None for binary / mostly-undecodable data."""
    if b"\0" in data:
        return None
    text = data.decode("utf-8", errors="replace")
    if text.count("�") > max(8, len(text) // 100):
        return None
    return text


def chunk_file(path: str, data: bytes) -> list[Chunk]:
    if len(data) > MAX_FILE_BYTES or PurePosixPath(path).suffix.lower() in SKIP_SUFFIXES:
        return []
    text = decode_text(data)
    if text is None or not text.strip():
        return []
    if language_of(path) == "python":
        chunks = chunk_python(path, text)
        if chunks is not None:
            return chunks
    n_lines = len(_NEWLINES.split(text.rstrip("\r\n")))
    c = _make(path, PurePosixPath(path).name, "file", _NEWLINES.split(text), 1, n_lines, language_of(path))
    return [c] if c else []
