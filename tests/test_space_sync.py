"""space/ must stay self-contained: its copies of retrieval modules must match the originals."""
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# space/<name> <- retrieval/<source>
COPIED = {
    "query_clean.py": "query_clean.py",
    "model_loading.py": "model_loading.py",
    "versioned_store.py": "versioned/store.py",
    "versioned_evolution.py": "versioned/evolution.py",
}


class SpaceCopiesInSync(unittest.TestCase):
    def test_copies_match(self):
        for name, source in COPIED.items():
            with self.subTest(name=name):
                src = (ROOT / "retrieval" / source).read_text(encoding="utf-8").replace("\r\n", "\n")
                dst = (ROOT / "space" / name).read_text(encoding="utf-8").replace("\r\n", "\n")
                self.assertEqual(src, dst, f"space/{name} differs from retrieval/{source}; "
                                           f"copy it: cp retrieval/{source} space/{name}")

    def test_space_has_no_repo_imports(self):
        for f in (ROOT / "space").glob("*.py"):
            with self.subTest(file=f.name):
                self.assertNotIn("from retrieval", f.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
