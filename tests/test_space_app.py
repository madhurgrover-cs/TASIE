"""Gradio wiring of space/app.py with a fake engine. Skipped when gradio isn't installed."""
from __future__ import annotations

import unittest

try:
    import gradio  # noqa: F401
except ImportError:
    gradio = None

from fakes import FakeEncoder
from test_space_search import OtherSource, make_source

if gradio is not None:
    import app
    from search import SearchEngine


@unittest.skipIf(gradio is None, "gradio not installed")
class AppTest(unittest.TestCase):
    def setUp(self):
        self.engine = SearchEngine(FakeEncoder(), [make_source(app.APPS)])

    def test_format_result(self):
        res = self.engine.search("count islands in a grid", k=3)
        status, updates = app.format_result(res)
        self.assertEqual(len(updates), 3 * app.MAX_K)
        self.assertIn("3 results", status)
        self.assertIn(" ms", status)
        group, header, code = updates[:3]
        self.assertTrue(group["visible"])
        self.assertIn("**#1**", header["value"])
        self.assertIn("id `d2`", header["value"])
        self.assertIn("score **", header["value"])
        self.assertIn("[problem](<https://example.com/d2>)", header["value"])
        self.assertIn("count_islands", code["value"])
        self.assertEqual(code["language"], "python")
        self.assertFalse(updates[3 * 3]["visible"])  # slot 4 hidden

    def test_demo_builds_without_repo(self):
        demo = app.create_demo(self.engine)
        self.assertIsInstance(demo, gradio.Blocks)
        radios = [b for b in demo.blocks.values() if isinstance(b, gradio.Radio)]
        self.assertEqual(len(radios), 1)
        self.assertFalse(radios[0].visible)  # nothing to pick from

    def test_no_zerogpu(self):
        src = open(app.__file__, encoding="utf-8").read()
        self.assertNotIn("spaces.GPU", src)
        self.assertNotIn("ZeroGPU", src)

    def test_example_labels_are_short(self):
        for label, query in app.APPS_EXAMPLES + app.REPO_EXAMPLES:
            self.assertLessEqual(len(label), 32, label)
            self.assertLess(len(label), len(query))

    def test_repo_source_picker(self):
        engine = SearchEngine(FakeEncoder(), [make_source(app.APPS), OtherSource("org/repo@abc1234", 0.99)])
        repo = app.RepoInfo("org/repo", [("v1 · abc1234", "org/repo@abc1234")])
        demo = app.create_demo(engine, repo)
        radios = [b for b in demo.blocks.values() if isinstance(b, gradio.Radio)]
        self.assertTrue(radios[0].visible)
        self.assertEqual([c[0] for c in radios[0].choices], [app.APPS, "org/repo (git repo)"])
        dropdowns = [b for b in demo.blocks.values() if isinstance(b, gradio.Dropdown)]
        self.assertEqual(dropdowns[0].value, "v1 · abc1234")


if __name__ == "__main__":
    unittest.main()
