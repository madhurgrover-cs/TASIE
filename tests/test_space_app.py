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

    def radios(self, demo):
        return {b.label: b for b in demo.blocks.values() if isinstance(b, gradio.Radio)}

    def test_demo_builds_without_repo_or_gpu(self):
        demo = app.create_demo(self.engine)
        self.assertIsInstance(demo, gradio.Blocks)
        radios = self.radios(demo)
        self.assertFalse(radios["Search in"].visible)  # nothing to pick from
        self.assertFalse(radios["Query encoding"].visible)  # no GPU engine off ZeroGPU

    def test_gpu_option_cpu_default(self):
        gpu_enc = FakeEncoder()
        gpu = SearchEngine(gpu_enc, [make_source(app.APPS)])
        demo = app.create_demo(self.engine, None, gpu)
        radio = self.radios(demo)["Query encoding"]
        self.assertTrue(radio.visible)
        self.assertEqual(radio.value, app.CPU)
        self.assertEqual([c[0] for c in radio.choices], ["CPU", "GPU (optional)"])
        status, _ = app.format_result(gpu.search("gcd", k=1), device=app.GPU)
        self.assertIn("GPU, incl. allocation", status)
        self.assertIn(", CPU)", app.format_result(self.engine.search("gcd", k=1))[0])

    def test_zerogpu_function_registered_only_on_zerogpu(self):
        src = open(app.__file__, encoding="utf-8").read()
        self.assertIn("spaces.GPU(", src)  # ZeroGPU refuses to start without one
        self.assertFalse(app.ON_ZEROGPU)  # not set locally: no `spaces` import, plain function

    def test_example_labels_are_short(self):
        for label, query in app.APPS_EXAMPLES + app.REPO_EXAMPLES:
            self.assertLessEqual(len(label), 32, label)
            self.assertLess(len(label), len(query))

    def test_repo_source_picker(self):
        engine = SearchEngine(FakeEncoder(), [make_source(app.APPS), OtherSource("org/repo@abc1234", 0.99)])
        repo = app.RepoInfo("org/repo", [("v1 · abc1234", "org/repo@abc1234")])
        demo = app.create_demo(engine, repo)
        where = self.radios(demo)["Search in"]
        self.assertTrue(where.visible)
        self.assertEqual([c[0] for c in where.choices], [app.APPS, "org/repo (git repo)"])
        dropdowns = [b for b in demo.blocks.values() if isinstance(b, gradio.Dropdown)]
        self.assertEqual(dropdowns[0].value, "v1 · abc1234")


if __name__ == "__main__":
    unittest.main()
