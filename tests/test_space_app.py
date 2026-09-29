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
        self.engine = SearchEngine(FakeEncoder(), [make_source("APPS corpus")])

    def test_format_result(self):
        res = self.engine.search("count islands in a grid", k=3)
        status, updates = app.format_result(res, show_source=False)
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
        self.assertFalse(updates[3 * 3]["visible"])  # slot 4 hidden

    def test_demo_builds(self):
        demo = app.create_demo(self.engine)
        self.assertIsInstance(demo, gradio.Blocks)

    def test_device_option(self):
        gpu_enc = FakeEncoder()
        engines = {app.CPU: self.engine, app.ZEROGPU: SearchEngine(gpu_enc, [make_source("APPS corpus")])}
        demo = app.create_demo(engines)
        radios = [b for b in demo.blocks.values() if isinstance(b, gradio.Radio)]
        self.assertEqual(len(radios), 1)
        self.assertTrue(radios[0].visible)
        self.assertEqual(radios[0].choices, [(app.CPU, app.CPU), (app.ZEROGPU, app.ZEROGPU)])

    def test_single_engine_hides_device(self):
        demo = app.create_demo(self.engine)
        radios = [b for b in demo.blocks.values() if isinstance(b, gradio.Radio)]
        self.assertFalse(radios[0].visible)

    def test_multi_source_label(self):
        engine = SearchEngine(FakeEncoder(), [make_source("APPS corpus"), OtherSource("repo@v1", 0.99)])
        _, updates = app.format_result(engine.search("gcd", k=2), show_source=True)
        self.assertIn("repo@v1", updates[1]["value"])


if __name__ == "__main__":
    unittest.main()
