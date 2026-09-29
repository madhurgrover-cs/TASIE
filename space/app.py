"""
Code-search demo (Hugging Face Space, Gradio SDK; CPU inference, optional ZeroGPU).

Natural-language query -> ranked Python solutions from the AppsRetrieval corpus, using
madhurr382/coderankembed-apps-ft (CodeRankEmbed fine-tuned on APPS train).

At startup:
  1. download the precomputed corpus index (dataset repo, see retrieval/precompute_corpus.py);
  2. download the model at the exact revision recorded in the index manifest, and load
     it on CPU with load_st_model (fp32 + transformers v5 buffer fix);
  3. encode queries like retrieval/submission.py: desc-io cleanup + query prefix.
No corpus embedding happens on the Space.

Hardware: queries are encoded on CPU by default (the benchmark setting). On ZeroGPU
hardware (SPACES_ZERO_GPU=true) the platform requires at least one @spaces.GPU function,
so a second copy of the model is moved to CUDA and a "ZeroGPU" device option encodes
the query inside a @spaces.GPU call. Its latency includes GPU allocation. Elsewhere
(cpu-basic, local, tests) there is no `spaces` import and no device option.

Env overrides: INDEX_REPO, INDEX_REVISION, MODEL_ID (must match the index manifest).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ON_ZEROGPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true")
if ON_ZEROGPU:
    import spaces  # must be imported before torch / any CUDA package

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import MANIFEST_FILE, DenseIndexSource, SearchEngine, SearchResult  # noqa: E402

logger = logging.getLogger("space")

MODEL_ID = os.environ.get("MODEL_ID", "madhurr382/coderankembed-apps-ft")
INDEX_REPO = os.environ.get("INDEX_REPO", "madhurr382/apps-corpus-index")
INDEX_REVISION = os.environ.get("INDEX_REVISION") or None
SOURCE_NAME = "APPS corpus"
MAX_K = 10
DEFAULT_K = 5

EXAMPLES = [
    "Find the length of the longest increasing subsequence of an array.",
    "Count the number of ways to climb n stairs taking 1 or 2 steps at a time, modulo 10^9+7.",
    "Given a grid of 0s and 1s, count the number of islands of connected 1s.",
    "Check whether a string is a palindrome after removing at most one character.",
    "Find the shortest path between two nodes in an unweighted graph using BFS.",
    "Compute the greatest common divisor of all numbers in a list.",
]

CPU, ZEROGPU = "CPU", "ZeroGPU"
_gpu_encoder = None  # STQueryEncoder on CUDA, set by build_engines() on ZeroGPU


def _encode_on_gpu(text: str):
    return _gpu_encoder.encode_query(text)


if ON_ZEROGPU:
    # Defined at import time so ZeroGPU registers it during startup.
    _encode_on_gpu = spaces.GPU(duration=20)(_encode_on_gpu)


class ZeroGPUEncoder:
    """QueryEncoder that runs each query inside a @spaces.GPU call."""

    def encode_query(self, text: str):
        return _encode_on_gpu(text)


def build_engines() -> dict[str, SearchEngine]:
    """Download index + model and wire them up ({CPU: ...} plus {ZeroGPU: ...} on ZeroGPU).
    Imports torch lazily."""
    global _gpu_encoder
    from huggingface_hub import snapshot_download

    from model_loading import load_st_model
    from query_encoder import STQueryEncoder

    index_dir = snapshot_download(INDEX_REPO, repo_type="dataset", revision=INDEX_REVISION)
    source = DenseIndexSource.from_dir(index_dir, name=SOURCE_NAME)
    m = source.manifest
    if m.get("model_id") != MODEL_ID:
        raise RuntimeError(f"index {INDEX_REPO} was built with {m.get('model_id')!r}, app expects {MODEL_ID!r}; "
                           "rebuild the index or set MODEL_ID")
    # Local snapshot at the index's model revision: NomicBert's remote code ignores
    # `revision` for Hub weights, so loading from a pinned local dir is the only way
    # to guarantee query and corpus embeddings come from the same weights.
    model_dir = snapshot_download(MODEL_ID, revision=m.get("model_revision"))
    model, _ = load_st_model(model_dir, "cpu", trust_remote_code=True)
    model.max_seq_length = m.get("max_seq_length", 512)
    prep = {"query_prefix": m.get("query_prefix", ""), "query_clean": m.get("query_clean", "desc-io")}
    engines = {CPU: SearchEngine(STQueryEncoder(model, **prep), [source])}
    engines[CPU].search("warm up", k=1)  # first forward pass is slow; don't bill it to the first user
    if ON_ZEROGPU:
        gpu_model, _ = load_st_model(model_dir, "cpu", trust_remote_code=True)  # buffers fixed on CPU first
        gpu_model.max_seq_length = model.max_seq_length
        gpu_model.to("cuda")  # ZeroGPU: tensors move to the GPU when a @spaces.GPU call starts
        _gpu_encoder = STQueryEncoder(gpu_model, **prep)
        engines[ZEROGPU] = SearchEngine(ZeroGPUEncoder(), [source])
    logger.info("Ready: %d docs (dim %d), model %s @ %s, devices %s", len(source), source.dim, MODEL_ID,
                m.get("model_revision"), list(engines))
    return engines


def _hit_header(rank: int, hit) -> str:
    parts = [f"**#{rank}**", f"score **{hit.score:.4f}**", f"id `{hit.doc_id}`"]
    if hit.meta.get("partition"):
        parts.append(f"split {hit.meta['partition']}")
    if hit.url and hit.url.startswith(("https://", "http://")):
        parts.append(f"[problem](<{hit.url}>)")
    return " · ".join(parts)


def format_result(result: SearchResult, show_source: bool) -> tuple[str, list]:
    """Status line + MAX_K * (group, header, code) updates."""
    status = (f"{len(result.hits)} results from {result.n_searched:,} snippets in "
              f"**{result.total_ms:.0f} ms** (encode {result.encode_ms:.0f} ms, search {result.search_ms:.1f} ms)")
    updates: list = []
    for i in range(MAX_K):
        if i < len(result.hits):
            hit = result.hits[i]
            header = _hit_header(i + 1, hit) + (f" · {hit.source}" if show_source else "")
            updates += [gr.update(visible=True), gr.update(value=header), gr.update(value=hit.code)]
        else:
            updates += [gr.update(visible=False), gr.update(value=""), gr.update(value="")]
    return status, updates


def create_demo(engines: dict[str, SearchEngine] | SearchEngine) -> gr.Blocks:
    if isinstance(engines, SearchEngine):
        engines = {CPU: engines}
    engine = next(iter(engines.values()))
    multi_source = len(engine.source_names) > 1

    def run(query: str, k: float, sources: list[str] | None, device: str | None = None):
        try:
            result = engines.get(device or CPU, engine).search(query, k=int(k), sources=sources or None)
        except ValueError as e:
            empty = [gr.update(visible=False), gr.update(value=""), gr.update(value="")] * MAX_K
            return [f"⚠️ {e}"] + empty
        status, updates = format_result(result, show_source=multi_source)
        return [status] + updates

    with gr.Blocks(title="Code Search") as demo:
        gr.Markdown(
            "# Natural-language → code search\n"
            "Describe a programming problem and get ranked Python solutions from the "
            "[APPS](https://huggingface.co/datasets/CoIR-Retrieval/apps) corpus. Model: "
            f"[`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) (CodeRankEmbed fine-tuned on APPS train, "
            "AppsRetrieval NDCG@10 0.4720). Queries are encoded on CPU by default; corpus embeddings "
            "are precomputed."
        )
        with gr.Row():
            query = gr.Textbox(label="Query", placeholder="e.g. find the longest increasing subsequence",
                               lines=2, scale=5, autofocus=True)
            with gr.Column(scale=1, min_width=160):
                k = gr.Slider(1, MAX_K, value=DEFAULT_K, step=1, label="Top-k")
                btn = gr.Button("Search", variant="primary")
        with gr.Row():
            sources = gr.CheckboxGroup(engine.source_names, value=engine.source_names, label="Sources",
                                       visible=multi_source)
            device = gr.Radio(list(engines), value=CPU, label="Query encoding device",
                              info="ZeroGPU latency includes GPU allocation (first call is slowest)",
                              visible=len(engines) > 1)
        gr.Examples(EXAMPLES, inputs=[query], label="Example queries")
        status = gr.Markdown()
        outputs: list = [status]
        for _ in range(MAX_K):
            with gr.Group(visible=False) as group:
                header = gr.Markdown()
                code = gr.Code(language="python", interactive=False, show_label=False, max_lines=30)
            outputs += [group, header, code]

        btn.click(run, [query, k, sources, device], outputs)
        query.submit(run, [query, k, sources, device], outputs)
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    demo = create_demo(build_engines())
    demo.queue(default_concurrency_limit=2).launch(theme=gr.themes.Soft())
