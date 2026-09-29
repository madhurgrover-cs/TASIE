"""
Code-search demo (Hugging Face Space, Gradio SDK). Queries are encoded on CPU.

Two kinds of source, both precomputed on Kaggle, so the Space only encodes queries:
  - "APPS corpus": 8,765 Python solutions from MTEB AppsRetrieval
    (dataset madhurr382/apps-corpus-index, retrieval/precompute_corpus.py);
  - a git repo indexed at several commits, one SearchSource per commit
    (dataset madhurr382/repo-versions-index, retrieval/versioned/precompute_repo.py).
The model madhurr382/coderankembed-apps-ft is loaded once, at the revision recorded in
the APPS index manifest; the repo index must have been embedded with the same revision.
Queries get the same desc-io cleanup + prefix as retrieval/submission.py.

Env overrides: INDEX_REPO, INDEX_REVISION, REPO_INDEX (empty = no repo source),
REPO_INDEX_REVISION, MODEL_ID (must match the index manifests).
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search import DenseIndexSource, Hit, SearchEngine, SearchResult  # noqa: E402

logger = logging.getLogger("space")

MODEL_ID = os.environ.get("MODEL_ID", "madhurr382/coderankembed-apps-ft")
INDEX_REPO = os.environ.get("INDEX_REPO", "madhurr382/apps-corpus-index")
INDEX_REVISION = os.environ.get("INDEX_REVISION") or None
REPO_INDEX = os.environ.get("REPO_INDEX", "madhurr382/repo-versions-index")
REPO_INDEX_REVISION = os.environ.get("REPO_INDEX_REVISION") or None
APPS = "APPS corpus"
MAX_K = 10
DEFAULT_K = 5

# (short label shown in the UI, full query)
APPS_EXAMPLES = [
    ("Longest increasing subsequence", "Find the length of the longest strictly increasing subsequence of an array."),
    ("Count islands", "Given a grid of 0s and 1s, count the number of islands of connected 1s."),
    ("Stairs, mod 1e9+7", "Count the ways to climb n stairs taking 1 or 2 steps at a time, modulo 10^9+7."),
    ("Almost palindrome", "Check whether a string can become a palindrome by removing at most one character."),
    ("BFS shortest path", "Find the shortest path between two nodes in an unweighted graph."),
]
REPO_EXAMPLES = [
    ("Proxies from env", "get proxy settings from environment variables"),
    ("Auth on redirect", "strip the Authorization header when redirected to another host"),
    ("Multipart upload", "encode files for a multipart/form-data POST body"),
    ("Merge settings", "merge session-level settings with per-request settings"),
    ("Retry adapter", "mount an HTTP adapter with a maximum number of retries"),
]
CODE_LANGUAGES = {"python", "c", "cpp", "markdown", "json", "html", "css", "javascript", "typescript",
                  "yaml", "shell"}


@dataclass
class RepoInfo:
    name: str  # e.g. "psf/requests"
    versions: list[tuple[str, str]] = field(default_factory=list)  # (dropdown label, source name), newest first

    @property
    def choice(self) -> str:
        return f"{self.name} (git repo)"


def build_state() -> tuple[SearchEngine, RepoInfo | None]:
    """Download indexes + model and wire them up. Imports torch lazily."""
    from huggingface_hub import snapshot_download

    from model_loading import load_st_model
    from query_encoder import STQueryEncoder

    index_dir = snapshot_download(INDEX_REPO, repo_type="dataset", revision=INDEX_REVISION)
    apps = DenseIndexSource.from_dir(index_dir, name=APPS)
    m = apps.manifest
    if m.get("model_id") != MODEL_ID:
        raise RuntimeError(f"index {INDEX_REPO} was built with {m.get('model_id')!r}, app expects {MODEL_ID!r}; "
                           "rebuild the index or set MODEL_ID")
    sources, repo = [apps], None
    if REPO_INDEX:
        try:
            repo_sources, repo = load_repo(REPO_INDEX, REPO_INDEX_REVISION, f"{MODEL_ID}@{m.get('model_revision')}")
            sources += repo_sources
        except Exception:  # the APPS demo still works without the repo index
            logger.exception("Repo index %s not loaded", REPO_INDEX)

    # Local snapshot at the index's model revision: NomicBert's remote code ignores
    # `revision` for Hub weights, so a pinned local dir guarantees query and corpus
    # embeddings come from the same weights.
    model_dir = snapshot_download(MODEL_ID, revision=m.get("model_revision"))
    model, _ = load_st_model(model_dir, "cpu", trust_remote_code=True)
    model.max_seq_length = m.get("max_seq_length", 512)
    encoder = STQueryEncoder(model, query_prefix=m.get("query_prefix", ""), query_clean=m.get("query_clean", "desc-io"))
    engine = SearchEngine(encoder, sources)
    engine.search("warm up", k=1, sources=[APPS])  # first forward pass is slow; don't bill it to a user
    logger.info("Ready: %s | model %s @ %s", {s.name: len(s) for s in sources}, MODEL_ID, m.get("model_revision"))
    return engine, repo


def load_repo(dataset: str, revision: str | None, expected_model_id: str):
    from huggingface_hub import snapshot_download

    from versioned_source import load_repo_sources

    store, sources = load_repo_sources(snapshot_download(dataset, repo_type="dataset", revision=revision))
    if store.model_id != expected_model_id:
        raise RuntimeError(f"{dataset} was embedded with {store.model_id!r}, queries use {expected_model_id!r}; "
                           "rebuild it with the same model revision")
    return sources, repo_info(store.registry.repo or dataset, sources)


def repo_info(name: str, sources) -> RepoInfo:
    """Dropdown entries, newest commit first (sources come oldest first)."""
    return RepoInfo(name, [(s.label, s.name) for s in reversed(sources)])


# ---- rendering
def _hit_header(rank: int, hit: Hit) -> str:
    parts = [f"**#{rank}**", f"score **{hit.score:.4f}**"]
    meta = hit.meta
    if "path" in meta:  # repo chunk
        s, e = meta["start_line"], meta["end_line"]
        parts += [f"`{meta['path']}`", f"**`{meta['name']}`** ({meta['kind']})", f"L{s}–{e}" if e != s else f"L{s}"]
        if hit.url:
            parts.append(f"[GitHub](<{hit.url}>)")
    else:  # APPS solution
        parts.append(f"id `{hit.doc_id}`")
        if meta.get("partition"):
            parts.append(f"split {meta['partition']}")
        if hit.url and hit.url.startswith(("https://", "http://")):
            parts.append(f"[problem](<{hit.url}>)")
    return " · ".join(parts)


def format_result(result: SearchResult, scope: str = "") -> tuple[str, list]:
    """Status line + MAX_K * (group, header, code) updates."""
    status = (f"{len(result.hits)} results from {result.n_searched:,} snippets{scope} in "
              f"**{result.total_ms:.0f} ms** (encode {result.encode_ms:.0f} ms, search {result.search_ms:.1f} ms, CPU)")
    updates: list = []
    for i in range(MAX_K):
        if i < len(result.hits):
            hit = result.hits[i]
            lang = hit.language if hit.language in CODE_LANGUAGES else None
            updates += [gr.update(visible=True), gr.update(value=_hit_header(i + 1, hit)),
                        gr.update(value=hit.code, language=lang)]
        else:
            updates += [gr.update(visible=False), gr.update(value=""), gr.update(value="")]
    return status, updates


def _empty(message: str) -> list:
    return [message] + [gr.update(visible=False), gr.update(value=""), gr.update(value="")] * MAX_K


def create_demo(engine: SearchEngine, repo: RepoInfo | None = None) -> gr.Blocks:
    versions = dict(repo.versions) if repo else {}
    choices = [APPS] + ([repo.choice] if repo else [])

    def run(query: str, k: float, where: str, version: str | None):
        if repo and where == repo.choice:
            if version not in versions:
                return _empty("⚠️ pick a commit")
            source, scope = versions[version], f" of {repo.name} @ {version}"
        else:
            source, scope = APPS, ""
        try:
            result = engine.search(query, k=int(k), sources=[source])
        except ValueError as e:
            return _empty(f"⚠️ {e}")
        status, updates = format_result(result, scope)
        return [status] + updates

    def on_source(where: str):
        is_repo = bool(repo) and where == repo.choice
        return gr.update(visible=is_repo), gr.update(visible=not is_repo), gr.update(visible=is_repo)

    with gr.Blocks(title="Code Search") as demo:
        gr.Markdown(
            "# Natural-language → code search\n"
            "Describe what the code should do and get ranked snippets, from the "
            "[APPS](https://huggingface.co/datasets/CoIR-Retrieval/apps) solutions or from a real git repo at "
            "several points in its history. Model: "
            f"[`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) (CodeRankEmbed fine-tuned on APPS train, "
            "AppsRetrieval NDCG@10 0.4720). Queries are encoded on CPU; all code embeddings are precomputed."
        )
        with gr.Row():
            where = gr.Radio(choices, value=APPS, label="Search in", visible=len(choices) > 1, scale=2)
            version = gr.Dropdown([label for label, _ in repo.versions] if repo else [],
                                  value=repo.versions[0][0] if repo and repo.versions else None,
                                  label="Commit", visible=False, scale=2)
        with gr.Row():
            query = gr.Textbox(label="Query", placeholder="e.g. find the longest increasing subsequence",
                               lines=2, scale=5, autofocus=True)
            with gr.Column(scale=1, min_width=160):
                k = gr.Slider(1, MAX_K, value=DEFAULT_K, step=1, label="Top-k")
                btn = gr.Button("Search", variant="primary")
        with gr.Column(visible=True) as apps_examples:
            gr.Examples([q for _, q in APPS_EXAMPLES], inputs=[query], label="Examples (APPS)",
                        example_labels=[lbl for lbl, _ in APPS_EXAMPLES])
        with gr.Column(visible=False) as repo_examples:
            gr.Examples([q for _, q in REPO_EXAMPLES], inputs=[query],
                        label=f"Examples ({repo.name if repo else 'repo'})",
                        example_labels=[lbl for lbl, _ in REPO_EXAMPLES])
        status = gr.Markdown()
        outputs: list = [status]
        for _ in range(MAX_K):
            with gr.Group(visible=False) as group:
                header = gr.Markdown()
                code = gr.Code(language="python", interactive=False, show_label=False, max_lines=30)
            outputs += [group, header, code]

        where.change(on_source, [where], [version, apps_examples, repo_examples])
        btn.click(run, [query, k, where, version], outputs)
        query.submit(run, [query, k, where, version], outputs)
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    demo = create_demo(*build_state())
    demo.queue(default_concurrency_limit=2).launch(theme=gr.themes.Soft())
