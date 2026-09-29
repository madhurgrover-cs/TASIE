"""
Retrieval across versions of a codebase (hackathon P1).

    gitrepo.py    git plumbing via subprocess (ls-tree, cat-file --batch, diff --name-status)
    chunker.py    function/class-level chunks with ast, file-level fallback, normalised content hash
    store.py      on-disk store: registry, per-commit indexes, content + embedding caches
                  (numpy/json only; copied verbatim to space/versioned_store.py)
    embedder.py   Embedder protocol, HashingEmbedder (no torch), STEmbedder (CodeRankEmbed)
    builder.py    full and incremental (git-diff based) index builds
    searcher.py   search(query, commit, k)
    cli.py        python -m retrieval.versioned {index,update,query,list-versions}
"""
