# Grouping by Result, Not Difficulty, Delays Grokking

Code for the paper *"Grouping by Result, Not Difficulty, Delays Grokking:
A Replication Failure and a Robust Effect"* (Isaac Gong, 2026).

We test five ways of ordering training examples against grokking on modular
addition. An exploratory result (Study 1, fine-tuning GPT-2) suggesting
difficulty-based ordering delays grokking by 97% did **not** replicate in a
controlled follow-up (Study 2, training from scratch). A structurally
different ordering, grouping examples by their result value, produced a
125% delay that held up across every seed tested. See the paper for the
full story, including what we could and could not explain about the
discrepancy between the two studies.

## Repo layout

```
src/
  curriculum/curriculum.py    Shared ordering logic. Imported unchanged by
                               both studies -- this is what lets the paper
                               say the two studies use identical curriculum
                               code (see paper Section 5).
  study1_gpt2/train.py         GPT-2 fine-tuning training loop.
  study2_scratch/
    model.py                   Small transformer (matches Nanda et al. 2023
                               architecture). Fully tested end-to-end.
    train.py                   Training loop. Fully tested end-to-end.
  analysis/
    stats.py                   Grokking-step statistics, Welch's t-test.
    gini.py                    Gini coefficient progress measure
                               (Nanda et al. 2023's actual metric).
    fourier.py                 Dominant-frequency comparison across
                               conditions.
scripts/
  generate_report.py           Regenerates the paper's Table 2 and LaTeX
                               source from results/summary.json.
notebooks/
  study2_from_scratch.ipynb    The actual Colab notebook that produced the
                               full 15-run grid reported in the paper.
results/
  summary.json                 Verified numeric results from both studies.
```

## Quickstart

```bash
pip install -r requirements.txt

# Verify the curriculum logic
python src/curriculum/curriculum.py

# Verify the Gini module (includes a regression test for a real bug
# caught during development -- see model.py docstring)
python src/analysis/gini.py

# Verify the stats module reproduces the paper's exact numbers
python src/analysis/stats.py

# Regenerate Table 2 and its LaTeX source from results/summary.json
python scripts/generate_report.py

# Smoke-test the from-scratch training loop (grokks at step ~4500,
# takes ~1 min on a GPU, ~2.5 min on CPU)
python src/study2_scratch/train.py
```

## What is and isn't in this repo

**Tested and verified before being committed:**
- `src/curriculum/curriculum.py` — every ordering checked to be a
  permutation of its input.
- `src/study2_scratch/model.py` and `train.py` — run end-to-end on CPU
  before use, including the smoke-test config that must reach grokking.
- `src/analysis/gini.py` — includes a regression test against a real bug
  (embedding-snapshot aliasing) that was caught and fixed during
  development. See the docstring in `model.py`'s
  `number_embeddings()` for details.
- `src/analysis/stats.py` and `fourier.py` — each asserts its output
  matches the exact numbers reported in the paper.
- `results/summary.json` — cross-validated against a live run of
  `stats.py`, not hand-typed and trusted.

**Reconstructed, not extracted verbatim:**
- `src/study1_gpt2/train.py` — the curriculum logic is exact (shared
  module), but the GPT-2-specific helper functions (dataset, collate,
  accuracy eval) are faithful reconstructions of what was run across many
  interactive Colab sessions, not a literal export of executed code. See
  the provenance note at the top of that file.

**Not included:**
- Full per-step `history.json` / `embed_log.npz` files for every run.
  These live on the author's Google Drive. `results/summary.json`
  contains the verified aggregate numbers derived from them (grokking
  steps, means, standard deviations, t-test results, Gini lead times,
  dominant frequencies) but not the raw per-step logs themselves. An
  earlier draft of this project included fabricated per-step figures
  before the real data existed; this repo does not reproduce that
  mistake by inventing per-step files after the fact either. If you need
  the raw logs, open an issue.

## Reproducing the grid

`notebooks/study2_from_scratch.ipynb` is the notebook that was actually
run in Colab to produce the 15-run grid (5 curricula x 3 seeds) reported
in the paper. It mounts Google Drive, runs a smoke test before the full
grid, and writes `history.json` and `embed_log.npz` per run, skipping any
run that already has output so it is safe to resume after a disconnect.

The equivalent `.py` modules in `src/study2_scratch/` are the
non-notebook version of the same code, usable outside Colab.

## Citation

If you use this code, please cite the paper. BibTeX for the papers this
work builds on and replicates is in the paper's `references.bib`.

## License

MIT. See `LICENSE`.
