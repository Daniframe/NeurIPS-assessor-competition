# Submission log

One branch per substantially different attempt; the submission is the zipped contents of
`sample_code_submission/` on that branch. Notation as in `approaches.md`: P = prior, T = test-time.

| branch | approach | local evaluation (Brier ALC) | CodaBench Brier ALC | notes |
|---|---|---|---|---|
| `submission/a4-embed-t2` | A4: R1 + R2 + S1 + I0 + I1 \| T2 | 0.1966 (protocol v1, measurement-db) | — | first submission |

## submission/a4-embed-t2

- **Prior:** LightGBM on subject features (S1), cross-benchmark model strength (R2), item text
  statistics (I0) and 48 PCA components of bge-base-en-v1.5 embeddings (I1). It is started from the mean
  benchmark base rate (R1).
- **Training:** all six measurement-db benchmarks (revision `bc8204d`), 229,297 pairs.
- **Test-time (T2):** MAP fit, at every call, of:
  - a benchmark shift, a subject shift per benchmark and an item shift;
  - a slope on the prior's deviation, with a Gaussian prior centred on 0.5.

  The strengths are those tuned for A4 (benchmark 1.0, subject 1.2, item 2.0, slope 0.5), the same in
  all four folds.
- **Acquisition:** random, so there is no `labeling.py`.
- **Not A10:** A10 adds ADeLe 1.0 levels (I2), which would need the 7B annotator on every test item at
  runtime. A4 scored 0.1966 against A10's 0.1962 in the same evaluation.
- **Files:** `model.py`; `artifacts/` (booster, embedding PCA, config with strengths and adaptation
  strengths); `models.txt` (bge-base-en-v1.5); `requirements.txt` (lightgbm 4.7.0).
- **Rebuild and check:**
  - `python -m pipeline.build_submission` writes `artifacts/`;
  - `python -m pipeline.check_submission` checks the runtime embeddings, prior and adaptation against the
    offline pipeline.

  Both scripts live in `pipeline/`, which is gitignored.
- **Local checks:**
  - runtime embeddings match the cached ones (cosine 1.00000);
  - runtime prior logits match the offline model (mean difference 0.0004);
  - `predict()` reproduces the simulator's Brier scores to 1e-9;
  - 1.8 ms per call on average once an item is embedded;
  - kit smoke test passes.
