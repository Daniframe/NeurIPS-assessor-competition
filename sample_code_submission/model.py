"""Submission A4: offline prior + test-time adaptation with a prior slope.

    logit P(success) = prior + (slope - 1) * dev + beta[benchmark] + delta[subject, benchmark] + gamma[item]

- prior: LightGBM (artifacts/prior_lgbm.txt) on subject features (provider, dates, reasoning effort, size
  from the name, harness), cross-benchmark model strength (artifacts/config.json, by base model; missing
  for unseen models), item text statistics, and 48 PCA components of the item's bge-base-en-v1.5
  embedding. It starts from the mean benchmark base rate of the training data (`default_offset`); `dev` is
  the prior's deviation from that start.
- test-time: at every call, the labels in `labeled` (all subject-benchmark pairs of the draw) are used to
  fit, by MAP, a benchmark shift, a shift per subject on each benchmark, a shift per item (from other
  subjects' labels on it) and the slope, i.e. how far to trust the prior's deviations on this benchmark
  (Gaussian prior centred on alpha = 0.5). With no labels, the prediction is start + alpha * dev.

Trained offline by pipeline/build_submission.py on measurement-db. Random acquisition (no labeling.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
CONFIG = json.loads((ART / "config.json").read_text(encoding="utf-8"))
BOOSTER = lgb.Booster(model_file=str(ART / "prior_lgbm.txt"))
with np.load(ART / "embedding_pca.npz") as _pca:   # read and close (an open handle blocks cleanup on Windows)
    PCA_MEAN, PCA_COMPONENTS = _pca["mean"].copy(), _pca["components"].copy()
COLUMNS, PROVIDERS = CONFIG["columns"], CONFIG["providers"]
OFFSET, TAU, STRENGTH = CONFIG["default_offset"], CONFIG["tau"], CONFIG["strength"]
EMBED_MODEL = "BAAI/bge-base-en-v1.5"   # declared in models.txt
SUBJECT_KEYS = ["normalized_name", "provider", "release_date", "access_date", "harness",
                "reasoning_effort", "harness_version", "subject_features_extra"]
ITEM_KEYS = ["item_content", "item_features", "interactors", "benchmark_id"]


# ----------------------------------------------------------------------------- embedding model

def _cache_dir():
    for candidate in [os.environ.get("HF_HOME", "").strip(), "/app/hf_cache", str(HERE / ".hf_cache")]:
        if candidate and Path(candidate).is_dir():
            return candidate
    return None


def _load_encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    if os.environ.get("PAIEC_PREDICTION_WORKER_ID"):   # several CPU workers share the machine
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))
    local = not os.environ.get("PREDICTIVE_EVAL_LOCAL_SMOKE_TEST")
    kwargs = {"cache_dir": _cache_dir(), "local_files_only": True} if local else {}
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, **kwargs)
    model = AutoModel.from_pretrained(EMBED_MODEL, **kwargs).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch, tokenizer, model.to(device), device


TORCH, TOKENIZER, ENCODER, DEVICE = _load_encoder()


def _embed(texts):
    """CLS embedding, L2-normalised (bge-base-en-v1.5's sentence-transformers configuration)."""
    out = []
    with TORCH.inference_mode():
        for start in range(0, len(texts), 16):
            batch = TOKENIZER(texts[start:start + 16], padding=True, truncation=True,
                              max_length=CONFIG["max_tokens"], return_tensors="pt").to(DEVICE)
            cls = ENCODER(**batch).last_hidden_state[:, 0]
            out.append(TORCH.nn.functional.normalize(cls, p=2, dim=1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ----------------------------------------------------------------------------- features (as in training)

COMPANY_WORDS = {"anthropic", "openai", "google", "alibaba", "meta", "xai", "moonshot", "zhipu",
                 "bytedance", "microsoft", "nvidia", "stepfun", "amazon", "cohere", "openbmb",
                 "opengvlab", "stanford"}
DROP_WORDS = {"think", "thinking", "agent", "preview", "exp", "instruct", "it", "beta"}
EFFORT = {"low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}
SMALL_WORDS = {"mini", "nano", "flash", "lite", "small", "haiku", "air", "turbo", "fast"}
LARGE_WORDS = {"pro", "opus", "max", "ultra", "plus"}
REASONING_PAT = re.compile(r"think|reason|\br1\b|qwq|\bo[134]\b|o[134] mini|deep think|"
                           r"distill|s1\.1|limo", re.I)
PLACEHOLDER = re.compile(r"see image|see the (?:image|figure)|you are chatgpt|^date: ", re.I)


def _parse_kv(s):
    return dict(part.split("=", 1) for part in (s or "").split(";") if "=" in part)


def _base_model_key(name):
    s = name.lower().replace("*", "")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[-_/:]", " ", s)
    s = re.sub(r"(\d)\.0\b", r"\1", s)
    words = [w for w in s.split() if w not in DROP_WORDS]
    while words and words[0] in COMPANY_WORDS:
        words = words[1:]
    return " ".join(words)


def _years(date_str, origin=pd.Timestamp("2023-01-01")):
    t = pd.to_datetime(date_str, errors="coerce")
    return np.nan if pd.isna(t) else (t - origin).days / 365.25


def _params_b(name):
    m = re.findall(r"(\d+(?:\.\d+)?)\s*b\b", name.lower().replace("-", " "))
    return float(m[0]) if m else np.nan


def _subject_features(subj):
    name = subj["normalized_name"] or _parse_kv(subj["subject_features_extra"]).get("source_model_name", "")
    low = name.lower()
    words = set(re.split(r"[\s\-_/:()]+", low))
    effort = subj["reasoning_effort"].lower() or next((e for e in EFFORT if re.search(rf"\({e}\)", low)), "")
    return {
        "strength": STRENGTH.get(_base_model_key(name), np.nan),
        "release_years": _years(subj["release_date"]),
        "access_years": _years(subj["access_date"]),
        "effort": EFFORT.get(effort, 0),
        "reasoning": float(bool(REASONING_PAT.search(low)) or effort != ""),
        "has_harness": float(subj["harness"] != ""),
        "agent_scaffold": float("agent" in low or "scaffold" in subj["provider"].lower()),
        "log_params": np.log1p(_params_b(name)),
        "small_tier": float(bool(words & SMALL_WORDS)),
        "large_tier": float(bool(words & LARGE_WORDS)),
        "name_blank": float(subj["normalized_name"] == ""),
        "provider": subj["provider"].lower() or "(blank)",
    }


def _item_stats(item):
    c = item["item_content"]
    n = max(len(c), 1)
    return {
        "log_chars": np.log1p(len(c)),
        "log_words": np.log1p(len(c.split())),
        "log_lines": np.log1p(c.count("\n")),
        "digit_share": len(re.findall(r"\d", c)) / n,
        "math_share": len(re.findall(r"[\\$^_{}=]", c)) / n,
        "code_share": len(re.findall(r"[`;(){}\[\]<>]", c)) / n,
        "upper_share": len(re.findall(r"[A-Z]", c)) / n,
        "n_urls": np.log1p(len(re.findall(r"https?://", c))),
        "n_questions": np.log1p(c.count("?")),
        "has_choices": float(bool(re.search(r"\(A\)|\bA\)|\bA\.", c))),
        "placeholder": float(bool(PLACEHOLDER.search(c))),
        "n_item_features": float(item["item_features"].count("=")),
    }


def _item_text(item):
    return item["item_content"][:CONFIG["text_chars"]] + " " + re.sub(r"[=;_]", " ", item["item_features"])


# ----------------------------------------------------------------------------- prior (cached per worker)

def _key(d, keys):
    return hashlib.sha1("\x1f".join(str(d.get(k, "")) for k in keys).encode("utf-8")).hexdigest()[:16]


_EMB, _ITEM_ROW, _PRIOR = {}, {}, {}


def _priors(pairs):
    """Prior logits for a list of (subject, item) pairs; embeds missing items in one batch."""
    todo = [(s, it) for s, it in pairs if (_key(s, SUBJECT_KEYS), _key(it, ITEM_KEYS)) not in _PRIOR]
    if todo:
        texts = {}
        for _, it in todo:
            ik = _key(it, ITEM_KEYS)
            if ik not in _ITEM_ROW:
                texts.setdefault(_item_text(it), []).append((ik, it))
        new = [t for t in texts if t not in _EMB]
        if new:
            _EMB.update(zip(new, _embed(new)))
        for text, entries in texts.items():
            z = (_EMB[text].astype(np.float64) - PCA_MEAN) @ PCA_COMPONENTS.T
            emb = {f"emb{i}": v for i, v in enumerate(z)}
            for ik, it in entries:
                _ITEM_ROW[ik] = {**_item_stats(it), **emb}
        rows = [{**_subject_features(s), **_ITEM_ROW[_key(it, ITEM_KEYS)]} for s, it in todo]
        X = pd.DataFrame(rows)[COLUMNS]
        X["provider"] = pd.Categorical(X["provider"], categories=PROVIDERS)
        raw = BOOSTER.predict(X, raw_score=True, num_threads=1)
        for (s, it), r in zip(todo, raw):
            _PRIOR[(_key(s, SUBJECT_KEYS), _key(it, ITEM_KEYS))] = float(r) + OFFSET
    return [_PRIOR[(_key(s, SUBJECT_KEYS), _key(it, ITEM_KEYS))] for s, it in pairs]


# ----------------------------------------------------------------------------- test-time adaptation

def _fit_effects(prior, y, groups):
    """MAP fit of the random effects and the slope (pipeline/adapt.py, mode full+slope)."""
    comps = ["bench", "subject", "item"]
    idx, sizes, levels = [], [], []
    for c in comps:
        codes, uniq = pd.factorize(pd.Series(groups[c]))
        idx.append(codes)
        sizes.append(len(uniq))
        levels.append(uniq)
    offsets = np.cumsum([0] + sizes)
    n_re = offsets[-1]
    prec = np.concatenate([np.full(n, 1 / TAU[c] ** 2) for c, n in zip(comps, sizes)])
    dev = prior - OFFSET
    base = prior - dev
    alpha, prec_s = TAU["alpha"], 1 / TAU["slope"] ** 2

    def loss(theta):
        lam = theta[-1]
        z = base + lam * dev + sum(theta[offsets[k] + idx[k]] for k in range(len(comps)))
        p = 1 / (1 + np.exp(-z))
        nll = -np.sum(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        g = p - y
        re_ = theta[:n_re]
        grad = np.concatenate([np.bincount(idx[k], g, sizes[k]) for k in range(len(comps))]) + prec * re_
        val = nll + 0.5 * np.sum(prec * re_ ** 2) + 0.5 * prec_s * (lam - alpha) ** 2
        return val, np.concatenate([grad, [np.sum(g * dev) + prec_s * (lam - alpha)]])

    x0 = np.zeros(n_re + 1)
    x0[-1] = alpha
    res = minimize(loss, x0, jac=True, method="L-BFGS-B")
    effects = {c: dict(zip(levels[k], res.x[offsets[k]:offsets[k + 1]])) for k, c in enumerate(comps)}
    effects["slope"] = float(res.x[-1])
    return effects


def _groups(subject, item):
    bench = item.get("benchmark_id", "")
    return bench, f"{_key(subject, SUBJECT_KEYS)}|{bench}", _key(item, ITEM_KEYS)


_EFFECTS = {}


def _effects(labeled):
    """Effects for this evidence set; every call of a checkpoint sees the same `labeled`, so cache it."""
    if not labeled:
        return {"slope": TAU["alpha"]}
    fingerprint = hashlib.sha1("".join(
        f"{_key(p[0], SUBJECT_KEYS)}{_key(p[1], ITEM_KEYS)}{int(lab)}" for p, lab in labeled
    ).encode()).hexdigest()
    if fingerprint not in _EFFECTS:
        pairs = [(p[0], p[1]) for p, _ in labeled]
        prior = np.asarray(_priors(pairs))
        y = np.asarray([float(lab) for _, lab in labeled])
        keys = [_groups(s, it) for s, it in pairs]
        groups = {"bench": [k[0] for k in keys], "subject": [k[1] for k in keys], "item": [k[2] for k in keys]}
        _EFFECTS.clear()
        _EFFECTS[fingerprint] = _fit_effects(prior, y, groups)
    return _EFFECTS[fingerprint]


# ----------------------------------------------------------------------------- entry point

_WARNED = []


def predict(input: list, labeled: list | None = None) -> float:
    """Return the predicted probability that the subject answers correctly."""
    subject, item = input
    try:
        prior = _priors([(subject, item)])[0]
        eff = _effects(labeled or [])
        bench, subj, it = _groups(subject, item)
        z = prior + (eff["slope"] - 1.0) * (prior - OFFSET)
        z += eff.get("bench", {}).get(bench, 0.0) + eff.get("subject", {}).get(subj, 0.0) \
            + eff.get("item", {}).get(it, 0.0)
        p = 1.0 / (1.0 + np.exp(-z))
    except Exception as exc:   # never fail the whole run on one input: fall back to the base rate
        if not _WARNED:
            print(f"[A4] prediction failed, using the base rate: {exc!r}", file=sys.stderr, flush=True)
            _WARNED.append(True)
        p = 1.0 / (1.0 + np.exp(-OFFSET))
    return float(np.clip(p, 1e-4, 1 - 1e-4)) if np.isfinite(p) else 0.5
