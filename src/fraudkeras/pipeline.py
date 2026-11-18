"""End-to-end evaluation with the same protocol as the scikit-learn repository.

Days 0-59 train, 60-74 validation (early stopping + alert threshold), 75-89 test. The
``mule_cashout`` attack appears only in the test period. Alert threshold = validation score
that alarms 0.5% of transactions, applied unchanged to test.

Extras specific to neural networks:
* three class-imbalance strategies (plain BCE / weighted BCE / focal loss),
* the same model trained with several random seeds, because neural-network results move with
  the seed and a single run can mislead,
* an autoencoder trained on legitimate transactions only.
"""

from __future__ import annotations

import time

import numpy as np

from . import metrics, models
from .data import build_features, simulate, split_by_day

LOSSES = ("bce", "bce_weighted", "focal")


def _pct(val: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(val), x) / len(val)


def _summary(r: dict) -> dict:
    return {
        "pr_auc": r["pr_auc"], "pr_auc_known_patterns": r["pr_auc_known_patterns"],
        "recall": r["recall"], "precision": r["precision"], "alert_rate": r["alert_rate"],
        "net_savings_pln": r["net_savings_pln"],
        "recall_by_pattern": {k: v["recall"] for k, v in r["recall_by_pattern"].items()},
    }  # fmt: skip


def run_eval(n_cards: int = 2500, days: int = 90, seed: int = 0, n_seeds: int = 3, n_boot: int = 200) -> dict:
    t0 = time.time()
    df = simulate(n_cards, days, seed)
    X = build_features(df)
    tr, va, te = split_by_day(df)
    y = df["is_fraud"].to_numpy()
    dte = df[te].reset_index(drop=True)
    prep = models.Prep().fit(X[tr])
    out: dict = {
        "data": {
            "n_cards": n_cards, "days": days, "seed": seed, "n_transactions": int(len(df)),
            "fraud_rate": round(float(y.mean()), 4),
            "n_fraud": {"train": int(y[tr].sum()), "validation": int(y[va].sum()), "test": int(y[te].sum())},
        },
        "models": {},
    }  # fmt: skip

    def run(name, sv, st):
        thr = metrics.threshold_for_budget(sv)
        out["models"][name] = metrics.evaluate(dte, st, thr)
        return thr

    # imbalance strategies, seed 0
    scores = {}
    for loss in LOSSES:
        m, epochs = models.fit_mlp(prep, X[tr], y[tr], X[va], y[va], loss=loss, seed=seed)
        scores[loss] = (models.predict_mlp(m, prep, X[va]), models.predict_mlp(m, prep, X[te]))
        run(f"mlp_{loss}", *scores[loss])
        out["models"][f"mlp_{loss}"]["epochs_run"] = epochs

    # seed variance of the main configuration
    per_seed = []
    for s in range(n_seeds):
        m, _ = models.fit_mlp(prep, X[tr], y[tr], X[va], y[va], loss="bce_weighted", seed=seed + s)
        sv, st = models.predict_mlp(m, prep, X[va]), models.predict_mlp(m, prep, X[te])
        per_seed.append(metrics.evaluate(dte, st, metrics.threshold_for_budget(sv)))
    keys = ("pr_auc", "pr_auc_known_patterns", "recall", "net_savings_pln")
    out["seed_variance_mlp_bce_weighted"] = {
        "n_seeds": n_seeds,
        **{k: {"mean": round(float(np.mean([r[k] for r in per_seed])), 4),
               "std": round(float(np.std([r[k] for r in per_seed])), 4)} for k in keys},
        "mule_recall": {
            "mean": round(float(np.mean([r["recall_by_pattern"]["mule_cashout"]["recall"] for r in per_seed])), 3),
            "std": round(float(np.std([r["recall_by_pattern"]["mule_cashout"]["recall"] for r in per_seed])), 3),
        },
    }  # fmt: skip

    # unsupervised autoencoder and an a-priori hybrid (mean of percentile ranks)
    ae = models.fit_autoencoder(prep, X[tr], y[tr], X[va], y[va], seed=seed)
    a_va, a_te = models.anomaly_score(ae, prep, X[va]), models.anomaly_score(ae, prep, X[te])
    run("autoencoder", a_va, a_te)
    g = scores["bce_weighted"]
    hyb = tuple(0.5 * (_pct(g[0], g[k]) + _pct(a_va, a)) for k, a in ((0, a_va), (1, a_te)))
    run("hybrid_mean", *hyb)
    for name, (sv, st) in (("mlp_bce_weighted", g), ("hybrid_mean", hyb)):
        out["models"][name].update(
            metrics.bootstrap_ci(dte, st, metrics.threshold_for_budget(sv), n_boot=n_boot, seed=seed)
        )
    out["models_summary"] = {k: _summary(v) for k, v in out["models"].items()}
    out["runtime_s"] = round(time.time() - t0, 1)
    return out
