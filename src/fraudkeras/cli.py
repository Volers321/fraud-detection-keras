"""Command line: ``fraudkeras eval`` runs the whole study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(res: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(res["models"])
    palette = ["#9aa0a6", "#1f77b4", "#d95f02", "#7b4fa3", "#2f855a", "#b7791f"]
    colors = dict(zip(names, palette))
    patterns = ["card_testing", "takeover", "stolen_card", "mule_cashout"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=140)
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [res["models"][n]["pr_auc_known_patterns"] for n in names], 0.4, label="known patterns", color="#2b6cb0")  # fmt: skip
    ax[0].bar(x + 0.2, [res["models"][n]["pr_auc"] for n in names], 0.4, label="all (incl. new attack)", color="#c05621")  # fmt: skip
    ax[0].set_xticks(x, names, rotation=25, ha="right")
    ax[0].set_ylabel("PR-AUC (test)")
    ax[0].set_title("Ranking quality")
    ax[0].legend(frameon=False)
    w = 0.8 / len(names)
    for j, n in enumerate(names):
        rec = [res["models"][n]["recall_by_pattern"].get(p, {}).get("recall", 0) for p in patterns]
        ax[1].bar(np.arange(len(patterns)) + (j - (len(names) - 1) / 2) * w, rec, w, label=n, color=colors[n])
    ax[1].set_xticks(range(len(patterns)), patterns, rotation=15)
    ax[1].set_ylabel("recall at 0.5% alert budget")
    ax[1].set_title("Per attack pattern (mule_cashout is new in test)")
    ax[1].legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    for a in ax:
        a.set_ylim(0, 1.3)
        a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="fraudkeras")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="run the full temporal evaluation")
    e.add_argument("--n-cards", type=int, default=2500)
    e.add_argument("--days", type=int, default=90)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--n-seeds", type=int, default=3)
    e.add_argument("--n-boot", type=int, default=200)
    e.add_argument("--out", default="results/eval.json")
    e.add_argument("--plot", default="docs/eval.png")
    a = p.parse_args(argv)
    from .pipeline import run_eval

    res = run_eval(a.n_cards, a.days, a.seed, a.n_seeds, a.n_boot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    Path(a.plot).parent.mkdir(parents=True, exist_ok=True)
    plot(res, a.plot)
    print(json.dumps({k: res[k] for k in ("models_summary", "seed_variance_mlp_bce_weighted")}, indent=2))
    print(f"\nwrote {a.out} and {a.plot} in {res['runtime_s']} s")


if __name__ == "__main__":
    main()
