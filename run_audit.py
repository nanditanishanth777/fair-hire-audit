"""CLI: python run_audit.py [--n-base 1000] [--bullet-level names] ..."""
from __future__ import annotations

import argparse

import pandas as pd

from fairhire.audit import AuditConfig, run_audit
from fairhire.redact import LEVELS
from fairhire.report import make_bullets, write_outputs


def main() -> None:
    p = argparse.ArgumentParser(description="Counterfactual fairness audit of a resume matcher.")
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-base", type=int, default=1000, help="base resumes (pairs = 3 x this)")
    p.add_argument("--n-pool", type=int, default=500)
    p.add_argument("--select-frac", type=float, default=0.25)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--bullet-level", choices=LEVELS[1:], default="names",
                   help="redaction level used to fill the second resume bullet")
    p.add_argument("--out", default="outputs")
    args = p.parse_args()

    cfg = AuditConfig(n_train=args.n_train, n_base=args.n_base, n_pool=args.n_pool,
                      select_frac=args.select_frac, n_boot=args.bootstrap, seed=args.seed)
    result = run_audit(cfg)
    write_outputs(result, args.out, args.bullet_level)

    pd.set_option("display.width", 200)
    cols = ["level", "n_pairs", "rank_change_pct", "shortlist_flip_pct",
            "worst_selection_rate_ratio", "worst_dimension", "auc_vs_merit"]
    print(result.overall[cols].round(3).to_string(index=False))
    print("\nResume bullets:")
    bullets, values = make_bullets(result, args.bullet_level)
    for b in bullets:
        print(f"* {b}")
    if not values["threshold_met"]:
        print(f"\nNote: worst-case ratio {values['Z']:.2f} is below 0.8 at level '{args.bullet_level}'.")
    print(f"\nFull report: {args.out}/report.md")


if __name__ == "__main__":
    main()
