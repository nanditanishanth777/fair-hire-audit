"""Turn an AuditResult into CSVs, a markdown report and resume-ready bullets."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .audit import AuditResult
from .redact import LEVEL_DESCRIPTIONS


def make_bullets(result: AuditResult, level: str = "names") -> tuple[list[str], dict]:
    """Return the two resume bullets (numbers filled in) and the values used."""
    ov = result.overall.set_index("level")
    base, fixed = ov.loc["none"], ov.loc[level]
    n, x, y = int(base["n_pairs"]), base["rank_change_pct"], fixed["rank_change_pct"]
    z = fixed["worst_selection_rate_ratio"]
    verb = "reached" if z >= 0.8 else ("improved to" if z > base["worst_selection_rate_ratio"] else "stayed at")
    bullets = [
        f"Tested the matcher on {n:,} paired synthetic resumes that differ only in name, gender cues "
        f"or college tier; ranking changed for {x:.1f}% of pairs.",
        f"After {LEVEL_DESCRIPTIONS[level]}, rank changes fell to {y:.1f}% and the worst-case "
        f"selection-rate ratio {verb} {z:.2f} against the 0.8 four-fifths threshold.",
    ]
    values = {"N": n, "X": round(x, 2), "Y": round(y, 2), "Z": round(z, 3),
              "baseline_Z": round(base["worst_selection_rate_ratio"], 3),
              "level": level, "threshold_met": bool(z >= 0.8)}
    return bullets, values


def _md_table(df: pd.DataFrame, formats: dict | None = None) -> str:
    formats = formats or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(formats[c].format(v) if c in formats and pd.notna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(result: AuditResult, outdir: str | Path, bullet_level: str = "names") -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    result.per_dimension.to_csv(out / "per_dimension.csv", index=False)
    result.overall.to_csv(out / "overall.csv", index=False)
    result.identity_weights.to_csv(out / "identity_token_weights.csv", index=False)

    bullets, values = make_bullets(result, bullet_level)
    (out / "summary.json").write_text(json.dumps(values, indent=2))

    cfg = result.config
    ov = result.overall.copy()
    ov["rank_change_95ci"] = ov.apply(lambda r: f"{r.rank_change_ci_lo:.1f}-{r.rank_change_ci_hi:.1f}", axis=1)
    ov["worst_ratio_95ci"] = ov.apply(lambda r: f"{r.worst_ratio_ci_lo:.2f}-{r.worst_ratio_ci_hi:.2f}", axis=1)
    ov = ov[["level", "n_pairs", "rank_change_pct", "rank_change_95ci", "shortlist_flip_pct",
             "worst_selection_rate_ratio", "worst_ratio_95ci", "worst_dimension", "auc_vs_merit"]]
    ov_md = _md_table(ov, {"rank_change_pct": "{:.1f}", "shortlist_flip_pct": "{:.1f}",
                           "worst_selection_rate_ratio": "{:.2f}", "auc_vs_merit": "{:.3f}"})

    pd_ = result.per_dimension[["level", "dimension", "rank_change_pct", "mean_abs_rank_shift_pct",
                                "shortlist_flip_pct", "selection_rates", "selection_rate_ratio",
                                "passes_four_fifths"]]
    pd_md = _md_table(pd_, {"rank_change_pct": "{:.1f}", "mean_abs_rank_shift_pct": "{:.2f}",
                            "shortlist_flip_pct": "{:.1f}", "selection_rate_ratio": "{:.2f}"})

    w = result.identity_weights
    w_md = ""
    if len(w):
        top = pd.concat([w.head(8), w.tail(8)])
        w_md = "\n## Identity tokens the matcher learned from biased history\n\n" + _md_table(top, {"weight": "{:+.2f}"}) + "\n"

    note = "" if values["threshold_met"] else (
        f"\n> At level `{bullet_level}` the worst-case ratio ({values['Z']:.2f}) is still below 0.8, "
        "so the bullet says so. Try `--bullet-level names+gender+college` or edit the wording.\n"
    )
    report = f"""# Fair-Hire Audit report

Synthetic benchmark. Seed {cfg.seed}; {cfg.n_train:,} training decisions; {cfg.n_base:,} base resumes
x 3 dimensions; reference pool {cfg.n_pool}; shortlist = top {cfg.select_frac:.0%}.
Injected bias (logit units): gender {cfg.bias.gender_gap}, name group {cfg.bias.name_gap}, college tier step {cfg.bias.tier_step}.

## Overall by redaction level

{ov_md}

## Per dimension

{pd_md}
{w_md}
## Resume bullets (level: {bullet_level})

""" + "\n".join(f"* {b}" for b in bullets) + "\n" + note
    (out / "report.md").write_text(report)
    return values
