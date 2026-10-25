# Fair-Hire Audit

A counterfactual fairness audit for resume-screening models. It answers one question:
**if two candidates have identical qualifications and differ only in a name, a gender cue, or
college tier, does the matcher treat them differently?**

For each base resume the tool builds paired variants that differ only in the identity cue,
scores both with the matcher, and reports how often the ranking changes and whether
selection rates pass the **four-fifths (0.8) rule**. It then strips identity cues from the
text at several levels and repeats the audit, so you can see which mitigation actually helps.

> **Note:** all data is synthetic with deliberately injected bias. The results demonstrate the
> audit pipeline on a controlled setup. They say nothing about the fairness of any real
> hiring system. See [Limitations](#limitations).

## Results

Fill these from `outputs/summary.json` after running with the default seed.

| Redaction level | Rank change % | Shortlist flip % | Mean abs. rank shift | Worst-case selection ratio | Passes 0.8? |
|---|---|---|---|---|---|
| none | | | | | |
| names | | | | | |
| names+gender | | | | | |
| names+gender+college | | | | | |

AUC vs merit (usefulness after redaction): _fill in_

## Quick start

```bash
pip install -r requirements.txt
pytest -q
python run_audit.py                                  # 1,000 base resumes -> 3,000 pairs
python run_audit.py --bullet-level names+gender+college
```

Outputs are written to `outputs/`:

| File | Contents |
|---|---|
| `report.md` | Human-readable audit report |
| `overall.csv` | Metrics per redaction level |
| `per_dimension.csv` | Metrics per dimension (name, gender, college) and level |
| `identity_token_weights.csv` | Model weights on identity tokens |
| `summary.json` | Headline numbers in machine-readable form |

## How it works

```
synthetic resumes (+ injected bias in hire labels)
        |
        v
train matcher on biased history  -->  build counterfactual pairs
                                              |
                          redact identity cues (4 levels)
                                              |
                              score, rank, shortlist
                                              |
                       rank change / flips / four-fifths ratio
```

| Component | File | Role |
|---|---|---|
| Data | `fairhire/data.py` | Generates resumes. Identity fields are independent of merit fields, so a variant changes identity and nothing else. Historical hire labels get a configurable bias (`BiasConfig`). |
| Matcher | `fairhire/matcher.py` | TF-IDF + logistic regression trained on the biased history. Swap in any object with a `score(texts)` method. |
| Redaction | `fairhire/redact.py` | Cumulative levels: `none`, `names`, `names+gender`, `names+gender+college`. |
| Metrics | `fairhire/metrics.py` | Rank change, shortlist flips, selection-rate ratio, bootstrap confidence intervals. |
| Audit | `fairhire/audit.py` | Runs every dimension at every redaction level. |

### Metrics

- **Rank change %**: each variant is ranked against a fixed reference pool of 500 resumes
  (rank = 1 + number of pool members scoring strictly higher). A pair changes if its two
  variants get different ranks. This is strict, since any non-zero score gap can move a rank,
  so read it together with the two metrics below.
- **Shortlist flip %**: share of pairs where exactly one variant lands in the top 25%.
- **Mean absolute rank shift**: average size of the rank movement.
- **Selection-rate ratio**: every base resume is scored under every group, the top 25% of all
  renderings is shortlisted, and each group's rate is the share of its renderings shortlisted.
  Ratio = lowest group rate / highest. Merit is identical across groups, so any gap is caused
  by the identity cue alone. The **worst-case ratio** is the minimum across the three dimensions.
- **AUC vs merit**: held-out check that redaction does not destroy the matcher's usefulness.

## Customizing

- **Bias strength:** edit `BiasConfig` in `fairhire/data.py`.
- **Name lists and college tiers:** also in `data.py`. They are illustrative assumptions.
- **Your own matcher:** pass any object exposing `score(texts) -> array`.
- **Shortlist size, pool size, seed:** see `run_audit.py --help`.

## Limitations

- **Synthetic data with injected bias.** The numbers validate the pipeline, not any real system.
- **Redaction is lexicon-based**, and the generator only uses cues on that lexicon. Full
  redaction therefore gives exactly 0% rank change and a ratio of 1.00 *by construction*.
  Real resumes leak identity through clubs, locations, employment gaps, writing style and more.
- **Removing names alone does not fix gender or college effects.** The report shows each level.
- **The four-fifths rule is a screening heuristic** from US employment guidance. It is not
  proof of fairness or a legal conclusion.
- **Name lists and college tiers are assumptions**, not validated demographic proxies.

## Possible extensions

- Audit a real or pretrained matcher (embedding model or LLM scorer) instead of TF-IDF.
- Add a leakage test: train a classifier to predict identity from redacted text.
- Add intersectional groups (for example, gender x college tier).
- Add statistical tests alongside the bootstrap intervals.

## License

Add a LICENSE file (MIT is a common choice).
