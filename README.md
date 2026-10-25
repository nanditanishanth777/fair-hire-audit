# Fair-Hire Audit

Counterfactual fairness audit for a resume matcher. It builds paired resumes that
differ only in **name**, **gender cues** (name, title, pronouns) or **college tier**,
scores both with the matcher, and measures how often the ranking changes and
whether selection rates pass the **four-fifths (0.8) rule**. It then repeats the
audit after removing identity cues from the text before scoring.

Python, pandas, scikit-learn, pytest.

## Run it

```bash
pip install -r requirements.txt
pytest -q
python run_audit.py                      # defaults: 1,000 base resumes -> 3,000 pairs
python run_audit.py --bullet-level names+gender+college
```

Outputs land in `outputs/`: `report.md`, `overall.csv`, `per_dimension.csv`,
`identity_token_weights.csv`, `summary.json` (the N / X / Y / Z for your bullets).

## How it works

| Piece | File | What it does |
|---|---|---|
| Data | `fairhire/data.py` | Synthetic resumes. Identity fields are independent of merit fields, so a variant changes identity and nothing else. Historical "hire" labels get a configurable bias injected (`BiasConfig`). |
| Matcher | `fairhire/matcher.py` | TF-IDF + logistic regression trained on the biased history. Swap in any object with `score(texts)`. |
| Redaction | `fairhire/redact.py` | Four cumulative levels: `none`, `names`, `names+gender`, `names+gender+college`. |
| Metrics | `fairhire/metrics.py` | Rank change, shortlist flips, selection-rate ratio, bootstrap CIs. |
| Audit | `fairhire/audit.py` | Runs every dimension at every redaction level. |

### Metric definitions

* **Rank change %**: each variant is ranked against a fixed reference pool of 500 resumes
  (rank = 1 + number of pool members scoring strictly higher). A pair "changes" if its two
  variants land on different ranks. This is strict: any non-zero score gap can move a rank, so
  also read **shortlist flip %** (exactly one variant is in the top 25%) and
  **mean absolute rank shift**.
* **Selection-rate ratio**: every base resume is scored under every group, the top 25% of all
  renderings is shortlisted, and a group's rate is the share of its renderings shortlisted.
  Ratio = lowest group rate / highest. Merit is identical across groups, so a gap is caused by
  the identity cue alone. **Worst-case ratio** is the minimum across the three dimensions.
* **AUC vs merit**: held-out check that redaction does not wreck usefulness.

## Limitations (say these in interviews)

* The data is **synthetic with injected bias**. The numbers show the audit pipeline working
  on a controlled setup, not the fairness of any real hiring system. State "synthetic" next
  to the numbers.
* Redaction is lexicon-based and the generator only uses cues on the lexicon, so full
  redaction gives exactly 0% rank change and a ratio of 1.00 **by construction**. Real resumes
  leak identity through clubs, locations, employment gaps, writing style and more.
* Removing names alone does not fix gender or college effects; the report shows each level.
* The four-fifths rule is a screening heuristic from US employment guidance, not proof of
  fairness or a legal conclusion.
* Name lists and college tiers are illustrative assumptions; edit them in `data.py`.
