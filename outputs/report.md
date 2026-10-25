# Fair-Hire Audit report

Synthetic benchmark. Seed 7; 6,000 training decisions; 1,000 base resumes
x 3 dimensions; reference pool 500; shortlist = top 25%.
Injected bias (logit units): gender 0.8, name group 0.6, college tier step 0.7.

## Overall by redaction level

| level | n_pairs | rank_change_pct | rank_change_95ci | shortlist_flip_pct | worst_selection_rate_ratio | worst_ratio_95ci | worst_dimension | auc_vs_merit |
|---|---|---|---|---|---|---|---|---|
| none | 3000 | 98.4 | 97.9-98.9 | 17.7 | 0.31 | 0.26-0.36 | college | 0.914 |
| names | 3000 | 66.6 | 66.6-66.7 | 12.8 | 0.28 | 0.25-0.33 | college | 0.939 |
| names+gender | 3000 | 33.3 | 33.3-33.3 | 8.7 | 0.28 | 0.24-0.33 | college | 0.949 |
| names+gender+college | 3000 | 0.0 | 0.0-0.0 | 0.0 | 1.00 | 1.00-1.00 | gender | 0.977 |

## Per dimension

| level | dimension | rank_change_pct | mean_abs_rank_shift_pct | shortlist_flip_pct | selection_rates | selection_rate_ratio | passes_four_fifths |
|---|---|---|---|---|---|---|---|
| none | gender | 99.4 | 14.57 | 17.3 | {"female": 0.164, "male": 0.336} | 0.49 | False |
| none | name | 95.9 | 9.62 | 11.8 | {"A": 0.3, "B": 0.2} | 0.67 | False |
| none | college | 100.0 | 21.27 | 23.9 | {"1": 0.383, "2": 0.247, "3": 0.12} | 0.31 | False |
| names | gender | 99.9 | 10.04 | 11.8 | {"female": 0.195, "male": 0.305} | 0.64 | False |
| names | name | 0.0 | 0.00 | 0.0 | {"A": 0.25, "B": 0.25} | 1.00 | True |
| names | college | 100.0 | 22.51 | 26.5 | {"1": 0.39, "2": 0.252, "3": 0.108} | 0.28 | False |
| names+gender | gender | 0.0 | 0.00 | 0.0 | {"female": 0.25, "male": 0.25} | 1.00 | True |
| names+gender | name | 0.0 | 0.00 | 0.0 | {"A": 0.25, "B": 0.25} | 1.00 | True |
| names+gender | college | 100.0 | 22.97 | 26.1 | {"1": 0.383, "2": 0.259, "3": 0.108} | 0.28 | False |
| names+gender+college | gender | 0.0 | 0.00 | 0.0 | {"female": 0.25, "male": 0.25} | 1.00 | True |
| names+gender+college | name | 0.0 | 0.00 | 0.0 | {"A": 0.25, "B": 0.25} | 1.00 | True |
| names+gender+college | college | 0.0 | 0.00 | 0.0 | {"1": 0.25, "2": 0.25, "3": 0.25} | 1.00 | True |

## Identity tokens the matcher learned from biased history

| token | weight |
|---|---|
| meridian | +1.45 |
| arcadia | +1.20 |
| keller | +1.10 |
| matthew | +0.86 |
| sullivan | +0.79 |
| halcyon | +0.64 |
| his | +0.61 |
| greg | +0.46 |
| she | -0.63 |
| aisha | -0.68 |
| anjali | -0.69 |
| gupta | -0.74 |
| greenfield | -0.80 |
| siddiqui | -0.82 |
| her | -1.06 |
| northbridge | -1.35 |

## Resume bullets (level: names+gender+college)

* Tested the matcher on 3,000 paired synthetic resumes that differ only in name, gender cues or college tier; ranking changed for 98.4% of pairs.
* After removing names, gender cues and college before scoring, rank changes fell to 0.0% and the worst-case selection-rate ratio reached 1.00 against the 0.8 four-fifths threshold.
