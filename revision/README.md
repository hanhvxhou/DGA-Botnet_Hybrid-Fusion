# Revision material

This folder holds everything needed to reproduce the tables that were added or changed in the revised manuscript (Tables 1–12, A1–A2).

## Contents

| Folder | Content |
|---|---|
| `code/` | Scripts for the revision analyses (see below). |
| `manifests/` | `train.csv`, `val.csv`, `test.csv`: the exact DataNew split used in the paper (`domain;label`). |
| `labels/` | Family and subtype labels, the external sample and the alias mapping (see below). |
| `perturbed/` | Test set with every suffix replaced by `.com` (`test_tldcom.csv`), with the suffix removed (`test_notld.csv`) and with a `www.` prefix (`test_www.csv`). |
| `results/` | Summary tables of all analyses, averaged over five seeds (see below). |
| `predictions/` | `test_predictions_seed<N>.csv.gz`: test-set probabilities and validation-selected thresholds of every model for each seed. Confusion matrices can be computed from these files. |

The scripts in `code/`:

- `revision_experiments.py`: builds the evaluation sets, implements the corrected Markov estimator, scores every model on every set for each seed, trains the early-fusion models and measures latency and energy.
- `run_revision.ps1`: runs the whole pipeline.
- `make_family_map.py`: assigns a DGA family to each test domain.
- `handcraft_corrected.py`: corrected consonant set plus the begin/end-of-string Markov estimator; retrains XGBoost for five seeds.
- `analyze_revision.py`, `ext_analysis.py`, `ben2.py`: offline analyses that produce the tables.

The files in `labels/`:

- `test_family_map.csv`: DGA family of every test domain.
- `test_dga_subtype_inputs.csv`: length, entropy, vowel ratio and digit ratio of every DGA test domain, i.e. the inputs of the subtype rule.
- `external_sampled_domains.csv`: the external domains sampled per family file (seed 42, at most 2,000 per file), with a flag `in_dev` for exact overlap with DataNew.
- `external_family_alias.csv`: alias reconciliation and seen/unseen assignment of each external family file.
- `external_overlap.csv`: exact-domain overlap of each full external family file with train, validation and test.

The tables in `results/`:

| File | Content |
|---|---|
| `summary_table.csv`, `all_runs.csv` | All models and fusion variants, five seeds (Tables 1–4, 7, 8). |
| `paired.csv` | Paired t-test, bootstrap, family-cluster bootstrap and McNemar test (Table 6). |
| `calibration.csv`, `temperature_scaling.csv` | Calibration (Section 6.6). |
| `disagreement.csv`, `per_family_errors.csv` | Branch disagreement and per-family error changes (Table 9). |
| `external_v2_summary.csv`, `external_v2_runs.csv`, `external_M6_perfamily_seed42_novel.csv` | External detection rates (Table 5). |
| `benign_v2.csv` | Independent benign sets (Table 10). |
| `robustness.csv` | Suffix and parsing perturbations (Table 11). |
| `latency.csv`, `latency_env.json` | Measured cost (Table 12). |
| `hcfix_results.csv` | Corrected handcrafted branch and fusion search (Section 6.7). |
| `tokenization_stats.csv`, `tokenization_examples.csv` | Table A2. |
| `feature_examples.csv`, `markov_spec.json` | Table A1 and the Markov specification. |

## How to reproduce

1. Train all models with five seeds (see `../MULTISEED.md`).
2. Copy the scripts of `code/` to the project root. Place DataNew in `DataNew/` (or use `manifests/`) and the external collections in `test_external/<dataset>/<family>.txt`.
3. Run `.\run_revision.ps1`. It takes about 5–6 h on an RTX 5070 Ti. It downloads the Tranco and Cisco Umbrella lists, builds the evaluation sets, scores all models for seeds 42–46, trains the randomly initialised DistilBERT, measures latency and writes `rev_out/`.
4. Run the offline analyses:
   ```bash
   python analyze_revision.py . analysis
   python ext_analysis.py . analysis
   python ben2.py . analysis
   python handcraft_corrected.py
   ```

The independent benign sets are not redistributed. They are regenerated deterministically (seed 42) from the Tranco and Cisco Umbrella top-1M lists. Lists downloaded on a different date will give slightly different samples.

## Corrections made during the revision

- **Markov features.** The estimator now uses begin- (`^`) and end-of-string (`$`) symbols and add-one smoothing over 36 + 1 symbols, so every conditional distribution sums to one. The perplexity is defined for every input, including strings shorter than two or three characters.
- **Consonant set.** It no longer contains `o`.
- **Inference time.** The 1.001 ms reported for M6 in the original submission was the sum of the per-sample times of three candidates. It is replaced by the end-to-end measurements in Table 12.
- **Model selection.** The embedding-selection script ranked the candidates by test F1. The validation ranking, now reported, is the same.
- **Early fusion.** The 96.37% early-fusion figure came from an earlier dataset. Early fusion with the same inputs on DataNew gives 97.39–97.43%.
- **Metric naming.** F1 is the positive-class F1 (not micro-averaged). DGA-R and DGA-C denote the same subtype.

None of these corrections changes the selected M6 configuration: effective weights (0.5, 0.5, 0) and threshold 0.5 in all seeds.
