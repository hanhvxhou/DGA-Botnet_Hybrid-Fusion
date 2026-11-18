# Multi-seed experiments

All models in the revised manuscript were trained with five seeds (42–46). The scripts read the seed and the output folder from environment variables. The defaults reproduce the original single-seed behaviour (seed 42).

| Script | Environment variables | Default output |
|---|---|---|
| `M1.py` … `M5.py` | `DGA_SEED`, `DGA_OUTPUT_DIR` | `out_M<k>_seed<N>/` |
| `selectAlgorithm.py` | `DGA_SEED`, `DGA_SELECT_ROOT`, `DGA_CANDIDATES` (e.g. `bilstm,distilbert,bert`), `DGA_BERT_RANDOM_INIT=1` | `out_select/` (seed 42), `out_select_seed<N>/` |
| `M6_v3_Markov.py` | `DGA_SEED`, `DGA_OUTPUT_DIR`, `DGA_SELECT_ROOT` | `out_M6_v3_Markov_seed<N>/` |

## 1. Baselines M1–M5 and M6 with fixed backbones

```powershell
.\run_multiseed_v3.ps1
```

This runs M1–M5 and M6 for seeds 42–46. It skips any run whose `summary.txt` already exists.

## 2. Backbones retrained per seed (used for BiLSTM, DistilBERT, BERT and M6 in the paper)

```powershell
.\run_fullseed.ps1
```

For each seed 43–46, this script does two things:

1. It retrains the BiLSTM, DistilBERT and BERT candidates (`out_select_seed<N>/`).
2. It trains M6 on these backbones (`out_M6_v3_Markov_fullseed<N>/`).

Seed 42 uses `out_select/` and `out_M6_v3_Markov_seed42/`.

## 3. Pretraining ablation (randomly initialised DistilBERT)

```powershell
$env:DGA_SEED = "42"; $env:DGA_SELECT_ROOT = "out_select_rand_seed42"
$env:DGA_CANDIDATES = "distilbert"; $env:DGA_BERT_RANDOM_INIT = "1"
python selectAlgorithm.py
```

The architecture, tokenizer and training recipe are unchanged; only the pretrained weights are replaced by a random initialisation. `revision/code/run_revision.ps1` runs this step for all five seeds.

## 4. Aggregation

`revision/code/run_revision.ps1` scores every model on every evaluation set for each seed. `revision/code/analyze_revision.py` then produces the tables of the revised manuscript. See `revision/README.md`.

On Windows, set `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` if you capture the console output. The runners do this automatically.
