# Multi-seed tools cho IJICS revision

Bộ công cụ chạy multi-seed experiments + aggregation để trả lời Reviewer 1, 2.

## Thứ tự dùng

### Bước 1 — Patch M1-M6 (5 phút)

Mở từng file sau trong PyCharm, tìm 2 dòng:

```python
SEED        = 42
OUTPUT_DIR  = Path("out_M<n>")
```

Thay bằng:

```python
SEED        = int(os.environ.get("DGA_SEED", 42))
OUTPUT_DIR  = Path(os.environ.get("DGA_OUTPUT_DIR", f"out_M<n>_seed{SEED}"))
```

Files cần patch:
- M1.py (OUTPUT_DIR = `Path("out_M1")`)
- M2.py (OUTPUT_DIR = `Path("out_M2")`)
- M3.py (OUTPUT_DIR = `Path("out_M3")`)
- M4.py (OUTPUT_DIR = `Path("out_M4")`)
- M5.py (OUTPUT_DIR = `Path("out_M5")`)
- M6_v3_Markov.py (OUTPUT_DIR = `Path("out_M6_v3_Markov")`)

Chi tiết trong `PATCH_INSTRUCTIONS.md`.

### Bước 2 — Verify patch work (2 phút)

```powershell
$env:DGA_SEED = 42
python M1.py
# Kết quả nên vào folder: out_M1_seed42/
# Và F1 ≈ 93.24% (giống kết quả cũ)
```

### Bước 3 — Chạy multi-seed (chạy qua đêm, ~5-10 giờ)

```powershell
cd D:\job\pycharm\DGABotnet2k25\DGA-HLF
Copy-Item path\to\run_multiseed.ps1 .\
.\run_multiseed.ps1
```

Script sẽ:
- Loop qua seeds 42-46
- Mỗi seed xoá cache Markov cũ rồi chạy M1 → M2 → M3 → M4 → M5 → M6_v3_Markov
- Output: `out_<MODEL>_seed<N>/`
- Log file: `multiseed_log_<timestamp>.txt`
- Skip nếu seed đã có `summary.txt` (resume-friendly)

### Bước 4 — Aggregate kết quả (2 phút)

```powershell
Copy-Item path\to\aggregate_seeds.py .\
python aggregate_seeds.py
```

Output: folder `aggregate_results/` chứa:
- `table_binary_mean_std.csv` — Table 1 revised
- `table_per_family_mean_std.csv` — Table 2 revised
- `paired_ci_m6_vs_others.txt` — Bootstrap CI cho F1 differences
- `aggregated_summary.md` — Paste vào response letter

### Bước 5 — Disagreement analysis (1 phút)

```powershell
Copy-Item path\to\disagreement_analysis.py .\
python disagreement_analysis.py
```

Output: 
- `aggregate_results/disagreement_matrix.csv`
- `aggregate_results/disagreement_by_subtype.csv`
- `aggregate_results/disagreement_summary.md`

### Bước 6 — Upload kết quả cho tôi

Zip folder `aggregate_results/` và upload. Tôi sẽ:
- Merge số liệu vào bảng Table 1-5 revised
- Tracked-changes bài IJICS
- Hoàn thiện response letter

## File list

| File | Purpose |
|------|---------|
| `PATCH_INSTRUCTIONS.md` | Chi tiết cách patch M1-M6 |
| `run_multiseed.ps1` | PowerShell script chạy 5 seeds |
| `aggregate_seeds.py` | Gộp kết quả → mean/std + bootstrap CI |
| `disagreement_analysis.py` | Phân tích BiLSTM vs DistilBERT fusion |
| `README.md` | File này |

## Troubleshooting

### Markov cache issues

Nếu M6_v3_Markov cho F1 lạ ở seed 43-46:
- Check `benign_markov.npz` có bị xóa giữa các seeds không
- Script đã có `Remove-Item benign_markov.npz` trước mỗi seed, nhưng verify

### OOM (Out of Memory) ở M5

M5 là model nặng nhất. Nếu OOM:
- Giảm batch_size trong M5.py từ 256 → 128
- Hoặc skip M5 với 1 số seeds (báo trong response: "M5 completed 3 seeds due to GPU constraints")

### selectAlgorithm có cần retrain không?

**KHÔNG**. selectAlgorithm chỉ dùng để chọn embedding backbone (DistilBERT), không claim statistical significance. Giữ nguyên kết quả seed 42 cũ.

## Expected output example (sau aggregate)

```
table_binary_mean_std.csv:

Model                        n_seeds  Acc (%)         Pre (%)         Rec (%)         F1 (%)          AUC-ROC
M1 (RF + n-gram + PCA)       5        93.4X ± 0.XX    96.1X ± 0.XX    90.5X ± 0.XX    93.2X ± 0.XX    0.981X ± 0.0XXX
M2 (TF-IDF + MLP)            5        95.9X ± 0.XX    ...
M3 (RCNN-SPP)                5        96.5X ± 0.XX    ...
M4 (ATT-CNN-BiLSTM)          5        96.1X ± 0.XX    ...
M5 (Dual Transformer)        5        95.6X ± 0.XX    ...
M6 Hybrid Late Fusion        5        97.7X ± 0.XX    98.0X ± 0.XX    97.4X ± 0.XX    97.7X ± 0.XX    0.9967 ± 0.00XX
```

Variance (± 0.XX) sẽ là small nếu training stable. Nếu variance lớn (> 1pp) 
có thể signal overfitting hoặc data leakage — cần investigate.
