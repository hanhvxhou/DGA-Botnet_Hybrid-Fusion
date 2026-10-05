# Hướng dẫn patch M1-M5 + M6_v3_Markov cho multi-seed

## Mục đích

Chạy 5 seeds (42, 43, 44, 45, 46) cho M1-M6 để có mean ± std + bootstrap CI,
đáp ứng Reviewer 1 yêu cầu "stability across independently seeded training runs".

**selectAlgorithm KHÔNG chạy lại** — giữ nguyên seed 42, kết quả cũ.

## Thay đổi cần làm trong MỖI file (M1.py, M2.py, M3.py, M4.py, M5.py, M6_v3_Markov.py)

### Dòng cần sửa

Tìm 2 dòng này trong CẤU HÌNH section (khoảng dòng 50-120, tùy file):

```python
SEED = 42
OUTPUT_DIR = Path("out_M1")      # M1.py — tương tự cho M2, M3, M4, M5
```

### Thay bằng

```python
SEED = int(os.environ.get("DGA_SEED", 42))
OUTPUT_DIR = Path(os.environ.get("DGA_OUTPUT_DIR", f"out_M1_seed{SEED}"))
```

### Lưu ý theo từng file

| File | OUTPUT_DIR hardcoded | Đổi thành |
|------|---------------------|-----------|
| M1.py | `Path("out_M1")` | `Path(os.environ.get("DGA_OUTPUT_DIR", f"out_M1_seed{SEED}"))` |
| M2.py | `Path("out_M2")` | `Path(os.environ.get("DGA_OUTPUT_DIR", f"out_M2_seed{SEED}"))` |
| M3.py | `Path("out_M3")` | `Path(os.environ.get("DGA_OUTPUT_DIR", f"out_M3_seed{SEED}"))` |
| M4.py | `Path("out_M4")` | `Path(os.environ.get("DGA_OUTPUT_DIR", f"out_M4_seed{SEED}"))` |
| M5.py | `Path("out_M5")` | `Path(os.environ.get("DGA_OUTPUT_DIR", f"out_M5_seed{SEED}"))` |
| M6_v3_Markov.py | `Path("out_M6_v3_Markov")` | `Path(os.environ.get("DGA_OUTPUT_DIR", f"out_M6_v3_Markov_seed{SEED}"))` |

### Verify sau khi patch

Chạy thử 1 lần với seed 42 để confirm vẫn work:

```powershell
$env:DGA_SEED = 42
python M1.py
# Expect: output → out_M1_seed42/ (không phải out_M1/)
# F1 ≈ 93.24% (giống kết quả cũ)
```

Nếu OK → kick off full multi-seed bằng `run_multiseed.ps1`.

## Lưu ý về Markov cache

File M6_v3_Markov.py dùng cache `benign_markov.npz`. Script `run_multiseed.ps1`
đã tự xoá cache trước mỗi seed để đảm bảo Markov fit trên đúng data mỗi lần.

## Phục hồi sau khi xong

Sau khi hoàn tất multi-seed, nếu muốn khôi phục lại code gốc (hardcoded SEED=42):

```powershell
git checkout -- M1.py M2.py M3.py M4.py M5.py M6_v3_Markov.py
```

Hoặc giữ nguyên version patched — vẫn tương thích ngược (default SEED=42 nếu không set env).
