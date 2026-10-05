import os
import pandas as pd
import random
from pathlib import Path
import re

# --- CẤU HÌNH ---
DIR_A = Path(r"D:\job\pycharm\DGABotnet2k25\dataset\utlDga22Origin\DGA_Botnets_Domains")  # Thư mục A (Label 1)
DIR_B = Path(r"D:\job\pycharm\DGABotnet2k25\dataset\legitDomain")  # Thư mục B (Label 0)
OUT_DIR = Path(r"D:\job\pycharm\DGABotnet2k25\dataset\utlDga22Origin\DGA_Botnets_Domains\DataNew")
TEST_DIR = Path(r"D:\job\pycharm\DGABotnet2k25\dataset\utlDga22Origin\DGA_Botnets_Domains\DataNew\Test")

# Tạo thư mục
OUT_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(exist_ok=True)

# Tỷ lệ chia
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15
N_SAMPLES_PER_FILE_A = 2000
N_SAMPLES_B = 152000

# Khởi tạo danh sách chứa dữ liệu tổng
all_train, all_val, all_test = [], [], []
seen_domains = set()


def read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


# 1. Xử lý thư mục B (Label 0) - Lấy 154.000 mẫu
print("Đang xử lý thư mục B (Benign)...")
file_b = list(DIR_B.glob("*.txt"))[0]
domains_b = read_lines(file_b)
random.shuffle(domains_b)

count_b = 0
selected_b = []
for d in domains_b:
    if d not in seen_domains:
        seen_domains.add(d)
        selected_b.append(d)
        count_b += 1
    if count_b >= N_SAMPLES_B:
        break

# Chia tập B
n_train_b = int(len(selected_b) * TRAIN_RATIO)
n_val_b = int(len(selected_b) * VAL_RATIO)

all_train.append(pd.DataFrame({'domain': selected_b[:n_train_b], 'label': 0}))
all_val.append(pd.DataFrame({'domain': selected_b[n_train_b:n_train_b + n_val_b], 'label': 0}))
all_test.append(pd.DataFrame({'domain': selected_b[n_train_b + n_val_b:], 'label': 0}))

# 2. Xử lý thư mục A (Label 1) - 77 file
print("Đang xử lý thư mục A (DGA)...")
files_a = sorted(list(DIR_A.glob("*.txt")))

for file_path in files_a:
    domains = read_lines(file_path)
    # Loại bỏ trùng lặp với B và nội bộ
    unique_domains = [d for d in domains if d not in seen_domains]

    if len(unique_domains) < N_SAMPLES_PER_FILE_A:
        print(f"Cảnh báo: File {file_path.name} không đủ {N_SAMPLES_PER_FILE_A} mẫu.")
        continue

    # Lấy ngẫu nhiên 2000
    subset = random.sample(unique_domains, N_SAMPLES_PER_FILE_A)
    for d in subset: seen_domains.add(d)

    # Chia tỷ lệ
    n_t = int(N_SAMPLES_PER_FILE_A * TRAIN_RATIO)
    n_v = int(N_SAMPLES_PER_FILE_A * VAL_RATIO)

    train_d = subset[:n_t]
    val_d = subset[n_t: n_t + n_v]
    test_d = subset[n_t + n_v:]

    # Lưu vào danh sách tổng
    all_train.append(pd.DataFrame({'domain': train_d, 'label': 1}))
    all_val.append(pd.DataFrame({'domain': val_d, 'label': 1}))
    all_test.append(pd.DataFrame({'domain': test_d, 'label': 1}))

    # Lưu riêng file test vào thư mục Test
    clean_name = re.sub(r'-\d+', '', file_path.stem)
    new_name = clean_name + file_path.suffix
    pd.DataFrame(test_d).to_csv(TEST_DIR / new_name, index=False)

# 3. Gộp và xuất file CSV cuối cùng
print("Đang lưu các file CSV...")
pd.concat(all_train).sample(frac=1).to_csv(OUT_DIR / "train.csv", index=False, sep=';')
pd.concat(all_val).sample(frac=1).to_csv(OUT_DIR / "val.csv", index=False, sep=';')
pd.concat(all_test).sample(frac=1).to_csv(OUT_DIR / "test.csv", index=False, sep=';')

print(f"Hoàn thành! Dữ liệu nằm tại thư mục: {OUT_DIR}")