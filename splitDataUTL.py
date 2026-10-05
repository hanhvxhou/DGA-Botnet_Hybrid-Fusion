"""
splitDataUTL.py - Pre-split UTL_DGA22 + benign (Tranco) into DataNew train/val/test (70:15:15).

The script reads DGA domain files from a directory (one .txt per family, ~2000 domains each)
and benign domains from a single .txt file (first file found in --benign-dir), then produces:
  - <out-dir>/train.csv       70%
  - <out-dir>/val.csv         15%
  - <out-dir>/test.csv        15%
  - <test-dir>/<family>.txt   per-family test files (one row per domain)

All CSVs use ';' as separator with columns: domain, label (0 = benign, 1 = DGA).

Usage
-----
Default (relative paths):
    python splitDataUTL.py

Custom paths:
    python splitDataUTL.py \\
        --dga-dir  data/raw/UTL_DGA22/DGA_Botnets_Domains \\
        --benign-dir data/raw/benign \\
        --out-dir  DataNew \\
        --test-dir DataNew/Test

Environment variable fallback:
    DGA_HLF_DATA_ROOT: root data directory (default: ./data)
    If set, defaults become $DGA_HLF_DATA_ROOT/raw/UTL_DGA22/DGA_Botnets_Domains, etc.

See DATASET.md for dataset download instructions and expected directory structure.
"""

import argparse
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd


# --- Constants ---
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15
N_SAMPLES_PER_FILE_A = 2000  # Max DGA samples per family file
N_SAMPLES_B = 152000          # Max benign samples (target 50/50 class balance)


def read_lines(file_path):
    """Read non-empty lines from a text file (UTF-8)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def parse_args():
    data_root = Path(os.environ.get("DGA_HLF_DATA_ROOT", "data"))

    parser = argparse.ArgumentParser(
        description="Pre-split UTL_DGA22 + benign (Tranco) into DataNew train/val/test (70:15:15).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See DATASET.md for dataset preparation."
    )
    parser.add_argument(
        "--dga-dir",
        type=Path,
        default=data_root / "raw" / "UTL_DGA22" / "DGA_Botnets_Domains",
        help="Directory containing DGA family .txt files (default: %(default)s)"
    )
    parser.add_argument(
        "--benign-dir",
        type=Path,
        default=data_root / "raw" / "benign",
        help="Directory containing one .txt file of benign domains, "
             "e.g., Tranco Top 1M (default: %(default)s)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("DataNew"),
        help="Output directory for train/val/test CSVs (default: %(default)s)"
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Directory for per-family test .txt files "
             "(default: <out-dir>/Test)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility (default: %(default)s)"
    )
    return parser.parse_args()


def validate_paths(args):
    """Validate input directories exist and are non-empty."""
    errors = []

    if not args.dga_dir.exists():
        errors.append(
            f"DGA directory not found: {args.dga_dir}\n"
            f"  Download UTL_DGA22 and place family .txt files in this directory.\n"
            f"  See DATASET.md section 1 for details."
        )
    elif not list(args.dga_dir.glob("*.txt")):
        errors.append(
            f"No .txt files found in DGA directory: {args.dga_dir}\n"
            f"  Expected one file per family (e.g., bamital.txt, banjori.txt, ...).\n"
            f"  See DATASET.md section 1 for details."
        )

    if not args.benign_dir.exists():
        errors.append(
            f"Benign directory not found: {args.benign_dir}\n"
            f"  Download Tranco Top 1M and place at this location.\n"
            f"  See DATASET.md section 2 for details."
        )
    elif not list(args.benign_dir.glob("*.txt")):
        errors.append(
            f"No .txt files found in benign directory: {args.benign_dir}\n"
            f"  Expected at least one .txt file containing benign domains "
            f"(one per line).\n"
            f"  See DATASET.md section 2 for details."
        )

    if errors:
        print("ERROR: dataset validation failed:\n", file=sys.stderr)
        for err in errors:
            print(f"- {err}\n", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()

    # Resolve test_dir default
    if args.test_dir is None:
        args.test_dir = args.out_dir / "Test"

    # Validate
    validate_paths(args)

    # Create output dirs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.test_dir.mkdir(parents=True, exist_ok=True)

    # Fix seed for reproducibility
    random.seed(args.seed)

    print(f"Config:")
    print(f"  DGA dir     : {args.dga_dir}")
    print(f"  Benign dir  : {args.benign_dir}")
    print(f"  Output dir  : {args.out_dir}")
    print(f"  Test dir    : {args.test_dir}")
    print(f"  Seed        : {args.seed}")
    print()

    # Shared buffers
    all_train, all_val, all_test = [], [], []
    seen_domains = set()

    # ---- 1. Process benign directory (Label 0) ----
    print("Processing benign (Label 0)...")
    file_b = list(args.benign_dir.glob("*.txt"))[0]
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

    # Split benign
    n_train_b = int(len(selected_b) * TRAIN_RATIO)
    n_val_b = int(len(selected_b) * VAL_RATIO)

    all_train.append(pd.DataFrame({'domain': selected_b[:n_train_b], 'label': 0}))
    all_val.append(pd.DataFrame({'domain': selected_b[n_train_b:n_train_b + n_val_b], 'label': 0}))
    all_test.append(pd.DataFrame({'domain': selected_b[n_train_b + n_val_b:], 'label': 0}))

    # ---- 2. Process DGA directory (Label 1) - one file per family ----
    print("Processing DGA (Label 1)...")
    files_a = sorted(list(args.dga_dir.glob("*.txt")))

    for file_path in files_a:
        domains = read_lines(file_path)
        # Deduplicate against benign and across families
        unique_domains = [d for d in domains if d not in seen_domains]

        if len(unique_domains) < N_SAMPLES_PER_FILE_A:
            print(f"  Warning: {file_path.name} has fewer than "
                  f"{N_SAMPLES_PER_FILE_A} unique samples ({len(unique_domains)}), skipping.")
            continue

        # Random sample 2000
        subset = random.sample(unique_domains, N_SAMPLES_PER_FILE_A)
        for d in subset:
            seen_domains.add(d)

        # Split
        n_t = int(N_SAMPLES_PER_FILE_A * TRAIN_RATIO)
        n_v = int(N_SAMPLES_PER_FILE_A * VAL_RATIO)

        train_d = subset[:n_t]
        val_d = subset[n_t: n_t + n_v]
        test_d = subset[n_t + n_v:]

        all_train.append(pd.DataFrame({'domain': train_d, 'label': 1}))
        all_val.append(pd.DataFrame({'domain': val_d, 'label': 1}))
        all_test.append(pd.DataFrame({'domain': test_d, 'label': 1}))

        # Save per-family test file
        clean_name = re.sub(r'-\d+', '', file_path.stem)
        new_name = clean_name + file_path.suffix
        pd.DataFrame(test_d).to_csv(args.test_dir / new_name, index=False)

    # ---- 3. Concatenate, shuffle, save ----
    print("Saving CSVs...")
    train_df = pd.concat(all_train).sample(frac=1, random_state=args.seed)
    val_df = pd.concat(all_val).sample(frac=1, random_state=args.seed)
    test_df = pd.concat(all_test).sample(frac=1, random_state=args.seed)

    train_df.to_csv(args.out_dir / "train.csv", index=False, sep=';')
    val_df.to_csv(args.out_dir / "val.csv", index=False, sep=';')
    test_df.to_csv(args.out_dir / "test.csv", index=False, sep=';')

    print()
    print(f"Done. Output written to: {args.out_dir}")
    print(f"  train.csv : {len(train_df):>7d} rows")
    print(f"  val.csv   : {len(val_df):>7d} rows")
    print(f"  test.csv  : {len(test_df):>7d} rows")
    print(f"  per-family test files : {args.test_dir}")


if __name__ == "__main__":
    main()
