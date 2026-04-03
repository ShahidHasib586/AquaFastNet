#!/usr/bin/env python3
import argparse, csv, random
from pathlib import Path

def read_pairs(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["input"], row["gt"]))
    return rows

def write_pairs(rows, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input","gt"])
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.03)  # 3% val
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_pairs(args.in_csv)
    random.seed(args.seed)
    random.shuffle(rows)

    n_val = max(1, int(len(rows) * args.val_ratio))
    val = rows[:n_val]
    train = rows[n_val:]

    write_pairs(train, args.out_train)
    write_pairs(val, args.out_val)
    print(f"Split: total={len(rows)} train={len(train)} val={len(val)} ratio={args.val_ratio}")

if __name__ == "__main__":
    main()
