#!/usr/bin/env python3
import argparse, csv, re
from pathlib import Path

IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}

def list_imgs(d: Path):
    if not d.exists(): return []
    return sorted([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])

def write_csv(pairs, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input","gt"])
        for a,b in pairs:
            w.writerow([str(a), str(b)])
    print(f"[OK] {out_csv}  pairs={len(pairs)}")

def build_euvp_paired(euvp: Path):
    pairs_train = []
    pairs_val = []
    paired_root = euvp / "Paired"
    for subset in ["underwater_dark","underwater_imagenet","underwater_scenes"]:
        base = paired_root / subset
        a_tr = base / "trainA"
        b_tr = base / "trainB"
        vdir = base / "validation"

        # trainA/trainB
        for a in list_imgs(a_tr):
            b = b_tr / a.relative_to(a_tr)
            if b.exists():
                pairs_train.append((a,b))

        # validation: dataset uses "validation" only (not separate A/B),
        # so we use it only for visuals unless you want to split manually.
        # We'll still collect input/gt pairs if validation contains A/B structure (rare).
        # Otherwise ignore.
        if (vdir / "valA").exists() and (vdir / "valB").exists():
            for a in list_imgs(vdir/"valA"):
                b = (vdir/"valB") / a.relative_to(vdir/"valA")
                if b.exists():
                    pairs_val.append((a,b))
    return pairs_train, pairs_val

def build_uieb_pairs(uieb: Path):
    pairs = []
    raw = uieb / "raw-890"
    ref = uieb / "reference-890"
    for a in list_imgs(raw):
        b = ref / a.relative_to(raw)
        if b.exists():
            pairs.append((a,b))
    return pairs

def build_euvp_test(euvp: Path):
    pairs = []
    inp = euvp / "test_samples" / "Inp"
    gt  = euvp / "test_samples" / "GTr"
    for a in list_imgs(inp):
        b = gt / a.relative_to(inp)
        if b.exists():
            pairs.append((a,b))
    return pairs

def build_teacher_pseudo_exact(inputs_dir: Path, teacher_clean: Path):
    """
    Pseudo pairs by exact filename match: input_name == teacher_output_name
    """
    pairs = []
    teacher_map = {p.name: p for p in list_imgs(teacher_clean)}
    for a in list_imgs(inputs_dir):
        if a.name in teacher_map:
            pairs.append((a, teacher_map[a.name]))
    return pairs

def build_teacher_pseudo_idmatch(inputs_dir: Path, teacher_clean: Path):
    """
    Pseudo pairs by matching first integer id found in filenames.
    Safe: only creates pair if id maps uniquely in both sets.
    Works with teacher names like SUIM__d_r_103_.png and input names that also contain 103.
    """
    def extract_id(name: str):
        m = re.search(r"(\d+)", name)
        return m.group(1) if m else None

    in_map = {}
    for a in list_imgs(inputs_dir):
        i = extract_id(a.name)
        if i:
            in_map.setdefault(i, []).append(a)

    t_map = {}
    for t in list_imgs(teacher_clean):
        i = extract_id(t.name)
        if i:
            t_map.setdefault(i, []).append(t)

    pairs = []
    # only unique-to-unique
    for i, alist in in_map.items():
        tlist = t_map.get(i, [])
        if len(alist)==1 and len(tlist)==1:
            pairs.append((alist[0], tlist[0]))
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--euvp", required=True)
    ap.add_argument("--uieb", required=True)
    ap.add_argument("--suim", required=True)
    ap.add_argument("--usod", required=True)
    ap.add_argument("--teacher_clean", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--use_teacher_pseudo", action="store_true")
    ap.add_argument("--teacher_mode", choices=["exact","idmatch"], default="exact")
    args = ap.parse_args()

    euvp = Path(args.euvp)
    uieb = Path(args.uieb)
    suim = Path(args.suim)
    usod = Path(args.usod)
    teacher = Path(args.teacher_clean)
    out = Path(args.out_dir)

    # core paired
    euvp_train, euvp_val = build_euvp_paired(euvp)
    uieb_pairs = build_uieb_pairs(uieb)
    test_pairs = build_euvp_test(euvp)

    train_pairs = []
    train_pairs += euvp_train
    train_pairs += uieb_pairs

    # OPTIONAL pseudo pairs
    pseudo_pairs = []
    if args.use_teacher_pseudo and teacher.exists():
        # try pseudo from SUIM train_val/images (best) + EUVP Unpaired trainA (optional)
        suim_in = suim / "train_val" / "images"
        euvp_unpaired_A = euvp / "Unpaired" / "trainA"
        usod_images = usod / "images"

        if args.teacher_mode == "exact":
            pseudo_pairs += build_teacher_pseudo_exact(suim_in, teacher)
            pseudo_pairs += build_teacher_pseudo_exact(euvp_unpaired_A, teacher)
            pseudo_pairs += build_teacher_pseudo_exact(usod_images, teacher)
        else:
            pseudo_pairs += build_teacher_pseudo_idmatch(suim_in, teacher)

        # Add pseudo pairs to training
        train_pairs += pseudo_pairs

    # Write CSVs
    write_csv(train_pairs, out / "train_pairs.csv")
    write_csv(test_pairs, out / "test_pairs.csv")
    write_csv(uieb_pairs, out / "uieb_pairs.csv")
    write_csv(euvp_train, out / "euvp_paired_train.csv")
    write_csv(pseudo_pairs, out / "teacher_pseudo_pairs.csv")

    print("\nSummary:")
    print("  train_pairs:", len(train_pairs))
    print("  test_pairs :", len(test_pairs))
    print("  euvp_train :", len(euvp_train))
    print("  uieb_pairs :", len(uieb_pairs))
    print("  pseudo     :", len(pseudo_pairs))

if __name__ == "__main__":
    main()
