#!/usr/bin/env python3
import csv, shutil, random
from pathlib import Path

# ----------------------------
# Settings (edit these 3 paths)
# ----------------------------
CSV = Path("/Users/dreamerricher/Downloads/newnew/train/data.csv")
IMAGES = Path("/Users/dreamerricher/Downloads/newnew/train/images")
OUT = Path("mahjong-data")          # final dataset root (train/, val/, classes.txt, data.yaml)
VAL_RATIO = 0.2                     # 20% validation split
SEED = 42                           # reproducible split

# ----------------------------
# Fujian Mahjong classes to keep
# ----------------------------
DOTS = [f"dots-{i}" for i in range(1, 10)]
BAMBOO = [f"bamboo-{i}" for i in range(1, 10)]
CHAR = [f"characters-{i}" for i in range(1, 10)]
WINDS = ["honors-east", "honors-south", "honors-west", "honors-north", "honors-red"]
ALLOWED = sorted(set(DOTS + BAMBOO + CHAR + WINDS))  # stable order for classes.txt

# ----------------------------
# Helpers
# ----------------------------
def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

# ----------------------------
# 1) Read CSV and collect (img_path, class_name) records
# ----------------------------
random.seed(SEED)
records_by_class = {c: [] for c in ALLOWED}
included = 0
skipped = 0

with CSV.open(newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        img = row.get("image-name").strip()
        label_name = row.get("label-name").strip()
        if not img or not label_name:
            skipped += 1
            continue
        if label_name not in ALLOWED:
            skipped += 1
            continue

        src = IMAGES / img
        if not src.exists():
            print(f"[WARN] missing: {src}")
            skipped += 1
            continue

        records_by_class[label_name].append(src)
        included += 1

print(f"Scanned CSV. Included={included}, Skipped={skipped}")

# ----------------------------
# 2) Stratified split into train/val and copy
#    (YOLO classify expects folder-per-class under train/ and val/)
# ----------------------------
train_root = OUT / "train"
val_root = OUT / "val"
for cls, paths in records_by_class.items():
    if not paths:
        continue
    random.shuffle(paths)
    n = len(paths)
    n_val = max(1, int(n * VAL_RATIO)) if n > 1 else 1  # ensure at least 1 if possible
    val_paths = paths[:n_val]
    train_paths = paths[n_val:] if n > 1 else paths  # if only 1 sample, put it in train

    # copy files
    for p in train_paths:
        safe_copy(p, train_root / cls / p.name)
    for p in val_paths:
        safe_copy(p, val_root / cls / p.name)

# ----------------------------
# 3) Write classes.txt and data.yaml (for YOLO classify)
# ----------------------------
(OUT / "classes.txt").write_text("\n".join(ALLOWED), encoding="utf-8")

data_yaml = f"""# Ultralytics YOLO classification dataset
# You can pass either this YAML or just the OUT path to `yolo classify train`
path: {OUT.as_posix()}
train: train
val: val
names:
"""
for i, name in enumerate(ALLOWED):
    data_yaml += f"  {i}: {name}\n"

(OUT / "data.yaml").write_text(data_yaml, encoding="utf-8")

print("✅ Done.")
print(f" - Output root: {OUT.resolve()}")
print(f" - Train dir : {train_root.resolve()}")
print(f" - Val dir   : {val_root.resolve()}")
print(f" - classes.txt & data.yaml written.")

# ----------------------------
# 4) Next steps (commands)
# ----------------------------
print("\nNext (classification training):")
print("  yolo classify train model=yolo11s-cls.pt data=mahjong-data epochs=50 imgsz=224 batch=-1 device=0")
