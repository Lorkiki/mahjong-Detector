#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path
import random
import shutil

'''
csv: read rows from a CSV file by column name.
argparse: parse --flags from the command line.
Pathlib.Path: path handling.
random: shuffling (for train/val split).
shutil: copying files
'''





CLASS_NAMES = (
    [f"bamboo-{i}" for i in range(1,10)]
    + [f"characters-{i}" for i in range(1,10)]
    + [f"dots-{i}" for i in range(1,10)]
    + ["east", "north", "red", "south", "west"]
)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}



# ----------------------------
# Allowed Fujian Mahjong subset from your CSV
# (bamboo/dots/characters 1..9 + east/south/west/north/red)
# ----------------------------
DOTS = [f"dots-{i}" for i in range(1, 10)]
BAMBOO = [f"bamboo-{i}" for i in range(1, 10)]
CHAR = [f"characters-{i}" for i in range(1, 10)]
WINDS_HONORS = ["honors-east", "honors-south", "honors-west", "honors-north", "honors-red"]
ALLOWED_RAW = set(DOTS + BAMBOO + CHAR + WINDS_HONORS)

# Map CSV labels to the final class names used by YOLO
NORMALIZE = {
    "honors-east": "east",
    "honors-south": "south",
    "honors-west": "west",
    "honors-north": "north",
    "honors-red": "red",
    # pass-through for numbered suits happens below
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def normalize_label(label: str) -> str | None:
    """Normalize CSV label to one of CLASS_NAMES (or None if not used)."""
    
    # handle None input
    l = (label or "").strip().lower()

    # convert honors- to east/north/south/west/red
    if l in NORMALIZE:
        return NORMALIZE[l]
    

    # allow-through suits already in desired format: bamboo-X, dots-X, characters-X
    if (
        l.startswith("bamboo-") or
        l.startswith("dots-") or
        l.startswith("characters-")
    ):
        return l
    return None

# creates YOLO format .txt label file for one tile, using normalized center coordinates and size
# for every image, YOLO expects a label file with one line per object 
# class_id center_x center_y width height
def write_label(label_path: Path, cls_id: int, x=0.5, y=0.5, w=0.9, h=0.9):
    """Write a YOLO txt label (full-image box by default)."""

    # check labels/ folder exists
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w", encoding="utf-8") as f:
        f.write(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        

def main():
    # create a CLI parser with description
    ap = argparse.ArgumentParser(description="Build Fujian Mahjong YOLO dataset from CSV in one step.")
    ap.add_argument("--csv", type=Path, required=True, help="Path to data.csv")
    ap.add_argument("--images", type=Path, required=True, help="Folder containing source images referenced by CSV")
    ap.add_argument("--dst", type=Path, required=True, help="Destination root for YOLO dataset")
    ap.add_argument("--val", type=float, default=0.2, help="Validation split ratio (default 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = ap.parse_args()

    random.seed(args.seed)

    # Collect eligible (src_path, class_id) from CSV
    rows = []
    included, skipped = 0, 0

    if not args.csv.exists():
        print(f"[ERR] CSV not found: {args.csv}")
        return

    with args.csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            img = row.get("image-name")
            label_name = row.get("label-name")
            if not img or not label_name:
                skipped += 1
                continue

            raw = label_name.strip().lower()
            if raw not in ALLOWED_RAW:
                skipped += 1
                continue

            cls = normalize_label(raw)
            if cls is None or cls not in CLASS_TO_ID:
                skipped += 1
                continue

            src = args.images / img
            if not src.exists() or src.suffix.lower() not in IMG_EXTS:
                print(f"[WARN] missing or unsupported image: {src}")
                skipped += 1
                continue

            rows.append((src, CLASS_TO_ID[cls]))
            included += 1

    if not rows:
        print("[ERR] No eligible images found. Check your --csv and --images paths and labels.")
        return

    # Shuffle and split
    random.shuffle(rows)
    n_val = int(len(rows) * args.val)
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    # Layout: dst/{train,val}/{images,labels}
    for split_name, split_set in [("train", train_rows), ("val", val_rows)]:
        for src_path, cls_id in split_set:
            stem = src_path.stem
            out_img = args.dst / split_name / "images" / (stem + src_path.suffix.lower())
            out_lbl = args.dst / split_name / "labels" / (stem + ".txt")
            out_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, out_img)
            write_label(out_lbl, cls_id)

    # Write reference class list
    (args.dst / "classes.txt").write_text(
        "\n".join(f"{i}\t{name}" for i, name in enumerate(CLASS_NAMES)),
        encoding="utf-8"
    )

    # Write data.yaml
    yaml_text = f"""# Auto-generated
path: {args.dst}
train: train/images
val: val/images
names:
"""
    for i, n in enumerate(CLASS_NAMES):
        yaml_text += f"  {i}: {n}\n"
    (args.dst / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("✅ Done.")
    print(f" - Included {included} images")
    print(f" - Skipped {skipped} rows")
    print(" - Train images:", len(train_rows))
    print(" - Val images:", len(val_rows))
    print(" - YOLO dataset:", args.dst.resolve())
    print(" - data.yaml:", (args.dst / 'data.yaml').resolve())

if __name__ == "__main__":
    main()
