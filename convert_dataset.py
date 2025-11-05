#!/usr/bin/env python3
import csv, shutil, re
from pathlib import Path

# ----------------------------
# Settings
# ----------------------------
CSV = Path("/Users/dreamerricher/Downloads/newnew/train/data.csv")
IMAGES = Path("/Users/dreamerricher/Downloads/newnew/train/images")
OUT = Path("ready_fujian_mahjong_dataset")  

# If the output directory doesn't exist, create it
# If it exists, its contents may be overwritten
OUT.mkdir(exist_ok=True)

# ----------------------------
# Fujian Mahjong classes to keep
# ----------------------------
DOTS = [f"dots-{i}" for i in range(1, 10)]
BAMBOO = [f"bamboo-{i}" for i in range(1, 10)]
CHAR = [f"characters-{i}" for i in range(1, 10)]
WINDS = ["honors-east", "honors-south", "honors-west", "honors-north", "honors-red"]


ALLOWED = set(DOTS + BAMBOO + CHAR + WINDS)


# ----------------------------
# Main processing
# ----------------------------
included = 0
skipped = 0

# with statements auto open and close files
# CSV.open handles csv file
# as f assigns the opened file to variable f
with CSV.open(newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        img = row.get("image-name")
        label_name = row.get("label-name")
        if not img or not label_name:
            continue

        cls = label_name.strip()

        if cls not in ALLOWED:
            skipped += 1
            continue
        

        # pathlib.Path objects for source and destination
        # / operator joins paths
        src = IMAGES / img
        if not src.exists():
            print(f"[WARN] missing: {src}")
            skipped += 1
            continue

        dst_dir = OUT / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Copy the image to the destination directory
        shutil.copy2(src, dst_dir / Path(img).name)
        included += 1
        
print("✅ Done.")
print(f" - Included {included} images")
print(f" - Skipped {skipped} images (not part of Fujian Mahjong set)")
print(" - Output per-class dataset:", OUT.resolve())
