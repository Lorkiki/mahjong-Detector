# mahjong-Detector
This project was created to solve automated Majong analysis: recognizing tile types from real-world images. It uses a custom-trained YOLO model with optimized preprocessing, data augmentation, and small-object tuning to detect tile classes. 

## Dataset Attribution & Usage
- This project uses a Mahjong tile dataset originally created by <https://github.com/Camerash/mahjong-dataset>
- The dataset is included in this repository under the terms of the original license, which is provided in the LICENSE file for full attribution

## Dataset Structure
- The dataset used for training is organized under the train.zip
- Inside train, there are two key components:

```
train/
├── images/
│ ├── 1.jpg
│ ├── 2.jpg
│ ├── 3.jpg
│ └── ...
└── data.csv
```

1. Images Folder
Description:
Contains the raw Mahjong tile images used for training
Each image is named numerically
These images correspond directly to entries in the data.csv file
2. CSV Metadata File

| Column       | Description                                 |
| ------------ | ------------------------------------------- |
| `image-name` | Filename of the image (e.g., `1.jpg`)       |
| `label`      | Numerical label ID (e.g., `0`, `1`, `2`, …) |
| `label-name` | Human-readable tile name (e.g., `bamboo-3`) |


## Convert Dataset
- Transforms the original dataset into a clean and YOLO-ready format.
- Filsters the dataset to include only Fujian Mahjong tiles
- Place convert_dataset.py on project folder
- Run the script to build dataset
```
python convert_dataset.py \
--csv PATH of csv file
--images PATH of images folder
--dst CREATE a folder name for clean dataset
--val 0.2 Vadidation split ratio
--seed 42 random seed for splitting
```

## Training the Data
For Macbook
```
yolo detect train \
    model=yolo11s.pt \
    data=dst/data.yaml \
    epochs=100 \
    imgsz=1024 \
    batch=-1 \
    device=mps \
    name=mahjong_y11s 
    cos_lr=True \
    patience=20 
```

## Predict the Image
- best.pt is result of training
- Run the command to predict images

```
yolo detect predict \
model=runs/detect/train/weights/best.pt \
source="PATH of image" \
save=True \
device=mps
```

## Host a website and show result
run command to create environment and download needed requirements
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export YOLO_MODEL=best.pt
python app.py
```

## License
Open sourced under MIT License
