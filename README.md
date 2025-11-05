# mahjong-Detector
Mahjong Tile Detector built with YOLO.

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
- Transforms the original dataset into a clean and YOLO-ready classification format.
- Filsters the dataset to include only Fujian Mahjong tiles
- Place convert_dataset.py on project folder
- Edit the CSV and IMAGES Path in the convert_dataset.py to your own path
- Run the script to build dataset
- python convert_dataset.py


## Training the Data
For NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

For Mac, just need YOLO
pip install ultralytics tensorboard



## License
Open sourced under MIT License
