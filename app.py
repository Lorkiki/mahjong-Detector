#!/usr/bin/env python3
import os, uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

# ---- YOLO / Torch
from ultralytics import YOLO
import torch
import cv2

APP_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = APP_DIR / "uploads"
RESULT_DIR = APP_DIR / "static" / "results"

UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
RESULT_DIR.mkdir(exist_ok=True, parents=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload cap
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

# ---- Load your trained model once
MODEL_PATH = os.environ.get("YOLO_MODEL", "best.pt")
model = YOLO(MODEL_PATH)

# Pick device automatically (MPS on Mac, else CUDA if available, else CPU)
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "0"
else:
    DEVICE = "cpu"

ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@app.route("/")
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part 'image'"}), 400
    f = request.files["image"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported file type {ext}"}), 400

    # save upload
    token = uuid.uuid4().hex
    fname = f"{token}{ext}"
    up_path = UPLOAD_DIR / secure_filename(fname)
    f.save(up_path)

    # run inference
    results = model.predict(
        source=str(up_path),
        device=DEVICE,
        imgsz=1024,      # larger helps tiny Mahjong tiles
        conf=0.25,       # adjust threshold as you like
        iou=0.45,        # NMS IoU
        verbose=False
    )

    res = results[0]
    # Make an annotated image
    annotated = res.plot()  # BGR numpy array
    out_name = f"{token}.jpg"
    out_path = RESULT_DIR / out_name
    cv2.imwrite(str(out_path), annotated)

    # Prepare detections JSON
    names = res.names
    dets = []
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            dets.append({
                "box": xyxy,
                "cls_id": cls_id,
                "cls_name": names.get(cls_id, str(cls_id)),
                "conf": round(conf, 4)
            })

    return jsonify({
        "result_url": url_for("static", filename=f"results/{out_name}"),
        "detections": dets,
        "device": DEVICE
    })
    
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
