# Hand Gesture Detection with YOLOv8 — Digits 1–5 and the OK Sign

A real-time detector that finds a hand in the webcam image and tells which gesture it makes: the digits
**one, two, three, four, five** (by the number of raised fingers) or the **OK** sign. The dataset was collected and
labelled by ourselves; the labelling is automated with MediaPipe, and a YOLOv8n model is trained on the result.

> The dataset is **not** included in this repository. Use `collect_data.py` to build your own.

## How it works

### 1. Collecting and auto-labelling data — `collect_data.py`

The webcam image is analysed with **MediaPipe Hands** (up to two hands). For every hand the script
- counts the raised fingers: the thumb is compared along the x-axis (depending on whether it is a left or right hand), and each of the other four fingers is "up" when its tip is above its middle joint;
- recognises **OK** when the distance between the thumb tip and the index tip (in normalised image coordinates) is below 0.05, otherwise maps the finger count 1–5 to the digit classes;
- builds the hand's bounding box from the landmarks (with 10 % padding) and shows it with the predicted gesture.

Pressing **Space** saves the frame and a label file in YOLO format (`class x_center y_center width height`,
normalised); pressing **q** quits. Frames with no recognised gesture are not saved. In this way a labelled dataset is
produced without drawing any boxes by hand.

### 2. Training — `train_yolo.ipynb`

Fine-tunes the pretrained `yolov8n.pt` with the Ultralytics library for 50 epochs at image size 640 on a Google Colab
T4 GPU (about 7 minutes). The classes are defined in `data.yaml`: `one, two, three, four, five, ok`.

### 3. Real-time inference — `realtime_demo.py`

Loads `weights/best.pt`, runs the detector on every webcam frame (confidence threshold 0.5, image size 640) and
draws the boxes and labels. Press **q** to quit.

## Results

Reported by Ultralytics on the final epoch (307 images, 414 hands):

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| All classes | 0.999 | 0.997 | 0.993 | 0.977 |

Every class reaches mAP50 of at least 0.985 (mAP50-95 between 0.952 for `one` and 0.989 for `five`). Inference takes
about 2.2 ms per image on the T4 GPU, so real-time use on a webcam is comfortable.

### Important limitation

`data.yaml` sets the validation folder to the **same images as the training folder** (`val: train/images`). The
numbers above therefore measure how well the model fits the collected images, **not** how well it generalises to
new hands, people, lighting or backgrounds. The very high scores are expected in this situation and should not be
read as real-world accuracy. A proper evaluation needs a held-out set recorded in a different session (a random
split of video frames would not do, because consecutive frames are almost identical).

## Files

| File | Purpose |
|---|---|
| `collect_data.py` | Webcam tool that detects the gesture with MediaPipe and saves images with YOLO-format labels |
| `train_yolo.ipynb` | Colab notebook that trains YOLOv8n on the collected data |
| `data.yaml` | Dataset configuration (6 classes) |
| `realtime_demo.py` | Real-time webcam detection with the trained model |
| `weights/best.pt` | The trained YOLOv8n weights (6 MB) |

## Run

```bash
pip install -r ../requirements.txt

# 1. collect data with a webcam (Space = save frame, q = quit); images and labels go to datasets/train/
python collect_data.py

# 2. train: upload the dataset and data.yaml to Colab and run train_yolo.ipynb
#    (adjust the `path` in data.yaml to wherever your dataset folder is)

# 3. try the detector with the webcam
python realtime_demo.py
```

To use your own trained model, replace `weights/best.pt` (Ultralytics saves it in `runs/detect/train/weights/`).
