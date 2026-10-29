# BSc Deep Learning — Course Projects

Applied projects from the **Deep Learning** course (BSc Computer Engineering, University of Isfahan):
image classification with transfer learning, ECG time-series classification with recurrent networks,
and real-time object detection with YOLO.

## Contents

| Folder | Task | Approach | Key result |
|---|---|---|---|
| [`smile-detection`](smile-detection) | Smile / no smile from a face image | MTCNN face alignment + MobileNetV3Large transfer learning, three-phase fine-tuning | Test accuracy 92.0 %, AUC 0.965 (400 images) |
| [`ecg-heartbeat-classification/mitbih-arrhythmia`](ecg-heartbeat-classification/mitbih-arrhythmia) | Five-class heartbeat (arrhythmia) classification | Class balancing + Bi-GRU/GRU network on 187-sample beats | Test accuracy 97 %, macro-F1 0.85 (21,892 beats) |
| [`ecg-heartbeat-classification/ptbdb-normal-vs-abnormal`](ecg-heartbeat-classification/ptbdb-normal-vs-abnormal) | Normal vs. abnormal heartbeat | Conv1D + two Bi-GRU layers | Test accuracy 99.6 % (2,183 beats) |
| [`hand-gesture-yolo`](hand-gesture-yolo) | Detect the digits 1–5 and the "OK" sign made with a hand, in real time | YOLOv8n trained on a self-collected dataset that was labelled automatically with MediaPipe | mAP50 0.993 on the collected images (no held-out split — see its README) |

Each project folder has its own README with the data, the approach, the results, the limitations and how to run it.

```
BSc-Deep-Learning/
├── smile-detection/
├── ecg-heartbeat-classification/
│   ├── mitbih-arrhythmia/
│   └── ptbdb-normal-vs-abnormal/
├── hand-gesture-yolo/
├── requirements.txt
└── README.md
```

## What is (and is not) in this repository

The repository contains the code, the notebooks with their saved outputs, and the small trained models.
Large or private files are left out on purpose:

| Not included | Why | Where to get it |
|---|---|---|
| Datasets of the smile and ECG projects | Public datasets, too large to store here (the ECG CSV files alone are about 550 MB) | See the "Data" section of each project README |
| Dataset of the hand-gesture project | Collected and labelled by ourselves, not published | Collect your own with `hand-gesture-yolo/collect_data.py` |
| Demo videos | Several tens of MB each | — |
| Trained weights of the smile, PTB-DB and gesture-YOLO models | 6–38 MB each | Re-run the notebooks to train them |

Only the small MIT-BIH model is included:
`ecg-heartbeat-classification/mitbih-arrhythmia/ecg_mitbih_model.keras` (0.9 MB).

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

A GPU is strongly recommended for training. The notebooks were run on Google Colab, Kaggle and a local machine;
each project README says how to run it. The demo scripts need a webcam or a video file.

## Author

Fatemeh Sayadzadeh — [GitHub](https://github.com/Ftm-Sayadzadeh)
