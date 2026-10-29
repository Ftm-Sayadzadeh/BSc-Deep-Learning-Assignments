# Smile Detection — Transfer Learning with MobileNetV3

A binary classifier that decides whether a face is smiling, plus a real-time demo that finds the face in
each video frame, aligns it, and labels it **Smile** or **No Smile**.

## Data

The GENKI-4K smile dataset (about 4,000 face photos labelled *smile* / *non-smile*). The notebook expects the
Kaggle copy arranged as two folders, `smile/` and `non_smile/`, in the `RAW_DIR` it defines (a Google Drive path
in the notebook; change it to yours). The dataset is **not** included in this repository.

## Approach

1. **Face detection and alignment (MTCNN).** For every image the largest face is found, the image is rotated so the eyes are level, and the face is cropped with padding (18 % of the width, 25 % of the height). 3,986 usable face crops remain.
2. **Split.** Random 80 % / 10 % / 10 % split with seed 42: 3,188 train / 398 validation / 400 test images.
3. **Augmentation (training only).** Horizontal flip, rotation, zoom, translation, contrast and brightness changes.
4. **Model.** `MobileNetV3Large` (ImageNet weights) → global average pooling → Dropout 0.3 → Dense 256 (ReLU) → Dropout 0.2 → one sigmoid output. Images are resized to 224 × 224 and passed through MobileNetV3's own preprocessing.
5. **Three-phase fine-tuning** (Adam, binary cross-entropy):

   | Phase | What is trained | Epochs | Learning rate |
   |---|---|---|---|
   | 1 | New classification head only (backbone frozen) | 5 | 1e-3 |
   | 2 | Head + last 20 backbone layers | 15 | 5e-5 |
   | 3 | Head + last 30 backbone layers | 10 | 1e-5 |

   Checkpointing, early stopping and learning-rate reduction all monitor the **validation AUC**.
6. **Test-time augmentation (TTA).** The notebook also averages 5 prediction passes over the test set. Note that the test pipeline applies no augmentation, so the passes are essentially identical and this step cannot change the result.

## Results

Validation AUC improved steadily through the phases: 0.817 after phase 1, 0.943 after phase 2 and **0.958** after phase 3.

Test set (400 images):

| | Accuracy | AUC |
|---|---|---|
| Single pass | 0.920 | 0.965 |
| With 5× TTA | 0.920 | 0.965 |

- About 9 in 10 faces are classified correctly; with 400 test images each image is worth 0.25 points of accuracy and the figure is only accurate to around ±3 points.
- Fine-tuning mattered: with a frozen backbone the validation AUC stayed at about 0.82, and unfreezing the last layers lifted it above 0.95.
- TTA brought no gain (accuracy and AUC are unchanged). That is expected as implemented: the repeated passes see the same un-augmented images. For TTA to help, the test images would have to be randomly flipped/shifted on each pass.
- The demo script calls a face a smile only above a probability of **0.8**, which is stricter than the 0.5 used for the evaluation above. This trades some missed smiles for fewer false alarms in the live video.

## Files

| File | Purpose |
|---|---|
| `smile_detection.ipynb` | Face cropping, training (three phases), evaluation and TTA — written for Google Colab with a GPU (training took roughly 35–40 minutes on a T4) |
| `video_demo.py` | Runs the trained model on a video with MTCNN face detection and draws the label on every frame |

The trained model (`best_smile_improved.keras`, about 28 MB) is not included; the notebook produces it.

## Run

1. **Train:** open `smile_detection.ipynb` in Colab (GPU runtime), put the dataset in Drive and set `RAW_DIR`. The notebook installs `mtcnn` and `opencv-python`, crops the faces, trains the model and saves `best_smile_improved.keras`.
2. **Demo:** put `best_smile_improved.keras` and a video called `test-video.mp4` next to the script, then

   ```bash
   pip install -r ../requirements.txt
   python video_demo.py
   ```

   Press `q` to quit. To use the webcam instead of a video file, replace `"test-video.mp4"` in `cv2.VideoCapture(...)` with `0`.
