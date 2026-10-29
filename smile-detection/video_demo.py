import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import load_model


IMG_SIZE = (224, 224)

model = load_model("best_smile_improved.keras", compile=False)
print("Model Loaded ✔")


def align_face(img, left_eye, right_eye):
    lx, ly = int(left_eye[0]), int(left_eye[1])
    rx, ry = int(right_eye[0]), int(right_eye[1])
    center = ((lx + rx) // 2, (ly + ry) // 2)
    angle = np.degrees(np.arctan2(ry - ly, rx - lx))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def detect_crop(detector, img_rgb):
    dets = detector.detect_faces(img_rgb)
    if not dets:
        return None, None, None

    # pick largest face (same as dataset)
    det = max(dets, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = [int(v) for v in det['box']]

    # align
    kp = det.get('keypoints', {})
    if 'left_eye' in kp and 'right_eye' in kp:
        try:
            img_rgb = align_face(img_rgb, kp['left_eye'], kp['right_eye'])
        except:
            pass

    # paddings EXACT like training script
    pad_w, pad_h = int(w * 0.18), int(h * 0.25)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_rgb.shape[1], x + w + pad_w)
    y2 = min(img_rgb.shape[0], y + h + pad_h)

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    face = img_rgb[y1:y2, x1:x2]
    return face, (x1, y1, x2, y2), img_rgb


# ============================
# REAL-TIME LOOP
# ============================

detector = MTCNN()
cap = cv2.VideoCapture("test-video.mp4")
print("Webcam Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face, bbox, aligned = detect_crop(detector, rgb)

    if face is not None:
        (x1, y1, x2, y2) = bbox

        # --------- model prep ---------
        img = cv2.resize(face, IMG_SIZE)
        img = img.astype(np.float32)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0][0]
        print(pred)
        label = "Smile" if pred > 0.8 else "No Smile"
        color = (0, 255, 0) if pred > 0.8 else (0, 0, 255)

        # draw bbox on original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # show cropped
        cv2.imshow("Cropped Face", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

    cv2.imshow("Smile Detection", frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
