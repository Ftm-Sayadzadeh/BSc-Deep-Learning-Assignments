import cv2
import mediapipe as mp
import os
import numpy as np
import math
import glob

output_base_dir = "datasets"
train_images_dir = os.path.join(output_base_dir, "train", "images")
train_labels_dir = os.path.join(output_base_dir, "train", "labels")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)

CLASSES = ['one', 'two', 'three', 'four', 'five', 'ok']
CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}

existing_files = glob.glob(os.path.join(train_images_dir, "hand_*.jpg"))
if existing_files:
    indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
    frame_count = max(indices) + 1
else:
    frame_count = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print(f"Started! Next image will be saved as hand_{frame_count}.jpg")
print("Press 'Space' to capture. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    display_frame = frame.copy()
    current_frame_labels = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_type = handedness.classification[0].label 
            
            fingers_up = 0
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            
            if hand_type == 'Right':
                if thumb_tip.x < thumb_ip.x: fingers_up += 1
            else:
                if thumb_tip.x > thumb_ip.x: fingers_up += 1
                
            for tip_id in [8, 12, 16, 20]:
                if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                    fingers_up += 1

            dist_ok = math.sqrt((hand_landmarks.landmark[4].x - hand_landmarks.landmark[8].x)**2 + 
                                (hand_landmarks.landmark[4].y - hand_landmarks.landmark[8].y)**2)
            
            if dist_ok < 0.05:
                gesture = "ok"
            else:
                mapping = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
                gesture = mapping.get(fingers_up, "unknown")

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = max(0, min(x_coords)), min(1, max(x_coords))
            y_min, y_max = max(0, min(y_coords)), min(1, max(y_coords))

            p_w, p_h = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1
            x_min_y = max(0, x_min - p_w); x_max_y = min(1, x_max + p_w)
            y_min_y = max(0, y_min - p_h); y_max_y = min(1, y_max + p_h)

            if gesture != "unknown":
                c_x = (x_min_y + x_max_y) / 2
                c_y = (y_min_y + y_max_y) / 2
                w_y = x_max_y - x_min_y
                h_y = y_max_y - y_min_y
                current_frame_labels.append(f"{CLASS_MAP[gesture]} {c_x:.6f} {c_y:.6f} {w_y:.6f} {h_y:.6f}")

                cv2.rectangle(display_frame, (int(x_min_y*w), int(y_min_y*h)), 
                              (int(x_max_y*w), int(y_max_y*h)), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{hand_type}: {gesture}", (int(x_min_y*w), int(y_min_y*h)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Data Collection", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '): 
        if current_frame_labels:
            img_path = os.path.join(train_images_dir, f"hand_{frame_count}.jpg")
            lbl_path = os.path.join(train_labels_dir, f"hand_{frame_count}.txt")
            
            cv2.imwrite(img_path, frame)
            with open(lbl_path, 'w') as f:
                f.write("\n".join(current_frame_labels))
            
            print(f"Captured hand_{frame_count}.jpg")
            frame_count += 1
        else:
            print("No valid gesture detected to save!")
            
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()