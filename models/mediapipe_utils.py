import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def create_hands_detector(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5):
    return mp_hands.Hands(static_image_mode=static_image_mode,
                          max_num_hands=max_num_hands,
                          min_detection_confidence=min_detection_confidence,
                          min_tracking_confidence=0.5)


def detect_hand_bbox(image, hands):
    # Returns pixel bbox (x_min, y_min, x_max, y_max) or None
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    x_min = max(int(min(xs) * w) - 10, 0)
    x_max = min(int(max(xs) * w) + 10, w)
    y_min = max(int(min(ys) * h) - 10, 0)
    y_max = min(int(max(ys) * h) + 10, h)

    return (x_min, y_min, x_max, y_max), hand_landmarks


def crop_and_resize(image, bbox, size=(128, 128)):
    x_min, y_min, x_max, y_max = bbox
    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None
    resized = cv2.resize(crop, size)
    return resized


def draw_hand_landmarks(image, hand_landmarks):
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
