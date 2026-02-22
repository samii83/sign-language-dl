import cv2
import numpy as np
import tensorflow as tf
import time

from models.mediapipe_utils import create_hands_detector, detect_hand_bbox, crop_and_resize, draw_hand_landmarks


def load_labels_from_generator(generator):
    # invert class_indices mapping
    return {v: k for k, v in generator.class_indices.items()} if hasattr(generator, 'class_indices') else None


def run_webcam(model_path, label_map=None, input_size=(128, 128)):
    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(0)
    hands = create_hands_detector()

    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox, hand_landmarks = detect_hand_bbox(frame, hands)
        if bbox is not None:
            crop = crop_and_resize(frame, bbox, size=input_size)
            if crop is not None:
                inp = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
                inp = np.expand_dims(inp, axis=0)
                preds = model.predict(inp)[0]
                idx = int(preds.argmax())
                prob = float(preds[idx])
                label = str(idx) if label_map is None else label_map.get(idx, str(idx))

                # draw bbox and label
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {prob:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if hand_landmarks is not None:
            draw_hand_landmarks(frame, hand_landmarks)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Sign Language Demo', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    from training.dataset import create_generators

    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to saved Keras model (SavedModel or .h5)')
    p.add_argument('--data_dir', required=False, help='Optional: dataset dir to infer labels')
    args = p.parse_args()

    label_map = None
    if args.data_dir:
        train_gen, _ = create_generators(args.data_dir, img_size=(128, 128), batch_size=1)
        label_map = {v: k for k, v in train_gen.class_indices.items()}

    run_webcam(args.model, label_map=label_map, input_size=(128, 128))
