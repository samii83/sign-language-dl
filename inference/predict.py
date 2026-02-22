import argparse
import cv2
import numpy as np
import tensorflow as tf

def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def preprocess_image(img, target_size=(128, 128)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


def predict_image(model, img_path, class_map=None):
    img = cv2.imread(img_path)
    inp = preprocess_image(img, target_size=(model.input_shape[1], model.input_shape[2]))
    preds = model.predict(inp)[0]
    idx = int(preds.argmax())
    prob = float(preds[idx])
    label = str(idx) if class_map is None else class_map.get(idx, str(idx))
    return label, prob, preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--image', required=True)
    args = p.parse_args()

    model = load_model(args.model)
    label, prob, _ = predict_image(model, args.image)
    print(f'Prediction: {label} ({prob:.3f})')


if __name__ == '__main__':
    main()
