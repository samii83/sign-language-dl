import os
import glob
import numpy as np
import tensorflow as tf
import cv2


def representative_data_gen_from_dir(data_dir, img_size=(128, 128), num_samples=100):
    files = glob.glob(os.path.join(data_dir, '*', '*'))
    if not files:
        return None

    def gen():
        count = 0
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            yield [img]
            count += 1
            if count >= num_samples:
                break

    return gen


def convert_to_tflite(saved_model_dir, tflite_path, quantize=True, representative_data_dir=None):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_data_dir:
            rep = representative_data_gen_from_dir(representative_data_dir, num_samples=100)
            if rep:
                converter.representative_dataset = rep
        # Use default supported types
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                               tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print('Saved TFLite model to', tflite_path)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--saved_model', required=True)
    p.add_argument('--out', default='model.tflite')
    p.add_argument('--data_dir', default=None, help='Representative dataset directory for quantization')
    args = p.parse_args()

    convert_to_tflite(args.saved_model, args.out, quantize=True, representative_data_dir=args.data_dir)
