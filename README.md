# Sign Language Real-time Recognition

This project provides a complete pipeline for real-time sign language recognition using TensorFlow/Keras, MediaPipe, and OpenCV.

Structure:
- `models/` - model and mediapipe utilities
- `training/` - dataset loader and training script
- `inference/` - single-image and webcam real-time demo
- `deployment/` - TFLite conversion and quantization
- `data/` - dataset (add your class subfolders here)

Quick start:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare dataset under `data/` with 12 class subfolders.

3. Train:

```bash
python training/train.py --data_dir data --epochs 20 --batch_size 32 --save_dir saved_model
```

4. Run webcam demo:

```bash
python inference/webcam_demo.py --model saved_model/saved_model --data_dir data
```

5. Convert to TFLite:

```bash
python deployment/tflite_converter.py --saved_model saved_model/saved_model --out model.tflite --data_dir data
```
# Real-Time Sign Language Recognition

CNN-based hand gesture classifier integrated with MediaPipe for real-time webcam inference.

## Features

- 12 gesture classes
- CNN architecture
- MediaPipe hand landmark extraction
- Real-time OpenCV webcam demo
- TensorFlow Lite conversion for edge deployment
- Post-training quantization

## Tech Stack

- TensorFlow
- OpenCV
- MediaPipe
- NumPy
- TFLite

## Project Structure

models/ → CNN architecture  
training/ → model training pipeline  
inference/ → real-time webcam inference  
deployment/ → TFLite conversion  