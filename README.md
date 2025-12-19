# Lightweight-DeepFake-Detection-System
A lightweight deepfake detection system built using PyTorch, OpenCV, and Gradio.
The project detects whether an image frame (or video frame) is REAL, FAKE, or UNCERTAIN based on model confidence.

This project was developed as part of the VoiceGuardAI Internship Hiring Challenge (Round 2).

##  Features

 Lightweight CNN-based deepfake detector

 Image-based inference (video frames supported)

 Confidence-aware predictions (REAL / FAKE / UNCERTAIN)

 Interactive Web UI using Gradio

 Dummy dataset generation for quick testing

📁 Clean, scalable project structure
Lightweight DeepFake Detection Model/
│
├── models/
│   └── deepfake_cnn.pth
│
├── src/
│   ├── data/
│   │   ├── create_dummy_data.py
│   │   ├── dataset.py
│   │   ├── frame_extractor.py
│   │   ├── preprocessing.py
│   │   └── run_frame_extraction.py
│   │
│   ├── train/
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── inference/
│   │   ├── app.py
│   │   ├── predict_image.py
│   │   └── predict_video.py
│   │
│   └── __init__.py
│
├── quick_train.py
├── requirements.txt
└── README.md

## 🧠 Model Architecture

A lightweight CNN with:

2 Convolutional layers

ReLU activations

MaxPooling

Fully connected classifier (2 classes: REAL / FAKE)

The model is intentionally lightweight for:
Fast inference
Low compute usage
Easy deployment

🧪 Prediction Logic

## The model outputs probabilities for both classes:

REAL probability

FAKE probability

Decision logic:

REAL ✅ → confidence > threshold

FAKE ❌ → confidence > threshold

UNCERTAIN 🤔 → low confidence

### ⚠️ The UNCERTAIN class is intentional to avoid false predictions when confidence is low.
