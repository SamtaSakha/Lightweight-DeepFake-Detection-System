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

Lightweight-DeepFake-Detection-Model/
│
├── models/
│   └── deepfake_cnn.pth              # Trained CNN model weights
│
├── src/
│   ├── data/
│   │   ├── create_dummy_data.py      # Generates sample training data
│   │   ├── dataset.py                # PyTorch Dataset class
│   │   ├── frame_extractor.py        # Extracts frames from videos
│   │   ├── preprocessing.py          # Image preprocessing utilities
│   │   └── run_frame_extraction.py   # Script to extract frames
│   │
│   ├── train/
│   │   ├── train.py                  # Model training logic
│   │   └── evaluate.py               # Model evaluation script
│   │
│   ├── inference/
│   │   ├── app.py                    # Gradio web application
│   │   ├── predict_image.py          # Image-based inference
│   │   └── predict_video.py          # Video-based inference
│   │
│   └── __init__.py
│
├── quick_train.py                    # Fast training script (demo use)
├── requirements.txt                  # Project dependencies
├── README.md                         # Project documentation
└── .venv/                            # Virtual environment (ignored)


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

http://127.0.0.1:7860/ 
