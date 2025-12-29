#  Lightweight DeepFake Detection System

A lightweight DeepFake detection system built using **PyTorch**, **OpenCV**, and **Gradio**. The project detects whether an image frame (or video frame) is **REAL**, **FAKE**, or **UNCERTAIN** based on model confidence.

---

## 🚀 Features

* ⚡ Lightweight CNN-based DeepFake detector
* 🖼️ Image-based inference (video frames supported)
* 📊 Confidence-aware predictions (**REAL / FAKE / UNCERTAIN**)
* 🌐 Interactive web UI using **Gradio**
* 🧪 Dummy dataset generation for quick testing

---

## 📸 Project Demo (UI Screenshots)

<p align="center">
  <img src="Screenshot%202025-12-20%20005434.png" width="45%" />
  <img src="Screenshot%202025-12-19%20203755.png" width="45%" />
  <img src="Screenshot%202025-12-29%20181830.png" width="45%" />
</p>

**Left → Right:** Upload Interface • Image Preview • Prediction Result with Confidence

---

## 🗂️ Project Structure

```
Lightweight-DeepFake-Detection-System/
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
```

---

## 🧠 Model Architecture

A lightweight **Convolutional Neural Network (CNN)** consisting of:

* 2 × Convolutional layers
* ReLU activation functions
* MaxPooling layers
* Fully connected classifier (2 classes: **REAL / FAKE**)

### 🎯 Design Goals

* Fast inference
* Low computational overhead
* Easy deployment on low-resource systems

---

## 🧪 Prediction Logic

The model outputs probabilities for both classes:

* **REAL probability**
* **FAKE probability**

### Decision Rules

* **REAL ✅** → confidence > threshold
* **FAKE ❌** → confidence > threshold
* **UNCERTAIN 🤔** → confidence below threshold

> ⚠️ The **UNCERTAIN** class is intentional to reduce false positives when the model is not confident.



---

## 📌 Use Cases

* DeepFake detection in facial images
* Educational demos for AI & Computer Vision
* Internship / hiring challenge submissions
* Research prototypes for media authenticity

---

## 🔮 Future Improvements

* Full video-level DeepFake detection
* Advanced architectures (EfficientNet / Vision Transformers)
* Explainable AI (Grad-CAM visualizations)
* Cloud deployment

---

## 👤 Author

**Samta Sakha**
*Data Science & AI Enthusiast*



