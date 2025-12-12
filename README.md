# ♻️ Smart Garbage Classification using Deep Learning

An end-to-end deep learning system for automatic garbage classification into six categories using transfer learning.  
The project covers the complete ML lifecycle: data exploration, model training, evaluation, and deployment with an interactive web interface.

---

## 🚀 Project Overview

Efficient waste sorting is critical for sustainable recycling systems.  
This project uses **deep learning and transfer learning (EfficientNetB0)** to classify garbage images into the following classes:

- Battery
- Glass
- Metal
- Organic
- Paper
- Plastic

The system achieves **~92% test accuracy** and is deployable as a **Streamlit web application** or **FastAPI service**.

---

## 🧠 Key Features

- 📊 Full EDA with class distribution and sample visualization
- 🏗️ Transfer Learning with EfficientNetB0
- 🔄 Two-stage training (frozen backbone + fine-tuning)
- 📈 Training & validation curve visualization
- 🧪 Robust evaluation (confusion matrix, classification report)
- 🖼️ Misclassified image analysis
- 💾 Production-ready model artifacts
- 🌐 Deployable web app (Streamlit / FastAPI)

---

## 🗂️ Dataset

- **Source:** Kaggle  
- **Name:** Garbage Classification (6 Classes)  
- **Total Images:** 4,650  
- **Classes:** 6 (balanced)

Dataset downloaded using **kagglehub**.

---

## 🏗️ Model Architecture

- Backbone: **EfficientNetB0 (ImageNet pretrained)**
- Input size: `224 × 224 × 3`
- Head: GlobalAveragePooling → Dropout → Dense (Softmax)
- Loss: Sparse Categorical Crossentropy
- Optimizer: Adam

---

## 📊 Results

| Metric | Value |
|------|------|
| Test Accuracy | **~92%** |
| Macro F1-Score | **0.92** |

Misclassifications mainly occur between visually similar materials (e.g., plastic vs metal).

---

## 🌐 Deployment

### Streamlit Web App
- Upload an image
- View predicted class & confidence
- Interactive and user-friendly UI

### FastAPI (Optional)
- REST API for inference
- `/health` and `/predict` endpoints
📌 Author

Muhammad Junaid Shah Bukhari
Public Administration & Governance | Data Science & AI
📍 Islamabad, Pakistan

