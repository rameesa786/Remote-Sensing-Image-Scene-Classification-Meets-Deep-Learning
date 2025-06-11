# 🛰️ Remote Sensing Image Scene Classification Meets Deep Learning

A **deep learning-based approach** for remote sensing image scene classification using advanced CNN architectures like **VGG19** and **MobileNetV2**. This project leverages **high-resolution satellite imagery** with powerful feature extraction to accurately classify various **land use** and **land cover scenes**.

## 🌍 Overview
By applying transfer learning on pretrained convolutional neural networks (CNNs), this project classifies remote sensing scenes into predefined categories. The models are trained on satellite imagery datasets to recognize patterns across different landscapes.

## 🚀 Features
- 🏞️ **Remote Sensing Image Scene Classification**
- 🧠 **Transfer Learning** with pretrained models (**VGG19**, **MobileNetV2**)
- 🎨 **Image Preprocessing & Augmentation** for better generalization
- 📊 **Confusion Matrix & Accuracy Metrics** for evaluation
- 🖥️ **Simple GUI (Tkinter)** for uploading images and viewing predictions

## 📁 Project Structure
```
├── dataset/         # 📂 Remote sensing image dataset (organized by scene classes)
├── models/          # 💾 Saved trained models
├── scripts/         # 📝 Python scripts for training/testing/evaluation
├── gui/             # 🎛️ GUI files built with Tkinter
├── notebooks/       # 📓 Development notebooks (optional)
└── requirements.txt # 📜 Python dependencies
```

## ⚙️ Installation

Follow the steps below to set up and run the project:

1️⃣ **Clone the repository**
```bash
git clone https://github.com/your-username/remote-sensing-scene-classification.git
cd remote-sensing-scene-classification
2️⃣ (Optional but recommended) Create and activate a virtual environment
```
Windows:
- python -m venv venv
- venv\Scripts\activate
  
macOS/Linux:
- python3 -m venv venv
- source venv/bin/activate

3️⃣ Install required dependencies
- pip install -r requirements.txt

4️⃣ Run the project
To train the model:
- python scripts/train_model.py

To launch the prediction GUI:
- python gui/app.py
---
## 📚 Requirements
All Python dependencies are listed in requirements.txt:
- TensorFlow / Keras
- scikit-learn
- Pillow
- numpy
- matplotlib
- tkinter (built-in with Python)
---
## 🔮 Future Scope
🌐 Integration of additional satellite datasets

⚡ Use of lightweight models for deployment on edge devices

🛰️ Real-time classification with streaming satellite data

📈 Automated reporting with visualization dashboards
