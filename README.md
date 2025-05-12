# Remote-Sensing-Image-Scene-Classification-Meets-Deep-Learning
A deep learning-based approach for remote sensing image scene classification using CNN architectures such as VGG19 and MobileNetV2. This project leverages high-resolution satellite imagery and powerful feature extraction techniques to accurately classify various land use and land cover scenes.

This project applies deep learning techniques for classifying scenes in remote sensing images. By using pretrained convolutional neural networks (CNNs) such as VGG19 and MobileNetV2, the model accurately identifies various land use and land cover categories from high-resolution satellite imagery.

# Features
- Scene classification of remote sensing images
- Transfer learning with pre-trained models (VGG19, MobileNetV2)
- Image preprocessing and augmentation
- Confusion matrix and accuracy metrics for model evaluation
- Simple GUI integration (using Tkinter) for user interaction

# Navigation
- dataset/: Contains the remote sensing image dataset (organized by scene classes)
- models/: Includes trained model weights or saved models
- scripts/: Python scripts for training, testing, and evaluation
- gui/: GUI files built using Tkinter for image upload and prediction
- notebooks/: Jupyter notebooks for development and visualization
- requirements.txt: List of Python dependencies

# Installation

Follow the steps below to set up and run the project:

1. Clone the repository:
   bash
   git clone https://github.com/your-username/remote-sensing-scene-classification.git

2. Navigate to the project directory:
   cd remote-sensing-scene-classification

3. Create and activate a virtual environment (optional but recommended):
  On Windows:
     python -m venv venv
     venv\Scripts\activate

  On macOS/Linux:
      python3 -m venv venv
      source venv/bin/activate

4. Install required dependencies:
  pip install -r requirements.txt

5. Run training or prediction script:
  To train the model:
    python scripts/train_model.py

  To run prediction with GUI:
    python gui/app.py
