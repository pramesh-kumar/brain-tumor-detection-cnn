# Brain Tumor Detection using DIP and CNN

A deep learning project that combines Digital Image Processing (DIP) techniques with Convolutional Neural Networks (CNN) to detect and classify brain tumors from MRI scans.

## 🎯 Project Overview

This system classifies brain MRI scans into 4 categories:
- **Glioma**: A type of brain tumor
- **Meningioma**: Tumor in brain/spinal cord membranes  
- **No Tumor**: Healthy brain scan
- **Pituitary**: Tumor in pituitary gland

## 🔧 Technical Stack

- **Deep Learning**: TensorFlow/Keras
- **Image Processing**: OpenCV
- **Web Interface**: Streamlit
- **Data Science**: NumPy, Matplotlib, Scikit-learn

## 📊 Dataset

- **Source**: Kaggle Brain Tumor MRI Dataset
- **Size**: ~3,000 MRI images
- **Classes**: 4 (Glioma, Meningioma, No Tumor, Pituitary)
- **Format**: JPG images, various sizes

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Kaggle API (Optional)
```bash
# Place kaggle.json in ~/.kaggle/
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download Dataset
```bash
cd src
python download_dataset.py
```

### 4. Train Model
```bash
python train.py
```

### 5. Run Web App
```bash
streamlit run app.py
```

## 🧠 Digital Image Processing Pipeline

1. **Grayscale Conversion**: Convert RGB to grayscale
2. **CLAHE**: Contrast Limited Adaptive Histogram Equalization
3. **Noise Removal**: Gaussian blur filtering
4. **Normalization**: Pixel values to [0,1] range
5. **Resizing**: Standardize to 224x224 pixels

## 🏗️ CNN Architecture

```
Input (224x224x1)
    ↓
Conv2D(32) → BatchNorm → MaxPool
    ↓
Conv2D(64) → BatchNorm → MaxPool
    ↓
Conv2D(128) → BatchNorm → MaxPool
    ↓
Conv2D(256) → BatchNorm → MaxPool
    ↓
Flatten → Dense(512) → Dropout(0.5)
    ↓
Dense(256) → Dropout(0.3)
    ↓
Dense(4) → Softmax
```

## 📈 Expected Performance

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~92%
- **Test Accuracy**: ~90%

## 🖥️ Web Interface Features

- Upload MRI scan images
- Real-time preprocessing visualization
- Confidence scores for all classes
- Medical disclaimer
- Responsive design

## 📁 Project Structure

```
brain_tumor_detection/
├── src/
│   ├── data_preprocessing.py    # DIP pipeline
│   ├── cnn_model.py            # CNN architecture
│   ├── train.py                # Training script
│   └── download_dataset.py     # Dataset downloader
├── models/                     # Saved models
├── data/                       # Dataset directory
├── app.py                      # Streamlit web app
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔬 Key Features

- **Digital Image Processing**: CLAHE, noise removal, normalization
- **Data Augmentation**: Rotation, flipping for better generalization
- **Transfer Learning Ready**: Easy to adapt for other medical imaging tasks
- **Web Deployment**: User-friendly Streamlit interface
- **Comprehensive Evaluation**: Confusion matrix, classification reports

## ⚠️ Medical Disclaimer

This tool is for educational and research purposes only. Always consult qualified medical professionals for proper diagnosis and treatment decisions.

## 🤝 Contributing

Feel free to contribute by:
- Improving model architecture
- Adding more preprocessing techniques
- Enhancing the web interface
- Adding new evaluation metrics

## 📄 License

This project is for educational purposes. Dataset usage follows Kaggle's terms of service.