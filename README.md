# Brain Tumor Detection using Digital Image Processing and CNN

This project was developed to help medical professionals and researchers detect brain tumors from MRI scans using advanced machine learning techniques. I combined traditional digital image processing methods with modern deep learning to create an accurate classification system.

## What This Project Does

The system analyzes brain MRI scans and classifies them into four categories:
- **Glioma** - A serious type of brain tumor that starts in glial cells
- **Meningioma** - Tumors that develop in the protective membranes around the brain
- **No Tumor** - Healthy brain scans with no abnormalities detected
- **Pituitary Tumor** - Growths in the pituitary gland at the base of the brain

## Dataset Information

I used the Brain Tumor MRI Dataset from Kaggle, which contains:
- **Training Images**: 5,712 MRI scans
  - Glioma: 1,321 images
  - Meningioma: 1,339 images  
  - No Tumor: 1,595 images
  - Pituitary: 1,457 images
- **Testing Images**: 1,311 MRI scans
- **Total Dataset Size**: ~7,000 high-quality MRI images
- **Image Format**: JPG files with varying dimensions

## Technical Approach

### Image Preprocessing Pipeline
Before feeding images to the neural network, I implemented several digital image processing techniques:

1. **Grayscale Conversion** - Reduces computational complexity while preserving important features
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** - Enhances local contrast to make tumor boundaries more visible
3. **Gaussian Noise Reduction** - Removes unwanted noise from MRI scans
4. **Standardization** - Resizes all images to 224x224 pixels and normalizes pixel values

### CNN Architecture
I designed a custom convolutional neural network with:
- 4 convolutional blocks with batch normalization and max pooling
- Progressive filter sizes (32 → 64 → 128 → 256) to capture features at different scales
- Dropout layers (0.5 and 0.3) to prevent overfitting
- Dense layers for final classification with softmax activation

## How to Use This Project

### Prerequisites
Make sure you have Python 3.8+ installed on your system.

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/brain-tumor-detection-cnn.git
cd brain-tumor-detection-cnn

# Install required packages
pip install -r requirements.txt
```

### Getting the Dataset
1. Visit the [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
2. Download and extract the files to the `data/` folder
3. Your folder structure should look like:
   ```
   data/
   ├── Training/
   │   ├── glioma/
   │   ├── meningioma/
   │   ├── notumor/
   │   └── pituitary/
   └── Testing/
       ├── glioma/
       ├── meningioma/
       ├── notumor/
       └── pituitary/
   ```

### Training the Model
```bash
cd src
python3 train.py
```
Training takes about 25-30 minutes on a modern CPU. The script will:
- Load and preprocess all 5,712 training images
- Train the CNN for 30 epochs with early stopping
- Generate accuracy plots and confusion matrix
- Save the trained model as `brain_tumor_cnn.h5`

### Running the Web Application
```bash
streamlit run app.py
```
This launches a user-friendly web interface where you can:
- Upload MRI scan images
- See the preprocessing steps in real-time
- Get predictions with confidence scores
- View results for all four tumor types

## Results and Performance

After extensive testing, the model achieves:
- **Training Accuracy**: 94.8%
- **Validation Accuracy**: 91.2% 
- **Test Accuracy**: 89.7%

The model performs particularly well on:
- No Tumor cases (95.1% accuracy)
- Pituitary tumors (92.3% accuracy)
- Glioma detection (88.9% accuracy)
- Meningioma classification (87.4% accuracy)

## Project Structure

```
brain_tumor_detection/
├── src/
│   ├── data_preprocessing.py    # Image processing functions
│   ├── cnn_model.py            # Neural network architecture
│   └── train.py                # Training and evaluation script
├── models/                     # Saved model files
├── data/                       # MRI dataset (not included in repo)
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation
```

## Technologies Used

- **TensorFlow 2.19** - Deep learning framework
- **OpenCV 4.11** - Image processing operations
- **Streamlit 1.49** - Web application framework
- **Scikit-learn 1.7** - Machine learning utilities
- **NumPy & Matplotlib** - Data manipulation and visualization

## Important Notes

⚠️ **Medical Disclaimer**: This project is intended for educational and research purposes only. It should never be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

The model was trained on a specific dataset and may not generalize to all types of MRI scans or imaging equipment. Real-world medical applications require extensive validation and regulatory approval.

## Future Improvements

I'm planning to enhance this project by:
- Implementing transfer learning with pre-trained models like ResNet or EfficientNet
- Adding data augmentation techniques to improve generalization
- Creating visualization tools to show which parts of the image influenced the prediction
- Expanding the dataset with more diverse MRI scans
- Adding support for different image formats and resolutions

## Contributing

If you find this project helpful or have suggestions for improvements, feel free to:
- Open an issue to report bugs or request features
- Submit pull requests with enhancements
- Share your results if you train the model on different datasets
- Provide feedback on the user interface

## Acknowledgments

- Dataset provided by Masoud Nickparvar on Kaggle
- Inspired by recent advances in medical image analysis
- Built with open-source tools and libraries

---

*This project was developed as part of my machine learning portfolio. If you use this code for research or educational purposes, please consider citing this repository.*