import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class BrainMRIPreprocessor:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        
    def apply_clahe(self, image):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)
    
    def remove_noise(self, image):
        """Apply Gaussian blur to remove noise"""
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    def preprocess_image(self, image_path):
        """Complete preprocessing pipeline"""
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE
        enhanced = self.apply_clahe(gray)
        
        # Remove noise
        denoised = self.remove_noise(enhanced)
        
        # Resize
        resized = cv2.resize(denoised, self.img_size)
        
        # Normalize
        normalized = resized / 255.0
        
        return normalized
    
    def load_dataset(self, data_dir):
        """Load and preprocess entire dataset"""
        images = []
        labels = []
        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        for idx, class_name in enumerate(class_names):
            class_path = os.path.join(data_dir, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        processed_img = self.preprocess_image(img_path)
                        images.append(processed_img)
                        labels.append(idx)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        return np.array(images), np.array(labels), class_names
    
    def augment_data(self, images, labels):
        """Simple data augmentation"""
        augmented_images = []
        augmented_labels = []
        
        for img, label in zip(images, labels):
            # Original
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # Horizontal flip
            flipped = cv2.flip(img, 1)
            augmented_images.append(flipped)
            augmented_labels.append(label)
            
            # Rotation
            rows, cols = img.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1)
            rotated = cv2.warpAffine(img, M, (cols, rows))
            augmented_images.append(rotated)
            augmented_labels.append(label)
        
        return np.array(augmented_images), np.array(augmented_labels)