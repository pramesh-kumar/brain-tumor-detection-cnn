import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from data_preprocessing import BrainMRIPreprocessor
from cnn_model import BrainTumorCNN

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('../models/training_history.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('../models/confusion_matrix.png')
    plt.show()

def main():
    # Initialize preprocessor
    preprocessor = BrainMRIPreprocessor()
    
    # Load dataset (you need to download from Kaggle first)
    data_dir = "../data/Training"  # Update this path
    print("Loading dataset...")
    
    try:
        images, labels, class_names = preprocessor.load_dataset(data_dir)
        print(f"Loaded {len(images)} images with {len(class_names)} classes")
        print(f"Classes: {class_names}")
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Initialize and build model
        cnn_model = BrainTumorCNN()
        model = cnn_model.build_model()
        cnn_model.compile_model()
        
        print("\nModel Architecture:")
        model.summary()
        
        # Train model
        print("\nStarting training...")
        history = cnn_model.train_model(X_train, y_train, X_val, y_val, epochs=30)
        
        # Evaluate model
        test_loss, test_accuracy = cnn_model.evaluate_model(X_test, y_test)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Generate predictions for classification report
        X_test_reshaped = X_test.reshape(-1, 224, 224, 1)
        y_pred = np.argmax(cnn_model.model.predict(X_test_reshaped), axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Plot results
        plot_training_history(history)
        plot_confusion_matrix(y_test, y_pred, class_names)
        
        # Save model
        cnn_model.save_model('../models/brain_tumor_cnn.h5')
        print("\nModel saved as 'brain_tumor_cnn.h5'")
        
    except FileNotFoundError:
        print("Dataset not found. Please download the dataset first using download_dataset.py")

if __name__ == "__main__":
    main()