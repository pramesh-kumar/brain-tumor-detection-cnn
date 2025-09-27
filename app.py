import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from src.data_preprocessing import BrainMRIPreprocessor
from src.cnn_model import BrainTumorCNN

# Page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('models/brain_tumor_cnn.h5')
        return model
    except:
        return None

def preprocess_uploaded_image(uploaded_file):
    """Preprocess uploaded image"""
    # Convert to PIL Image
    image = Image.open(uploaded_file)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Initialize preprocessor
    preprocessor = BrainMRIPreprocessor()
    
    # Apply CLAHE
    enhanced = preprocessor.apply_clahe(img_array)
    
    # Remove noise
    denoised = preprocessor.remove_noise(enhanced)
    
    # Resize
    resized = cv2.resize(denoised, (224, 224))
    
    # Normalize
    normalized = resized / 255.0
    
    return normalized, resized

def main():
    st.title("🧠 Brain Tumor Detection System")
    st.markdown("Upload an MRI scan to detect brain tumors using Deep Learning")
    
    # About section on top
    with st.expander("📋 About This Project", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 What This System Does
            This AI-powered system analyzes brain MRI scans and classifies them into **4 categories**:
            - **🔴 Glioma** - Serious brain tumor in glial cells
            - **🟡 Meningioma** - Tumors in protective brain membranes
            - **🟢 No Tumor** - Healthy brain scans
            - **🟠 Pituitary Tumor** - Growths in pituitary gland
            
            ### 🧠 How It Works
            1. **Digital Image Processing**: CLAHE enhancement, noise reduction, normalization
            2. **CNN Architecture**: 4 convolutional blocks with 32→64→128→256 filters
            3. **Classification**: Softmax output with confidence scores
            """)
        
        with col2:
            st.markdown("""
            ### 🛠️ Technology Stack
            - **TensorFlow 2.19**
            - **OpenCV 4.11** 
            - **Streamlit 1.49**
            - **Scikit-learn**
            
            ### 📊 Performance
            - **Overall Accuracy**: 89.7%
            - **Training Images**: 5,712
            - **No Tumor**: 95.1%
            - **Pituitary**: 92.3%
            """)
    
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first by running train.py")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an MRI scan image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a brain MRI scan image"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
        
        with col2:
            st.subheader("Preprocessed Image")
            
            # Preprocess image
            processed_img, display_img = preprocess_uploaded_image(uploaded_file)
            st.image(display_img, caption="After DIP Processing", use_column_width=True)
        
        # Make prediction
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing MRI scan..."):
                # Reshape for model
                input_img = processed_img.reshape(1, 224, 224, 1)
                
                # Predict
                prediction = model.predict(input_img, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0]) * 100
                
                # Class names
                class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                predicted_label = class_names[predicted_class]
                
                # Display results
                st.subheader("🎯 Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if predicted_label == 'No Tumor':
                        st.success(f"✅ **{predicted_label}**")
                    else:
                        st.warning(f"⚠️ **{predicted_label} Detected**")
                    
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                with col2:
                    # Probability distribution
                    st.subheader("Class Probabilities")
                    prob_data = {
                        'Class': class_names,
                        'Probability': prediction[0] * 100
                    }
                    
                    for i, (class_name, prob) in enumerate(zip(class_names, prediction[0] * 100)):
                        if i == predicted_class:
                            st.metric(f"**{class_name}**", f"{prob:.1f}%")
                        else:
                            st.metric(class_name, f"{prob:.1f}%")
                
                # Medical disclaimer
                st.warning(
                    "⚠️ **Medical Disclaimer**: This is an AI-based tool for educational purposes only. "
                    "Always consult with qualified medical professionals for proper diagnosis and treatment."
                )

if __name__ == "__main__":
    main()