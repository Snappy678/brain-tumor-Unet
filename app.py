
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json
import pickle
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)

# Load model and metadata
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('brain_tumor_model.h5')
    return model

@st.cache_data
def load_metadata():
    with open('class_info.json', 'r') as f:
        class_info = json.load(f)
    with open('preprocessing_info.pkl', 'rb') as f:
        preprocessing_info = pickle.load(f)
    return class_info, preprocessing_info

# Load model and metadata
try:
    model = load_model()
    class_info, preprocessing_info = load_metadata()
    class_names = class_info['class_names']
    IMG_SIZE = preprocessing_info['img_size']
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Preprocess function
def preprocess_image(image, img_size=IMG_SIZE):
    """Preprocess the uploaded image"""
    # Convert to numpy array
    image = np.array(image)
    
    # Convert RGBA to RGB if needed
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize and normalize
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype('float32') / 255.0
    return image

# Prediction function
def predict(image):
    """Make prediction on the image"""
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    
    prediction = model.predict(processed_image, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    return predicted_class, confidence, prediction[0]

# Main app
st.title("🧠 Brain Tumor MRI Classification")
st.markdown("""
This app classifies brain MRI images into one of four categories:
- **Glioma** - A type of tumor that occurs in the brain and spinal cord
- **Meningioma** - A tumor that arises from the meninges
- **No Tumor** - Healthy brain tissue
- **Pituitary** - Tumors in the pituitary gland
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This AI model was trained on the Brain Tumor MRI Dataset from Kaggle.
Upload a brain MRI image to get a classification prediction.
""")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose a brain MRI image", 
    type=['jpg', 'jpeg', 'png']
)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Image")
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        with st.spinner('Analyzing image...'):
            predicted_class, confidence, all_predictions = predict(image)
        
        # Display results
        with col2:
            st.header("Prediction Results")
            
            # Confidence meter
            st.metric("Confidence", f"{confidence:.2%}")
            
            # Progress bars for each class
            st.subheader("Class Probabilities:")
            for i, class_name in enumerate(class_names):
                prob = all_predictions[i]
                st.write(f"**{class_name}**: {prob:.2%}")
                st.progress(float(prob))
            
            # Final prediction
            st.success(f"**Prediction**: {class_names[predicted_class]}")
            
            # Interpretation
            st.subheader("Interpretation:")
            if class_names[predicted_class] == "notumor":
                st.info("✅ No tumor detected. The brain appears healthy.")
            else:
                st.warning(f"⚠️ Potential {class_names[predicted_class]} detected. Please consult with a medical professional for proper diagnosis.")
    
    else:
        st.info("👈 Please upload a brain MRI image using the sidebar")

# Footer
st.markdown("---")
st.caption("Note: This is an AI tool for educational purposes only. Always consult with medical professionals for diagnosis.")
