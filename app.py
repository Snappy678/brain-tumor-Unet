import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .segmentation-legend {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 10px 0;
        flex-wrap: wrap;
    }
    .legend-item {
        display: flex;
        align-items: center;
        margin: 5px;
    }
    .color-box {
        width: 20px;
        height: 20px;
        margin-right: 8px;
        border: 1px solid #ccc;
    }
</style>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('brain_tumor_unet_model.h5')
        st.sidebar.success("✅ Loaded U-Net segmentation model")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("ℹ️ Running in demo mode with sample segmentation")
        return None

@st.cache_data
def load_metadata():
    try:
        with open('class_info.json', 'r') as f:
            class_info = json.load(f)
        st.sidebar.success("✅ Loaded class information")
        return class_info
    except Exception as e:
        st.error(f"❌ Error loading class_info.json: {e}")
        return None

# Load resources
model = load_model()
class_info = load_metadata()

if class_info is None:
    st.error("❌ Failed to load class information")
    st.stop()

class_names = class_info['class_names']
color_map = class_info['color_map']
IMG_SIZE = class_info['input_shape'][0]

# Helper functions
def onehot_to_colour(mask_onehot):
    """Convert one-hot mask to RGB image"""
    idx = np.argmax(mask_onehot, axis=-1)
    rgb = np.zeros(mask_onehot.shape[:2] + (3,), dtype=np.uint8)
    rgb[idx == 0] = [0, 0, 0]      # Background
    rgb[idx == 1] = [255, 0, 0]    # Glioma - Red
    rgb[idx == 2] = [0, 255, 0]    # Meningioma - Green
    rgb[idx == 3] = [0, 0, 255]    # Pituitary - Blue
    return rgb

def preprocess_image(image, img_size=IMG_SIZE):
    """Preprocess image for model input"""
    if hasattr(image, 'mode'):
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode == 'L':  # Grayscale
            image = image.convert('RGB')

    image_np = np.array(image)

    if len(image_np.shape) == 2:  # Grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    image_np = cv2.resize(image_np, (img_size, img_size))
    image_np = image_np.astype('float32') / 255.0
    return image_np

def create_demo_mask(img_size=128):
    """Create a demo segmentation mask for demonstration"""
    mask = np.zeros((img_size, img_size, 4))
    # Add some demo tumor regions
    mask[30:60, 20:50, 1] = 1  # Glioma - Red
    mask[70:90, 60:90, 2] = 1  # Meningioma - Green
    mask[40:70, 80:110, 3] = 1  # Pituitary - Blue
    return mask

def predict_segmentation(image):
    """Predict segmentation mask - uses real model or demo"""
    if model is not None:
        processed_image = preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)
        prediction = model.predict(processed_image, verbose=0)
        return prediction[0]
    else:
        # Demo mode
        return create_demo_mask()

# Main app
def main():
    st.sidebar.title("🧠 Navigation")
    app_mode = st.sidebar.radio("Choose a page",
                               ["Segmentation", "Model Info", "About"])

    if app_mode == "Segmentation":
        render_segmentation_page()
    elif app_mode == "Model Info":
        render_info_page()
    else:
        render_about_page()

def render_segmentation_page():
    st.markdown('<h1 class="main-header">🧠 Brain Tumor MRI Segmentation</h1>', unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ Running in demo mode - add model files for real predictions")

    st.markdown("""
    <div class="info-box">
    Upload a brain MRI image to segment and identify tumor regions. The model detects:
    <strong>Glioma (Red), Meningioma (Green), Pituitary (Blue)</strong> tumors.
    </div>
    """, unsafe_allow_html=True)

    # Segmentation legend
    st.markdown("""
    <div class="segmentation-legend">
        <div class="legend-item"><div class="color-box" style="background-color: rgb(255,0,0);"></div>Glioma</div>
        <div class="legend-item"><div class="color-box" style="background-color: rgb(0,255,0);"></div>Meningioma</div>
        <div class="legend-item"><div class="color-box" style="background-color: rgb(0,0,255);"></div>Pituitary</div>
        <div class="legend-item"><div class="color-box" style="background-color: rgb(0,0,0);"></div>Background</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload MRI Image")
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI", use_column_width=True)

            if st.button("🔍 Segment Image", type="primary", use_container_width=True):
                with st.spinner('Segmenting image... This may take a few seconds.'):
                    segmentation_mask = predict_segmentation(image)
                    mask_rgb = onehot_to_colour(segmentation_mask)

                with col2:
                    st.header("📊 Segmentation Results")

                    # Display segmentation mask
                    st.subheader("Segmentation Mask")
                    st.image(mask_rgb, caption="Predicted Segmentation", use_column_width=True)

                    # Create overlay
                    original_img = preprocess_image(image)
                    overlay = original_img.copy()
                    tumor_mask = np.any(segmentation_mask[..., 1:] > 0.3, axis=-1)
                    overlay[tumor_mask] = [1, 0.5, 0]  # Orange overlay

                    st.subheader("Tumor Overlay")
                    st.image(overlay, caption="Tumor Regions Highlighted", use_column_width=True)

                    # Medical disclaimer
                    st.markdown("""
                    <div class="info-box">
                    <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational purposes only.
                    Always consult qualified healthcare professionals for medical diagnosis and treatment.
                    </div>
                    """, unsafe_allow_html=True)

    if uploaded_file is None:
        with col2:
            st.info("👈 Please upload a brain MRI image to get started")

def render_info_page():
    st.markdown('<h1 class="main-header">📊 Model Information</h1>', unsafe_allow_html=True)
    
    st.header("🏗️ Model Architecture")
    st.write(f"**Input Shape:** {IMG_SIZE}x{IMG_SIZE}x3")
    st.write(f"**Output Classes:** {len(class_names)}")
    
    if model is not None:
        st.write(f"**Total Parameters:** {model.count_params():,}")
        st.success("✅ Model loaded successfully")
    else:
        st.warning("⚠️ Model file not found - using demo mode")
        st.info("To use the real model, add 'brain_tumor_unet_model.h5' to the app directory")

def render_about_page():
    st.markdown('<h1 class="main-header">ℹ️ About This Application</h1>', unsafe_allow_html=True)

    st.header("🧠 U-Net Segmentation Model")
    st.write("""
    This application uses a U-Net convolutional neural network for semantic segmentation
    of brain MRI images. The model identifies and segments different types of brain tumors.
    """)

    st.header("🎯 Segmentation Classes")
    classes_info = {
        "Background": "Normal brain tissue with no abnormalities",
        "Glioma": "Tumors that occur in the brain and spinal cord (glial cells)",
        "Meningioma": "Tumors that arise from the meninges (protective membranes)",
        "Pituitary": "Tumors in the pituitary gland at the base of the brain"
    }

    for class_name, description in classes_info.items():
        color = color_map.get(class_name, [0,0,0])
        st.markdown(f"""
        <div class="legend-item">
            <div class="color-box" style="background-color: rgb({color[0]},{color[1]},{color[2]});"></div>
            <strong>{class_name}</strong>: {description}
        </div>
        """, unsafe_allow_html=True)

    st.header("⚠️ Important Disclaimer")
    st.warning("""
    **MEDICAL DISCLAIMER**

    This application is intended for **EDUCATIONAL AND RESEARCH PURPOSES ONLY**.

    - ❌ NOT a medical device
    - ❌ NOT for diagnostic use
    - ❌ NOT a replacement for professional medical advice

    Always consult qualified healthcare professionals for medical diagnosis and treatment.
    """)

# Run the app
if __name__ == "__main__":
    main()
