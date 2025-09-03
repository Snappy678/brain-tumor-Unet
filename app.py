
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="Brain Tumor Classifier",
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
    .positive {
        border-left-color: #28a745;
    }
    .warning {
        border-left-color: #ffc107;
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
</style>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_model():
    # Try to load model from multiple possible files
    model_files = ['final_brain_tumor_model.h5', 'final_brain_tumor_model.keras']
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model = tf.keras.models.load_model(model_file)
                st.sidebar.success(f"✅ Loaded model from {model_file}")
                return model
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {model_file}: {e}")
                continue
    
    st.error("""
    ❌ Could not load model. Please make sure you have one of these files:
    - final_brain_tumor_model.h5
    - final_brain_tumor_model.keras
    """)
    return None

@st.cache_data
def load_metadata():
    try:
        with open('class_info.json', 'r') as f:
            class_info = json.load(f)
        st.sidebar.success("✅ Loaded class information")
    except Exception as e:
        st.error(f"❌ Error loading class_info.json: {e}")
        class_info = None
    
    try:
        with open('preprocessing_info.pkl', 'rb') as f:
            preprocessing_info = pickle.load(f)
        st.sidebar.success("✅ Loaded preprocessing information")
    except Exception as e:
        st.error(f"❌ Error loading preprocessing_info.pkl: {e}")
        preprocessing_info = None
    
    return class_info, preprocessing_info

# Load test results if available
@st.cache_data
def load_test_results():
    try:
        test_results = np.load('test_results.npy', allow_pickle=True).item()
        st.sidebar.success("✅ Loaded test results")
        return test_results
    except:
        st.sidebar.info("ℹ️ Test results not available")
        return None

# Load resources
model = load_model()
class_info, preprocessing_info = load_metadata()
test_results = load_test_results()

if model is None or class_info is None:
    st.error("""
    ❌ Failed to load required files. Please make sure these files are in the same directory:
    - final_brain_tumor_model.h5 or final_brain_tumor_model.keras
    - class_info.json
    - preprocessing_info.pkl
    """)
    st.stop()

class_names = class_info['class_names']
IMG_SIZE = class_info['input_shape'][0]  # Get from class_info

# Enhanced preprocessing function
def preprocess_image(image, img_size=IMG_SIZE):
    """Robust preprocessing that handles various image formats"""
    # Convert to numpy array
    if hasattr(image, 'mode'):
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        elif image.mode == 'L':  # Grayscale
            image = image.convert('RGB')
    
    image_np = np.array(image)
    
    # Handle different color formats
    if len(image_np.shape) == 2:  # Grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # RGBA
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    # Resize and normalize
    image_np = cv2.resize(image_np, (img_size, img_size))
    image_np = image_np.astype('float32') / 255.0
    
    return image_np

# Prediction function
def predict(image):
    processed_image = preprocess_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    
    prediction = model.predict(processed_image, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    return predicted_class, confidence, prediction[0]

# Main app
def main():
    st.sidebar.title("🧠 Navigation")
    app_mode = st.sidebar.radio("Choose a page", 
                               ["Image Classification", "Model Performance", "About"])
    
    if app_mode == "Image Classification":
        render_classification_page()
    elif app_mode == "Model Performance":
        render_performance_page()
    else:
        render_about_page()

def render_classification_page():
    st.markdown('<h1 class="main-header">🧠 Brain Tumor MRI Classification</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Upload a brain MRI image to classify it into one of four categories: 
    <strong>Glioma, Meningioma, No Tumor, or Pituitary</strong>.
    </div>
    """, unsafe_allow_html=True)
    
    # Settings
    st.sidebar.header("⚙️ Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.7, 0.05)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image", 
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner('Analyzing image... This may take a few seconds.'):
                    predicted_class, confidence, all_predictions = predict(image)
                
                with col2:
                    st.header("📊 Results")
                    
                    # Confidence display
                    st.metric("Prediction Confidence", f"{confidence:.2%}")
                    
                    # Main prediction
                    st.subheader("Primary Prediction:")
                    if class_names[predicted_class] == "notumor":
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"## 🟢 NO TUMOR DETECTED")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown(f"## 🟡 {class_names[predicted_class].upper()} DETECTED")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Detailed probabilities
                    st.subheader("Class Probabilities:")
                    
                    # Create dataframe for visualization
                    prob_df = pd.DataFrame({
                        'Class': class_names,
                        'Probability': all_predictions
                    })
                    
                    # Bar chart
                    fig = px.bar(prob_df, x='Class', y='Probability', 
                                title="Prediction Confidence by Class",
                                labels={'Probability': 'Confidence', 'Class': 'Tumor Type'})
                    fig.update_layout(yaxis_tickformat='.0%', showlegend=False)
                    fig.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    st.subheader("💡 Interpretation:")
                    if class_names[predicted_class] == "notumor":
                        st.success("""
                        ✅ **No tumor detected** - The brain MRI appears normal.
                        """)
                    else:
                        st.warning(f"""
                        ⚠️ **Potential {class_names[predicted_class]} detected** - This requires professional medical evaluation.
                        """)
                    
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

def render_performance_page():
    st.markdown('<h1 class="main-header">📊 Model Performance Analysis</h1>', unsafe_allow_html=True)
    
    if test_results is not None:
        # Overall metrics
        st.header("📈 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{test_results['overall_accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{test_results['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{test_results['recall']:.3f}")
        with col4:
            st.metric("AUC", f"{test_results['auc']:.3f}")
        
        # Confusion Matrix
        st.header("🎯 Confusion Matrix")
        cm = np.array(test_results['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
        
    else:
        st.warning("Test results not available. Run comprehensive testing to generate performance metrics.")

def render_about_page():
    st.markdown('<h1 class="main-header">ℹ️ About This Application</h1>', unsafe_allow_html=True)
    
    st.header("🧠 Model Overview")
    st.write("""
    This AI model was trained on the Brain Tumor MRI Dataset from Kaggle to classify 
    brain MRI images into four categories with high accuracy.
    """)
    
    st.header("📋 Class Definitions")
    classes_info = {
        "glioma": "Tumors that occur in the brain and spinal cord",
        "meningioma": "Tumors that arise from the meninges (protective membranes)",
        "notumor": "Healthy brain tissue with no abnormalities",
        "pituitary": "Tumors in the pituitary gland at the base of the brain"
    }
    
    for class_name, description in classes_info.items():
        st.write(f"**{class_name.capitalize()}**: {description}")
    
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
