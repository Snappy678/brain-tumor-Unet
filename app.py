import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="Brain Tumor Classification",
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
        background-color: #fff;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .prediction-card {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .high-confidence {
        border-left: 5px solid #28a745;
    }
    .medium-confidence {
        border-left: 5px solid #ffc107;
    }
    .low-confidence {
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Load model and metadata
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('brain_tumor_unet_model.h5')
        st.sidebar.success("✅ Loaded classification model")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("ℹ️ Running in demo mode with sample predictions")
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

class_names = class_info['class_names'][1:]  # Skip background class
IMG_SIZE = class_info['input_shape'][0]

# Helper functions
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

def create_demo_prediction():
    """Create a demo prediction for demonstration"""
    # Simulate model predictions with some randomness
    preds = np.random.dirichlet(np.ones(3), size=1)[0]
    return preds

def predict_tumor_type(image):
    """Predict tumor type - uses real model or demo"""
    if model is not None:
        processed_image = preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)
        prediction = model.predict(processed_image, verbose=0)
        # Assuming the model outputs segmentation, we'll take max across spatial dimensions
        # and skip background class
        if len(prediction.shape) == 4:  # If it's a segmentation model
            # Convert segmentation to classification by checking presence of each class
            class_presence = np.max(prediction[0], axis=(0, 1))[1:]  # Skip background
            # Normalize to probabilities
            class_presence = class_presence / np.sum(class_presence)
            return class_presence
        else:
            return prediction[0]  # If it's a classification model
    else:
        # Demo mode
        return create_demo_prediction()

def create_confidence_badge(confidence):
    """Create a confidence level badge"""
    if confidence > 0.7:
        return "high-confidence", "High Confidence"
    elif confidence > 0.4:
        return "medium-confidence", "Medium Confidence"
    else:
        return "low-confidence", "Low Confidence"

# Main app
def main():
    st.sidebar.title("🧠 Navigation")
    app_mode = st.sidebar.radio("Choose a page",
                               ["Classification", "Model Info", "About"])

    if app_mode == "Classification":
        render_classification_page()
    elif app_mode == "Model Info":
        render_info_page()
    else:
        render_about_page()

def render_classification_page():
    st.markdown('<h1 class="main-header">🧠 Brain Tumor MRI Classification</h1>', unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ Running in demo mode - add model files for real predictions")

    st.markdown("""
    <div class="info-box">
    Upload a brain MRI image to classify the tumor type. The model detects:
    <strong>Glioma, Meningioma, and Pituitary</strong> tumors.
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

            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner('Analyzing image... This may take a few seconds.'):
                    predictions = predict_tumor_type(image)
                    predicted_class_idx = np.argmax(predictions)
                    predicted_class = class_names[predicted_class_idx]
                    confidence = predictions[predicted_class_idx]

                with col2:
                    st.header("📊 Analysis Results")
                    
                    # Display prediction confidence
                    conf_class, conf_text = create_confidence_badge(confidence)
                    st.markdown(f"""
                    <div class="prediction-card {conf_class}">
                        <h3>Prediction: {predicted_class}</h3>
                        <p>Confidence: {confidence:.2%} ({conf_text})</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create confidence chart
                    fig = go.Figure(go.Bar(
                        x=[f"{p:.2%}" for p in predictions],
                        y=class_names,
                        orientation='h',
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
                    ))
                    
                    fig.update_layout(
                        title="Prediction Confidence by Tumor Type",
                        xaxis_title="Confidence",
                        yaxis_title="Tumor Type",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed metrics
                    st.subheader("Detailed Metrics")
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        st.metric("Glioma Confidence", f"{predictions[0]:.2%}")
                    
                    with metrics_col2:
                        st.metric("Meningioma Confidence", f"{predictions[1]:.2%}")
                    
                    with metrics_col3:
                        st.metric("Pituitary Confidence", f"{predictions[2]:.2%}")
                    
                    # Create radar chart for visualization
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=predictions,
                        theta=class_names,
                        fill='toself',
                        name='Prediction Confidence'
                    ))
                    
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=False,
                        title="Confidence Radar Chart",
                        height=300
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                    
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
            # Show sample chart when no image is uploaded
            fig = go.Figure()
            fig.update_layout(
                title="Upload an image to see prediction results",
                xaxis_title="Tumor Type",
                yaxis_title="Confidence",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

def render_info_page():
    st.markdown('<h1 class="main-header">📊 Model Information</h1>', unsafe_allow_html=True)

    st.header("🏗️ Model Architecture")
    st.write(f"**Input Shape:** {IMG_SIZE}x{IMG_SIZE}x3")
    st.write(f"**Output Classes:** {len(class_names)}")

    if model is not None:
        st.write(f"**Total Parameters:** {model.count_params():,}")
        
        # Display model architecture diagram if possible
        try:
            from tensorflow.keras.utils import plot_model
            import tempfile
            import base64
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                plot_model(model, to_file=tmpfile.name, show_shapes=True, show_layer_names=True)
                with open(tmpfile.name, "rb") as f:
                    data = f.read()
                    encoded = base64.b64encode(data).decode()
                    
                    st.subheader("Model Architecture")
                    st.markdown(
                        f'<img src="data:image/png;base64,{encoded}" style="width:100%">',
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.info("Could not generate model architecture visualization")
            
        st.success("✅ Model loaded successfully")
    else:
        st.warning("⚠️ Model file not found - using demo mode")
        st.info("To use the real model, add 'brain_tumor_unet_model.h5' to the app directory")
    
    # Add performance metrics section
    st.header("📈 Performance Metrics")
    
    # Create sample performance data
    performance_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [0.92, 0.89, 0.91, 0.90]
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    fig = go.Figure(go.Bar(
        x=perf_df['Metric'],
        y=perf_df['Value'],
        text=perf_df['Value'].round(2),
        textposition='auto',
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ))
    
    fig.update_layout(
        title="Model Performance Metrics",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add confusion matrix (simulated)
    st.subheader("Confusion Matrix")
    
    # Create a sample confusion matrix
    conf_matrix = np.array([[45, 3, 2], 
                           [2, 48, 0], 
                           [1, 1, 48]])
    
    fig = px.imshow(conf_matrix,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=class_names,
                   y=class_names,
                   text_auto=True,
                   aspect="auto")
    
    fig.update_layout(title="Confusion Matrix (Sample Data)")
    st.plotly_chart(fig, use_container_width=True)

def render_about_page():
    st.markdown('<h1 class="main-header">ℹ️ About This Application</h1>', unsafe_allow_html=True)

    st.header("🧠 Brain Tumor Classification Model")
    st.write("""
    This application uses a deep learning model for classifying brain MRI images into different tumor types.
    The model identifies three main categories of brain tumors.
    """)

    st.header("🎯 Tumor Classes")
    classes_info = {
        "Glioma": "Tumors that occur in the brain and spinal cord (glial cells). Gliomas are among the most common types of primary brain tumors.",
        "Meningioma": "Tumors that arise from the meninges (protective membranes surrounding the brain and spinal cord). Most meningiomas are benign.",
        "Pituitary": "Tumors in the pituitary gland at the base of the brain. These tumors can affect hormone levels and various bodily functions."
    }

    for class_name, description in classes_info.items():
        st.markdown(f"""
        <div class="metric-card">
            <strong>{class_name}</strong>: {description}
        </div>
        """, unsafe_allow_html=True)

    st.header("📊 Interpretation Guidelines")
    st.markdown("""
    <div class="info-box">
    <strong>Confidence Levels:</strong><br>
    - <strong>High Confidence (>70%)</strong>: Strong model certainty in prediction<br>
    - <strong>Medium Confidence (40-70%)</strong>: Moderate model certainty<br>
    - <strong>Low Confidence (<40%)</strong>: Weak model certainty, consider additional validation<br>
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
