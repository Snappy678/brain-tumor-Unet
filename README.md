
# Brain Tumor Classification Streamlit App

This app classifies brain MRI images into four categories using a deep learning model.

## Files Included:
- `app.py` - Main Streamlit application
- `brain_tumor_model.h5` - Trained TensorFlow model
- `class_info.json` - Class names and indices
- `preprocessing_info.pkl` - Preprocessing parameters
- `requirements.txt` - Python dependencies

## How to Run:
1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
3. Open browser to the local URL shown

## Model Information:
- Architecture: Custom CNN
- Input size: 128x128 pixels
- Classes: glioma, meningioma, notumor, pituitary
- Accuracy: >85% on test set
