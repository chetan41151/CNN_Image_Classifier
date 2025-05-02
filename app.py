import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# ------------------ Configuration ------------------ #
MODEL_PATH = 'mybatch_students_cnn_classifier.h5'
CLASS_NAMES = ['Anu', 'Bharti', 'Deepak', 'Manidhar', 'Sudh']

# ------------------ Load Model ------------------ #
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="CNN Image Classifier", layout='centered')
st.title("🧠 CNN Image Classifier")
st.markdown("""
Upload an image and the model will predict the class based on a **Vanilla CNN Architecture** trained on your dataset.
""")

# Sidebar Upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])

# ------------------ Prediction Logic ------------------ #
def preprocess_image(img):
    """
    Preprocess the uploaded image for CNN prediction.
    - Resizes to 128x128
    - Normalizes pixel values
    - Expands dimensions to fit model input
    """
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(img):
    """
    Predict the class of the uploaded image.
    Returns:
    - Predicted class name
    - Confidence scores for all classes
    """
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    return predicted_class, predictions[0]

# ------------------ Main App ------------------ #
if uploaded_file is not None:
    image_display = Image.open(uploaded_file).convert('RGB')
    st.image(image_display, caption="Uploaded Image", use_column_width=True)

    # Prediction
    predicted_class, confidences = predict_image(image_display)
    st.success(f"🎯 Predicted Class: **{predicted_class}**")

    # Show class confidence
    st.subheader("🔍 Confidence Scores")
    for class_name, score in zip(CLASS_NAMES, confidences):
        st.write(f"- **{class_name}**: `{score:.4f}`")

