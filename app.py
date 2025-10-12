import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from utils import preprocess_image, predict_image
import numpy as np
import os
import glob

# ============================================================
# Streamlit App Configuration
# ============================================================
st.set_page_config(
    page_title="Pneumonia X-Ray Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Sample Image Directories
# ============================================================
SAMPLE_DIR = "sample_images"
SAMPLE_IMAGES = {}

NORMAL_DIR = os.path.join(SAMPLE_DIR, "NORMAL")
PNEUMONIA_DIR = os.path.join(SAMPLE_DIR, "PNEUMONIA")

# ============================================================
# Function to Load Sample Images Dynamically
# ============================================================
def load_sample_images(directory, prefix):
    """Safely loads all JPG/JPEG/PNG images from a given directory."""
    search_path = os.path.join(directory, "*.*")
    image_paths = [f for f in glob.glob(search_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for path in image_paths:
        file_name = os.path.basename(path)
        display_name = f"{prefix}: {file_name}"
        SAMPLE_IMAGES[display_name] = path


# Load all Normal and Pneumonia samples
if os.path.isdir(NORMAL_DIR):
    load_sample_images(NORMAL_DIR, "Normal")
if os.path.isdir(PNEUMONIA_DIR):
    load_sample_images(PNEUMONIA_DIR, "Pneumonia")

# Warn user if no images were found
if not SAMPLE_IMAGES:
    st.sidebar.warning(f"⚠️ Could not find sample images in '{SAMPLE_DIR}' directory. Check folder structure.")


# ============================================================
# 1️⃣ Load & Stabilize Model
# ============================================================
@st.cache_resource
def load_and_stabilize_model():
    """
    Loads and stabilizes the CNN model for inference.
    Ensures consistent behavior by rebuilding as a Functional model.
    """
    try:
        loaded_model = tf.keras.models.load_model("pneumonia_cnn_model.h5")

        # Recompile for safe inference
        loaded_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Convert Sequential to Functional API for Grad-CAM compatibility
        inputs = tf.keras.Input(shape=(224, 224, 3), name="app_functional_input")
        x = inputs

        # Skip automatically created InputLayer (if present)
        start_index = 1 if loaded_model.layers[0].__class__.__name__ == "InputLayer" else 0

        for layer in loaded_model.layers[start_index:]:
            x = layer(x)

        model = Model(inputs=inputs, outputs=x)
        model.set_weights(loaded_model.get_weights())

        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


# Load the CNN model
model = load_and_stabilize_model()


# ============================================================
# 2️⃣ Streamlit UI & Prediction Logic
# ============================================================
st.title("🩺 AI-Powered Chest X-Ray Analysis")
st.markdown("Upload a chest X-ray image (JPEG/PNG) for Pneumonia detection.")

# --- Sidebar Section ---
st.sidebar.markdown("### 🖼️ Try Sample Images")

sample_options = ["Upload your own image"] + list(SAMPLE_IMAGES.keys())
selected_sample = st.sidebar.selectbox("Choose a sample X-ray to test:", sample_options)

uploaded_file = st.file_uploader(
    "Choose an X-ray image...",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# Input Handling Logic
# ============================================================
input_file = None

if selected_sample != "Upload your own image":
    # Use sample image
    file_path = SAMPLE_IMAGES[selected_sample]
    if os.path.exists(file_path):
        input_file = open(file_path, "rb")
    else:
        st.error(f"❌ Sample image not found at: {file_path}")
        st.stop()

elif uploaded_file is not None:
    # Use uploaded file
    input_file = uploaded_file

# ============================================================
# Prediction Pipeline
# ============================================================
if input_file is not None:
    col1, col2 = st.columns([1, 2])

    # --- Image Display ---
    with col1:
        st.subheader("Original X-Ray")
        input_file.seek(0)
        img_array, original_img_rgb = preprocess_image(input_file)
        st.image(original_img_rgb, caption="Input Image (RGB)", use_column_width=True)

        if selected_sample != "Upload your own image":
            input_file.close()

    # --- Prediction Results ---
    with col2:
        st.subheader("Diagnosis Result")

        # Run model prediction
        label, confidence = predict_image(model, img_array)

        if label == "PNEUMONIA":
            color = "red"
            icon = "⚠️"
            message = "Immediate attention is recommended."
        else:
            color = "green"
            icon = "✅"
            message = "No sign of bacterial or viral pneumonia detected."

        st.markdown(f"### <p style='color:{color};'>{icon} Prediction: {label}</p>", unsafe_allow_html=True)
        st.metric(label="Model Confidence", value=f"{confidence * 100:.2f} %")
        st.markdown(f"<p style='margin-top: 10px;'>{message}</p>", unsafe_allow_html=True)


# ============================================================
# 3️⃣ Sidebar Technical Info
# ============================================================
st.sidebar.info("Upload your own image, or select a sample from the dropdown above.")
st.sidebar.markdown("""
---
**Technical Details**
- **Model:** Custom CNN  
- **Task:** Binary Classification (Normal/Pneumonia)  
- **Image Size:** 224×224  
""")
