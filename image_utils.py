import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(uploaded_file, target_size=(224, 224)):
    """Reads and preprocesses an uploaded image for model prediction."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # CRITICAL FIX: Convert image from BGR (OpenCV default) to RGB 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_resized = cv2.resize(img_rgb, target_size)
    img_norm = img_resized / 255.0
    img_expanded = np.expand_dims(img_norm, axis=0)
    
    # Return the normalized, expanded array for prediction, 
    # AND the original-sized, color-corrected RGB image for display.
    return img_expanded, img_rgb


def predict_image(model, preprocessed_img):
    """Runs prediction and returns label + confidence."""
    # Note: Model prediction is stable and works correctly
    pred = model.predict(preprocessed_img)[0][0]
    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
    conf = float(pred) if label == "PNEUMONIA" else float(1 - pred)
    return label, conf
