import tensorflow as tf
import numpy as np
import json
from PIL import Image
import os

# -------------------------
# Load SavedModel
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "crop_disease_saved")

loaded = tf.saved_model.load(MODEL_PATH)
infer = loaded.signatures["serving_default"]

# -------------------------
# Load class names
# -------------------------
CLASS_PATH = os.path.join(BASE_DIR, "..", "models", "class_names.json")

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)


def predict_image(image_path):

    # Read Image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    # Convert to numpy
    image = np.array(image).astype(np.float32)

    # EfficientNet preprocessing
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    # Batch dimension
    image = np.expand_dims(image, axis=0)

    # Tensor
    tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    # Prediction
    output = infer(tensor)

    predictions = list(output.values())[0].numpy()[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "class": class_names[predicted_index],
        "confidence": round(confidence * 100, 2)
    }
    