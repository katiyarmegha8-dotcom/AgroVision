
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_image
import os

app = Flask(__name__)
CORS(app)

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "AgroVision Backend Running"

@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():

    print("========== REQUEST RECEIVED ==========")

    try:

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        print("Image Saved:", filepath)

        result = predict_image(filepath)

        print("Prediction:", result)

        return jsonify(result)

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500

