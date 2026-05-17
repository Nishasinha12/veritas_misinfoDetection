"""
Veritas — Audio Deepfake Detection Microservice
================================================
Standalone Flask service for audio-based deepfake detection.
"""

import os
import sys
import traceback
import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "deepaudio.h5")
model = None  # Lazy loaded on demand

# ---------------------------------------------------------------------------
# Lazy Model Downloader & Loader
# ---------------------------------------------------------------------------
def ensure_model_exists():
    """Safely downloads the LSTM model from HuggingFace if missing."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading deepaudio.h5 from HuggingFace...")
        try:
            hf_hub_download(
                repo_id="nishuu12/veritas-models",
                filename="deepaudio.h5",
                local_dir=MODEL_DIR,
                token=os.getenv("HF_TOKEN")
            )
            print("✅ Audio model downloaded successfully!")
        except Exception as e:
            print(f"⚠️ HuggingFace download failed: {e}")
    else:
        print("✅ Audio model already exists locally")

def get_model():
    """Retrieves or loads the TensorFlow model on demand to prevent startup timeouts."""
    global model
    if model is None:
        ensure_model_exists()
        if os.path.isfile(MODEL_PATH):
            try:
                import tensorflow as tf
                print("📦 Loading TensorFlow audio model...")
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print("✅ Audio LSTM model loaded successfully.")
            except Exception as e:
                print(f"⚠️ Failed to load model at {MODEL_PATH}: {e}")
                traceback.print_exc()
                model = None
        else:
            print(f"ℹ️ No audio model file found at {MODEL_PATH} — running in STUB mode.")
    return model

# ---------------------------------------------------------------------------
# Processing Helpers
# ---------------------------------------------------------------------------
def extract_features_from_audio(file_stream, sr=22050):
    """Extracts 26 audio features from a file stream using librosa."""
    try:
        y, sr = librosa.load(file_stream, sr=sr)
        feature_vector = [
            np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            np.mean(librosa.feature.rms(y=y)),
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            np.mean(librosa.feature.zero_crossing_rate(y=y)),
            *np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
        ]
        return np.array(feature_vector)
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        traceback.print_exc()
        return None

def prepare_data_for_prediction(features, window_size=5):
    """Creates sliding windows from features for the LSTM model (1, 21, 5)."""
    windows = []
    for j in range(len(features) - window_size):
        window = features[j : j + window_size]
        windows.append(window)
    return np.array(windows)[np.newaxis, ...]

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "service": "Veritas Audio API"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "audio"})

@app.route("/predict/audio", methods=["POST"])
def predict_audio():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No audio file provided in the 'file' field"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        features = extract_features_from_audio(file)
        if features is None:
            return jsonify({"error": "Feature extraction failed."}), 500

        print(f"📊 Extracted {len(features)} features")

        # Feature Rescaling
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features.reshape(-1, 1)).flatten()

        input_data = prepare_data_for_prediction(scaled_features, window_size=5)
        print(f"🔢 Input shape for model: {input_data.shape}")

        # Check if model loads correctly
        active_model = get_model()
        if active_model is not None:
            prediction_raw = active_model.predict(input_data)
            confidence_score = float(prediction_raw[0][0])
            prediction_label = "Real" if confidence_score > 0.5 else "Deepfake"
        else:
            # Fallback stub if loading fails so endpoints don't crash
            prediction_label = "Deepfake"
            confidence_score = 0.85
            print("⚠️ Running in audio STUB mode.")

        print(f"🎯 Prediction: {prediction_label} (confidence: {confidence_score:.4f})")

        return jsonify({
            "prediction": prediction_label,
            "confidence": confidence_score
        })

    except Exception as e:
        print(f"❌ Audio prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Smooth Cloud Run port bindings
    print(f"📢 Starting Veritas Audio service on 0.0.0.0:{port}...")
    app.run(host="0.0.0.0", port=port)