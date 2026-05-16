# backend-audio/app.py
# Standalone Flask service for audio deepfake detection
# Loads the LSTM model and performs feature extraction + prediction in one service.

import os
import sys
import gdown
import traceback
import numpy as np
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# Load LSTM Model
# ─────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "deepaudio.h5")

print(f"🚀 Attempting to load audio model from: {MODEL_PATH}")

try:
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Audio LSTM model loaded successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load audio model.")
    print(f"Error details: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────
# Feature Extraction (26 features)
# Exactly matches app1.py extract_features_from_audio()
# Features: chroma_stft, rms, spectral_centroid,
#   spectral_bandwidth, spectral_rolloff,
#   zero_crossing_rate, 20 MFCCs
# ─────────────────────────────────────────────
def extract_features_from_audio(file_stream, sr=22050):
    """
    Extracts 26 audio features from a file stream using librosa.
    Returns a numpy array of shape (26,) or None on failure.
    """
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


# ─────────────────────────────────────────────
# Sliding Window Preparation
# Exactly matches app1.py prepare_data_for_prediction()
# Input: 26 scaled features → Output: shape (1, 21, 5)
# ─────────────────────────────────────────────
def prepare_data_for_prediction(features, window_size=5):
    """
    Creates sliding windows from features for the LSTM model.
    Returns numpy array of shape (1, num_windows, window_size) → (1, 21, 5).
    """
    windows = []
    for j in range(len(features) - window_size):
        window = features[j : j + window_size]
        windows.append(window)

    # Reshape: (1, num_windows, window_size) → (1, 21, 5)
    return np.array(windows)[np.newaxis, ...]


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "audio"})


@app.route("/predict/audio", methods=["POST"])
def predict_audio():
    """
    Accepts a multipart form-data audio file, extracts features,
    applies scaling and sliding windows, runs LSTM prediction,
    and returns { prediction, confidence }.
    """
    try:
        # 1. Validate file upload
        if "file" not in request.files:
            return jsonify({"error": "No audio file provided in the 'file' field"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # 2. Extract 26 librosa features
        features = extract_features_from_audio(file)
        if features is None:
            return jsonify({"error": "Could not process the audio file. Feature extraction failed."}), 500

        print(f"📊 Extracted {len(features)} features")

        # 3. Scale features using MinMaxScaler
        # Note: The original app1.py uses fit_transform on each request
        # (no saved scaler for audio features exists in the models/ directory).
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features.reshape(-1, 1)).flatten()

        # 4. Create sliding windows → shape (1, 21, 5)
        input_data = prepare_data_for_prediction(scaled_features, window_size=5)
        print(f"🔢 Input shape for model: {input_data.shape}")

        # 5. Run LSTM prediction
        prediction_raw = model.predict(input_data)
        confidence_score = float(prediction_raw[0][0])
        prediction_label = "Real" if confidence_score > 0.5 else "Deepfake"

        print(f"🎯 Prediction: {prediction_label} (confidence: {confidence_score:.4f})")

        # 6. Return result
        return jsonify({
            "prediction": prediction_label,
            "confidence": confidence_score
        })

    except Exception as e:
        print(f"❌ Audio prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

def download_models():
    os.makedirs("models", exist_ok=True)
    
    model_path = os.path.join("models", "deepaudio.h5")
    
    if not os.path.exists(model_path):
        print("⬇️ Downloading deepaudio.h5 from Google Drive...")
        gdown.download(
            "https://drive.google.com/uc?id=1JRIS6rygNFcK65sayP0DjHgVq95gLkCb",
            model_path,  # ✅ lowercase, matches the variable above
            quiet=False
        )
        print("✅ Model downloaded successfully")
    else:
        print("✅ Model already exists locally")

download_models()
# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"📢 Starting Veritas Audio service on 0.0.0.0:{port}...")
    app.run(host="0.0.0.0", port=port)
