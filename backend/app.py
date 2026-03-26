from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import uuid
import librosa

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "AI Voice Spoof Detection Backend Running!"

@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["audio"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename = str(uuid.uuid4()) + "_" + file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        file_size = os.path.getsize(file_path) / (1024 * 1024)
        if file_size > 5:
            os.remove(file_path)
            return jsonify({"error": "File too large (max 5MB)"}), 400

        audio, sr = librosa.load(file_path, sr=16000, mono=True, duration=5)

        # --- DETECTION LOGIC ---
        score = 0

        # 1. Spectral flatness (AI voices are too "perfect")
        flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
        if flatness < 0.001:
            score += 2

        # 2. Zero crossing rate (AI voices have unnatural ZCR)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
        if zcr < 0.02 or zcr > 0.15:
            score += 2

        # 3. MFCC variance (AI voices lack natural variation)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_var = np.mean(np.var(mfcc, axis=1))
        if mfcc_var < 20:
            score += 2

        # 4. Silence ratio (AI voices often have unnatural silence)
        rms = librosa.feature.rms(y=audio)[0]
        silence_ratio = np.sum(rms < 0.01) / len(rms)
        if silence_ratio > 0.4:
            score += 1

        # 5. Pitch consistency (AI voices are too consistent)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_vals = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_vals) > 0:
            pitch_std = np.std(pitch_vals)
            if pitch_std < 50:
                score += 2

        if os.path.exists(file_path):
            os.remove(file_path)

        result = "Spoofed Voice" if score >= 4 else "Genuine Voice"
        return jsonify({"result": result, "score": int(score)})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
