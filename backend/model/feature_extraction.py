import librosa
import numpy as np

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        spectral = librosa.feature.spectral_contrast(y=audio, sr=sr)
        spectral_mean = np.mean(spectral.T, axis=0)

        # ✅ New mic-independent features
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)

        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)

        features = np.hstack([mfcc_mean, chroma_mean, spectral_mean, [zcr_mean, rms_mean]])

        features = np.nan_to_num(features)

        if np.isnan(features).any():
            return None

        return features

    except Exception as e:
        print("Feature extraction error:", e)
        return None  # ✅ Return None instead of zeros
