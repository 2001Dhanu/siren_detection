import pyaudio
import time
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load model
model = load_model("Model_V6_Hybrid_Improved.h5")

# Audio settings
CHUNK = 44100  # Try 2-second chunks (44100 samples @ 22050 Hz)
RATE = 22050
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Feature extraction (must match training)
def extract_features(audio, sr=22050, n_mfcc=40, max_len=174):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])  # Shape: (120, T)
    
    # Normalize (use same scaling as training)
    features = (features - np.mean(features)) / np.std(features)
    
    # Pad/truncate to fixed length
    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]
    
    return features[..., np.newaxis]  # Shape: (120, 174, 1)

# Start stream
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("🎤 Listening... Press Ctrl+C to stop")

try:
    while True:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio_np = audio_np / 32768.0  # Normalize to [-1, 1]

        # Skip silent/quiet frames
        max_volume = np.max(np.abs(audio_np))
        if max_volume < 0.01:  # Adjust based on testing
            print("🔇 Too quiet, ignoring")
            time.sleep(0.1)
            continue

        # Extract features and predict
        features = extract_features(audio_np)
        features = np.expand_dims(features, axis=0)  # Shape: (1, 120, 174, 1)
        prediction = model.predict(features, verbose=0)[0][0]
        
        if prediction > 0.9:  # Higher threshold reduces false positives
            print(f"🚨 SIREN DETECTED! (Confidence: {prediction:.2f})")
        else:
            print(f"✅ No Siren (Confidence: {1 - prediction:.2f})")

        time.sleep(0.5)  # Adjust delay between checks

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
    stream.stop_stream()
    stream.close()
    p.terminate()