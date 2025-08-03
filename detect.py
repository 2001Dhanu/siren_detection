import pyaudio
import numpy as np
import librosa
from collections import deque
import time
import tflite_runtime.interpreter as tflite

# --- Configuration ---
MODEL_PATH = "siren_model_quant.tflite" # Using the optimized TFLite model
STATS_PATH = "norm_stats.npz"

# Audio stream settings
RATE = 22050
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Real-time processing settings
BUFFER_SECONDS = 2.0
HOP_SECONDS = 0.5
BUFFER_SAMPLES = int(BUFFER_SECONDS * RATE)
HOP_SAMPLES = int(HOP_SECONDS * RATE)

# Feature extraction settings
N_MFCC = 40
MAX_LEN = 174

# Prediction smoothing and thresholds
PREDICTIONS_TO_KEEP = 5
CONFIDENCE_THRESHOLD = 0.85
DETECTION_THRESHOLD = 3

# --- Load TFLite Model and Normalization Stats ---
print("🍓 Loading board version model and stats...")
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    norm_stats = np.load(STATS_PATH)
    mean = norm_stats['mean']
    std = norm_stats['std']
except Exception as e:
    print(f"Error loading model or stats: {e}")
    exit()

# --- Feature Extraction for Real-time Audio ---
def extract_realtime_features(audio_buffer, sr, n_mfcc, max_len, mean, std):
    """Extracts features and applies pre-calculated normalization."""
    mfcc = librosa.feature.mfcc(y=audio_buffer, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])
    
    features = (features - mean) / std
    
    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]
        
    # Reshape and ensure correct data type for TFLite model
    return features[np.newaxis, ..., np.newaxis].astype(np.float32)

# --- Main detection loop ---
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=HOP_SAMPLES)

print("\n🎤 Listening... Press Ctrl+C to stop.")

audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)
predictions_history = deque(maxlen=PREDICTIONS_TO_KEEP)

try:
    while True:
        audio_data = stream.read(HOP_SAMPLES, exception_on_overflow=False)
        audio_new = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        audio_buffer = np.roll(audio_buffer, -HOP_SAMPLES)
        audio_buffer[-HOP_SAMPLES:] = audio_new
        
        if np.max(np.abs(audio_buffer)) < 0.01:
            print("🔇 Too quiet, ignoring.", end='\r')
            continue

        # Extract features
        features = extract_realtime_features(audio_buffer, RATE, N_MFCC, MAX_LEN, mean, std)
        
        # --- TFLite Inference ---
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        # ------------------------
        
        # Apply smoothing logic
        is_siren = prediction > CONFIDENCE_THRESHOLD
        predictions_history.append(is_siren)
        
        if sum(predictions_history) >= DETECTION_THRESHOLD:
            print(f"🚨 SIREN DETECTED! (Confidence: {prediction:.2f})    ", flush=True)
        else:
            print(f"✅ No Siren (Confidence: {1-prediction:.2f})          ", end='\r')
            
except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
finally:
    if 'stream' in locals() and stream.is_active():
        stream.stop_stream()
        stream.close()
    if 'p' in locals():
        p.terminate()
    print("Resources released.")