import pyaudio
import numpy as np
import librosa
from collections import deque
import time
# import tflite_runtime.interpreter as tflite
import tensorflow.lite as tflite
from RPLCD.i2c import CharLCD

# --- LCD Setup ---
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)

def display_on_lcd(line1, line2=""):
    lcd.clear()
    lcd.write_string(line1)
    if line2:
        lcd.crlf()
        lcd.write_string(line2)

# --- Configuration ---
MODEL_PATH = "/home/chamathsathsara/FYP2/siren_model_quant.tflite"
STATS_PATH = "/home/chamathsathsara/FYP2/norm_stats.npz"

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
    display_on_lcd("Model Load", "Error")
    exit()

# --- Feature Extraction Function ---
def extract_realtime_features(audio_buffer, sr, n_mfcc, max_len, mean, std):
    mfcc = librosa.feature.mfcc(y=audio_buffer, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])
    features = (features - mean) / std

    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]

    return features[np.newaxis, ..., np.newaxis].astype(np.float32)

# --- Main Detection Loop ---
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=HOP_SAMPLES)

print("\n🎤 Listening... Press Ctrl+C to stop.")
display_on_lcd("Listening...", "For Siren")
prev_display = None  # For tracking LCD state

audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)
predictions_history = deque(maxlen=PREDICTIONS_TO_KEEP)

try:
    while True:
        audio_data = stream.read(HOP_SAMPLES, exception_on_overflow=False)
        audio_new = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        audio_buffer = np.roll(audio_buffer, -HOP_SAMPLES)
        audio_buffer[-HOP_SAMPLES:] = audio_new

        # --- Silence Check ---
        if np.max(np.abs(audio_buffer)) < 0.01:
            msg = ("Too Quiet", "Ignoring...")
            if prev_display != msg:
                display_on_lcd(*msg)
                prev_display = msg
            print("🔇 Too quiet, ignoring.".ljust(50), end='\r')
            continue

        # --- Feature Extraction ---
        features = extract_realtime_features(audio_buffer, RATE, N_MFCC, MAX_LEN, mean, std)

        # --- Inference ---
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        is_siren = prediction > CONFIDENCE_THRESHOLD
        predictions_history.append(is_siren)

        # --- Decision Logic ---
        if sum(predictions_history) >= DETECTION_THRESHOLD:
            output_text = f"🚨 SIREN DETECTED! (Confidence: {prediction:.2f})"
            lcd_msg = ("Siren", "Detected!")
        else:
            output_text = f"✅ No Siren (Confidence: {1 - prediction:.2f})"
            lcd_msg = ("No Siren", "")

        print(output_text.ljust(50), end='\r')

        if prev_display != lcd_msg:
            display_on_lcd(*lcd_msg)
            prev_display = lcd_msg

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
    display_on_lcd("Stopped", "By User")

finally:
    if 'stream' in locals() and stream.is_active():
        stream.stop_stream()
        stream.close()
    if 'p' in locals():
        p.terminate()
    print("Resources released.")
