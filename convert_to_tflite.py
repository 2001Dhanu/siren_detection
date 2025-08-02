import tensorflow as tf

# Load the model
model = tf.keras.models.load_model(r"E:\Campus\Semester\FYP\siren_detection_project\src\Model_V6_Hybrid_Improved.h5")

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable Select TensorFlow Ops (to support LSTM and other complex ops)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,     # Default ops
    tf.lite.OpsSet.SELECT_TF_OPS        # Add support for TensorFlow ops
]

# Disable lowering tensor list ops (required for LSTM-based models)
converter._experimental_lower_tensor_list_ops = False

# Optional: apply quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open("model_v6_hybrid.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model converted and saved.")