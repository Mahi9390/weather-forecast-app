import os
from keras.models import load_model

# Get the current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the correct path to the model file inside the 'app' folder
model_path = os.path.join(BASE_DIR, "..", "app", "weather_lstm_model.h5")

# Load the model
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully from:", model_path)
except Exception as e:
    print("❌ Failed to load model.")
    print("Error:", e)
