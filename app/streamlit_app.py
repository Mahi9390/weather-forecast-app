import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
import plotly.express as px
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# === Setup paths ===
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "weather_lstm_model.h5")
CSV_PATH = os.path.join(APP_DIR, "..", "data", "GlobalWeatherRepository.csv")

# === Load model ===
try:
    model = load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")

# === Load dataset ===
try:
    data = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    st.error("❌ CSV file not found. Make sure it's in the 'data/' folder.")
    st.stop()

# === App Title ===
st.title("🌦️ Weather Forecast Dashboard")
st.sidebar.header("Explore Weather Data")

# === Sidebar Options ===
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Weather Data")
    st.write(data.head())

if st.sidebar.checkbox("Show Summary Stats"):
    st.subheader("Data Summary")
    st.write(data.describe())

# === Forecast Buttons ===
forecast_type = st.sidebar.radio("Select Forecast Type", ("Mock Forecast", "Real LSTM Forecast"))

if forecast_type == "Mock Forecast":
    if st.sidebar.button("Show Forecast", key="mock_forecast_button"):
        st.subheader("📅 7-Day Forecast (Mock Data)")

        last_temp = data["temperature_celsius"].iloc[-1]
        forecast = [last_temp + np.random.randn()*2 for _ in range(7)]
        dates = pd.date_range(datetime.date.today(), periods=7)

        forecast_df = pd.DataFrame({
            "Date": dates,
            "Predicted Temperature (°C)": forecast
        })

        st.write(forecast_df)
        st.line_chart(forecast_df.set_index("Date"))

elif forecast_type == "Real LSTM Forecast":
    if st.sidebar.button("Show Forecast", key="real_forecast_button"):
        st.subheader("📈 7-Day Forecast (Real LSTM Prediction)")

        try:
            # Prepare sequence for prediction
            seq_len = 10
            scaler = MinMaxScaler()
            scaled_temps = scaler.fit_transform(data["temperature_celsius"].values.reshape(-1, 1))
            input_seq = scaled_temps[-seq_len:].reshape(1, seq_len, 1)

            forecast_scaled = []
            current_input = input_seq

            for _ in range(7):
                next_temp_scaled = model.predict(current_input)[0][0]
                forecast_scaled.append(next_temp_scaled)
                next_input = np.append(current_input[0, 1:], [[next_temp_scaled]], axis=0)
                current_input = next_input.reshape(1, seq_len, 1)

            # Inverse transform to real values
            all_scaled = np.append(scaled_temps.flatten(), forecast_scaled)
            all_real = scaler.inverse_transform(all_scaled.reshape(-1, 1)).flatten()
            forecast_real = all_real[-7:]

            forecast_dates = pd.date_range(datetime.date.today(), periods=7)
            forecast_df = pd.DataFrame({
                "Date": forecast_dates,
                "Predicted Temperature (°C)": forecast_real
            })

            st.write(forecast_df)
            st.line_chart(forecast_df.set_index("Date"))

        except Exception as e:
            st.error("Forecast failed:")
            st.error(e)
