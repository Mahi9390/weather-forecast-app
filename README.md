# 🌤️ Weather Forecast App

A Streamlit-based web application that provides real-time weather forecasting using a trained
LSTM model. The app leverages deep learning to predict temperature trends based on historical
weather data.

---

## 📋 Project Description

This project combines data science, deep learning, and a clean user interface to deliver
accurate short-term weather forecasts. The LSTM (Long Short-Term Memory) neural network 
is trained on global weather data and integrated into a Streamlit app to enable interactive
predictions and visualizations.

---

## 🚀 How to Run the Project

### 1. Clone the Repository

git clone https://github.com/Mahi9390/weather-forecast-app.git
cd weather-forecast-app

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run the Application

streamlit run app/streamlit_app.py

---

## 📁 Project Structure

weather-forecast-app/
├── app/
│   └── streamlit_app.py           # Main Streamlit app
│   └── weather_lstm_model.h5      # Pre-trained LSTM model
├── data/
│   └── GlobalWeatherRepository.csv # Dataset used for training
├── src/
│   ├── data_loader.py             # Handles data loading and preprocessing
│   ├── model.py                   # LSTM model definition and training logic
│   └── utils.py                   # Utility functions for visualization and transformation
├── train_model.py                 # Script to train the model
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation

---

## 🧠 Technologies Used

- Python 3.8+
- Streamlit – For building the interactive web interface
- TensorFlow / Keras – For deep learning and LSTM model
- Pandas, NumPy – For data processing
- Matplotlib, Plotly – For data visualization

---

## 📸 Screenshots

(Optional: Add screenshots of your Streamlit app here)
Example: Upload an image and insert like below

![Weather Forecast UI Screenshot](https://via.placeholder.com/800x400.png?text=App+Screenshot)

---

## 💡 Future Improvements

- Add live weather API integration for real-time updates.
- Improve model accuracy using additional weather parameters.
- Deploy on cloud (Streamlit Cloud / Heroku / AWS).

---

## 🙌 Author

Mahesh Mahi  
GitHub: https://github.com/Mahi9390  
Email: [YourEmail@example.com]

---

## ⭐ Give a Star!

If you like this project, give it a ⭐ on GitHub — it helps!
