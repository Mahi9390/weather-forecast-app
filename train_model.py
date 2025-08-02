import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Generate dummy weather-like temperature data
np.random.seed(42)
n_samples = 200
temps = np.sin(np.linspace(0, 10, n_samples)) * 10 + 25 + np.random.randn(n_samples)
data = temps.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# Prepare sequences
X, y = [], []
seq_len = 10
for i in range(len(scaled) - seq_len):
    X.append(scaled[i:i+seq_len])
    y.append(scaled[i+seq_len])
X, y = np.array(X), np.array(y)

# Build and train model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])
model.compile(optimizer=Adam(), loss='mse')
model.fit(X, y, epochs=5, batch_size=16, verbose=1)

# Save the model to the "app" folder
model.save("app/weather_lstm_model.h5")
print("✅ Model saved to app/weather_lstm_model.h5")
