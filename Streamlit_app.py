
@st.cache_data and @st.cache_resource.

import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU
import joblib
from datetime import datetime
import seaborn as sb
from sklearn.metrics import roc_curve,auc
import warnings
warnings.filterwarnings('ignore')

st.title("Reliance Industries Stock Data Application")

# Load data
data = pd.read_csv('Reliance data.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Display raw data
st.subheader('Raw Data')
st.dataframe(data)

# Display data summary
st.header('Data Summary')
st.write(data.describe())

# Display visualizations when button is clicked
if st.button('Visualizations'):
    st.subheader("Close Price Over Time")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close '])
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    st.pyplot(fig)

    st.subheader("Trading Volume Over Time")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Volume'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sb.heatmap(data.corr(), annot=True, ax=ax)
    st.pyplot(fig)

# Prepare data for GRU model
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Close ', 'Volume']])

# Create sequences for GRU
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predicting 'Close '
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(data_scaled, seq_length)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build GRU model
model = tf.keras.Sequential([
    tf.keras.layers.GRU(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.GRU(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Save the model
model.save('reliance_gru_model.h5')

# Load the model
try:
    model = tf.keras.models.load_model('reliance_gru_model.h5')
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

# Set title for prediction section
st.title("Reliance Industries Stock Data Prediction")

# Get date input for prediction
prediction_date = st.date_input("Enter a date for prediction (2024-2029):")

# Ensure date is within the specified range
start_date = datetime(2024, 1, 1).date()
end_date = datetime(2029, 12, 31).date()

if prediction_date < start_date or prediction_date > end_date:
    st.error("Please select a date between 2024 and 2029.")
else:
    # Prepare data for prediction
    last_sequence = data_scaled[-seq_length:]
    last_sequence = last_sequence.reshape((1, seq_length, 2))  # Reshape for GRU

    # Button to predict closing price
    if st.button('Predict Close Price'):
        predicted_price = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(np.array([[predicted_price[0][0], 0]]))  # Inverse scale
        st.subheader(f"Predicted Close Price for {prediction_date}:")
        st.write(f"{predicted_price[0][0]:.2f}")
        st.balloons()
