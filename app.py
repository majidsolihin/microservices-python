import os
import time
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import execute_values
from tensorflow.keras.models import load_model
import schedule

# Load model and scaler from the environment paths
MODEL_PATH = os.getenv('MODEL_PATH', 'my_model.h5')
SCALER_PATH = os.getenv('SCALER_PATH', 'scaler.save')
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# Database configuration for Eddy (CO2)
EDDY_CONFIG = {
    'host': os.getenv('PGHOST_EDDY'),
    'port': int(os.getenv('PGPORT_EDDY', 5432)),
    'database': os.getenv('PGDATABASE_EDDY'),
    'user': os.getenv('PGUSER_EDDY'),
    'password': os.getenv('PGPASSWORD_EDDY'),
}

# Helper function to open DB connection
def open_conn(config):
    conn = psycopg2.connect(
        host=config['host'], port=config['port'],
        database=config['database'], user=config['user'],
        password=config['password']
    )
    conn.autocommit = True
    return conn

# Function to fetch the last n_past records for prediction, with the date shifted back by 48 days
def fetch_data_for_prediction(n_past=1440):
    # Get the date 48 days ago
    date_48_days_ago = datetime.utcnow() - timedelta(days=48)
    start_date = date_48_days_ago.strftime("%Y-%m-%d %H:%M:%S")  # Format the date to match the timestamp format
    
    # Fetch data from the database starting from the date 48 days ago
    conn = open_conn(EDDY_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT timestamp, co2, temperature, humidity, rainfall, pyrano
                FROM co2_backend
                WHERE timestamp >= %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (start_date, n_past))
            rows = cur.fetchall()
    finally:
        conn.close()

    df = pd.DataFrame(rows, columns=['timestamp', 'co2', 'temperature', 'humidity', 'rainfall', 'pyrano'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.sort_values('timestamp').ffill().bfill()
    return df

# Prediction function
def predict_and_insert():
    df_hist = fetch_data_for_prediction(n_past=1440)
    
    # Check if enough data is available
    if len(df_hist) < 1440:
        print("Not enough data for prediction.")
        return

    # Scaling the data for prediction
    feature_cols = ['co2', 'temperature', 'humidity', 'rainfall', 'pyrano']
    df_scaled = df_hist.copy()
    df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])

    # Prepare the data for the model
    X = df_scaled[['temperature', 'humidity', 'rainfall', 'pyrano']].values
    X = X.reshape(1, 1440, 4)

    # Make the prediction for the next 60 minutes
    predicted_co2 = model.predict(X)[0][-1]
    predicted_co2 = scaler.inverse_transform([[predicted_co2 if j == 0 else 0 for j in range(5)]])[0][0]

    # Get the timestamp for the prediction
    timestamp_now = datetime.utcnow().replace(second=0, microsecond=0)  # Current time without seconds and microseconds
    timestamp = timestamp_now + timedelta(minutes=60)  # 60 minutes ahead

    # Insert the prediction into the database
    conn = open_conn(EDDY_CONFIG)
    try:
        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO co2_predicted_cp (timestamp, predicted_co2)
                VALUES %s
            """, [(timestamp, predicted_co2)])
    finally:
        conn.close()

    print(f"Prediction inserted for timestamp {timestamp}: CO2 = {predicted_co2} ppm")

# Schedule to run every 60 minutes
schedule.every(60).minutes.do(predict_and_insert)

# Keep running the schedule
while True:
    schedule.run_pending()
    time.sleep(1)
