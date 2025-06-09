import os
import json
import threading
import traceback
from collections import deque
from datetime import datetime, timedelta

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
import paho.mqtt.client as mqtt
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ------ Flask app & error handlers ------
app = Flask(__name__)
CORS(app)

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return jsonify({"error": e.name, "detail": e.description}), e.code

@app.errorhandler(Exception)
def handle_exception(e):
    tb = traceback.format_exc().splitlines()
    return jsonify({
        "error": "Internal Server Error",
        "detail": str(e),
        "trace": tb[-3:]
    }), 500

# ------ Load environment variables ------
load_dotenv()

# Database config
EDDY_CONFIG = {
    'host': os.getenv('PGHOST_EDDY'),
    'port': int(os.getenv('PGPORT_EDDY', 5432)),
    'database': os.getenv('PGDATABASE_EDDY'),
    'user': os.getenv('PGUSER_EDDY'),
    'password': os.getenv('PGPASSWORD_EDDY'),
}
CO2_TABLE = os.getenv('CO2_TABLE', 'station2s')
CLIMATE_CONFIG = {
    'host': os.getenv('PGHOST_CLIMATE'),
    'port': int(os.getenv('PGPORT_CLIMATE', 5432)),
    'database': os.getenv('PGDATABASE_CLIMATE'),
    'user': os.getenv('PGUSER_CLIMATE'),
    'password': os.getenv('PGPASSWORD_CLIMATE'),
}

# MQTT config (optional)
MQTT_BROKER   = os.getenv('MQTT_BROKER', '').strip()
MQTT_PORT     = int(os.getenv('MQTT_PORT', 1883))
MQTT_USER     = os.getenv('MQTT_USER')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD')
MQTT_TOPIC    = os.getenv('MQTT_TOPIC')

# Model & scaler
MODEL_PATH  = os.getenv('MODEL_PATH', 'my_model.h5')
SCALER_PATH = os.getenv('SCALER_PATH', 'scaler.save')

# Load ML model & scaler
model  = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# Helper: open new DB connection
def open_conn(config):
    conn = psycopg2.connect(
        host=config['host'], port=config['port'],
        database=config['database'], user=config['user'],
        password=config['password']
    )
    conn.autocommit = True
    return conn

# Fetch past predictions
def fetch_prediction_history(limit=100):
    conn = open_conn(EDDY_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT timestamp, predicted_co2 FROM co2_predicted_cp "
                "ORDER BY timestamp DESC LIMIT %s", (limit,)
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {"timestamp": r[0].strftime("%Y-%m-%d %H:%M:%S"), "predicted_co2": r[1]}
        for r in rows
    ]

# MQTT realtime buffer
sensor_buffer = deque(maxlen=1440)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        ts = datetime.fromisoformat(payload.get('timestamp'))
    except:
        ts = datetime.utcnow()
    sensor_buffer.append({
        'timestamp': ts,
        'temperature': float(payload.get('temperature', 0)),
        'humidity': float(payload.get('humidity', 0)),
        'rainfall': float(payload.get('rainfall', 0)),
        'pyrano': float(payload.get('pyrano', 0)),
        'co2': float(payload.get('co2', 0)),
    })

if MQTT_BROKER:
    def start_mqtt_thread():
        try:
            client = mqtt.Client()
            client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
            client.on_message = on_message
            client.connect(MQTT_BROKER, MQTT_PORT)
            client.subscribe(MQTT_TOPIC)
            client.loop_forever()
        except Exception as e:
            print(f"[MQTT] Failed to start: {e}")
    threading.Thread(target=start_mqtt_thread, daemon=True).start()
    print(f"[MQTT] Started, broker={MQTT_BROKER}")

# Fetch historical sensor data
def fetch_sensor_data_from_db(n=1440, start_date="2025-04-01 00:00:00"):
    # Env data
    env_conn = open_conn(CLIMATE_CONFIG)
    try:
        with env_conn.cursor() as cur:
            cur.execute(
                "SELECT timestamp, temperature, humidity, rainfall, pyrano "
                "FROM microclimate_kalimantan "
                "WHERE timestamp >= %s ORDER BY timestamp ASC", (start_date,)
            )
            env_rows = cur.fetchall()
    finally:
        env_conn.close()
    # CO2 data
    co2_conn = open_conn(EDDY_CONFIG)
    try:
        with co2_conn.cursor() as cur:
            cur.execute(
                f"SELECT timestamp, co2 FROM {CO2_TABLE} "
                "WHERE timestamp >= %s ORDER BY timestamp ASC", (start_date,)
            )
            co2_rows = cur.fetchall()
    finally:
        co2_conn.close()

    df_env = pd.DataFrame(env_rows, columns=['timestamp','temperature','humidity','rainfall','pyrano'])
    df_co2 = pd.DataFrame(co2_rows, columns=['timestamp','co2'])

    # Normalize timestamps
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'], errors='coerce')
    df_co2['timestamp'] = pd.to_datetime(df_co2['timestamp'], utc=True, errors='coerce')
    df_co2['timestamp'] = df_co2['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

    df_env.sort_values('timestamp', inplace=True)
    df_co2.sort_values('timestamp', inplace=True)

    df_hist = pd.merge_asof(
        df_env, df_co2, on='timestamp',
        direction='nearest', tolerance=pd.Timedelta('5min')
    ).dropna(subset=['co2'])

    if len(df_hist) < n:
        raise RuntimeError(f"DB only has {len(df_hist)} rows matched, need {n}")
    return df_hist.iloc[-n:].reset_index(drop=True)

# =========================== PATCH: Rolling forecast 24 jam per 30 menit ========================
def predict_24h_rolling(df, n_past=1440, step=30, n_future=60):
    """
    Rolling predict 24 jam ke depan setiap 30 menit (total 48 step).
    df: DataFrame dengan ['timestamp', 'co2', 'temperature', 'humidity', 'rainfall', 'pyrano']
    """
    # Urutkan, pastikan bersih
    df = df.sort_values('timestamp').ffill().bfill()
    feature_cols = ['co2', 'temperature', 'humidity', 'rainfall', 'pyrano']
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])
    
    preds = []
    for i in range(48):
        start_idx = -n_past + i*step
        window = df_scaled.iloc[start_idx:start_idx+n_past][['temperature', 'humidity', 'rainfall', 'pyrano']].values
        window = window.reshape(1, n_past, 4)
        pred_scaled = model.predict(window)[0][-1]  # Prediksi menit ke-60 (akhir)
        # Inverse hanya kolom 'co2'
        inv = scaler.inverse_transform([[pred_scaled if j==0 else 0 for j in range(5)]])[0][0]
        preds.append(inv)
    # Timestamp 30 menit ke depan dari last ts
    last_timestamp = df['timestamp'].iloc[-1]
    timestamps = [last_timestamp + pd.Timedelta(minutes=30*(i+1)) for i in range(48)]
    return pd.DataFrame({'timestamp': timestamps, 'predicted_co2': preds})

# =========================== END PATCH ========================

# Prediction function with logging (original batch prediction, not rolling by 30min)
def predict_1440_minutes(df, n_past=1440, n_future=60, total_minutes=1440):
    df = df.sort_values('timestamp').ffill().bfill()
    cols = ['co2','temperature','humidity','rainfall','pyrano']
    df_scaled = df.copy()
    df_scaled[cols] = scaler.transform(df_scaled[cols])
    buf = df_scaled.iloc[-n_past:].copy()
    preds = []
    steps = total_minutes // n_future
    print(f"[Predict] {steps} batches of {n_future} min")
    for i in range(steps):
        x = np.expand_dims(buf[['temperature','humidity','rainfall','pyrano']].values,0)
        p = model.predict(x)[0]
        batch = np.zeros((n_future,5)); batch[:,0]=p
        co2_batch = scaler.inverse_transform(batch)[:,0]
        preds.extend(co2_batch)
        pct = (i+1)/steps*100
        print(f"[Predict] Batch {i+1}/{steps} ({pct:.1f}%)")
        last = buf.iloc[-1][['temperature','humidity','rainfall','pyrano']].values
        future = np.tile(last, (n_future,1))
        tmp = pd.DataFrame(future, columns=['temperature','humidity','rainfall','pyrano'])
        tmp['co2']=0
        env_scaled = scaler.transform(tmp)[:,1:]
        new = np.column_stack((co2_batch.reshape(-1,1), env_scaled))
        buf = pd.concat([buf, pd.DataFrame(new,columns=cols)], ignore_index=True).iloc[-n_past:]
    print("[Predict] Done")
    return preds

# =========================== API ENDPOINTS ========================

@app.route('/api/carbon-predict/forecast_24h')
def forecast_24h():
    """
    Endpoint prediksi rolling 24 jam ke depan, setiap 30 menit (48 data)
    """
    try:
        now = datetime.now()
        # Force bulan & tahun ke April 2025
        query_now = now.replace(year=2025, month=4)
        # Rolling window 1440 menit ke belakang dari waktu 'sekarang' versi April 2025
        start_date = (query_now - timedelta(minutes=1440)).strftime("%Y-%m-%d %H:%M:%S")
        df = fetch_sensor_data_from_db(n=1440, start_date=start_date)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # Pastikan kolom co2, temperature, humidity, rainfall, pyrano ada
    for col in ['co2', 'temperature', 'humidity', 'rainfall', 'pyrano']:
        if col not in df.columns:
            return jsonify({"error": f"Column '{col}' not in data"}), 400
    # Rolling predict
    out = predict_24h_rolling(df)
    # Ubah ke JSON (isoformat)
    result = [
        {"timestamp": ts.isoformat(sep=' '), "predicted_co2": float(co2)}
        for ts, co2 in zip(out['timestamp'], out['predicted_co2'])
    ]
    return jsonify(result)

@app.route('/api/predict/next1440')
def get_next1440():
    src = request.args.get('source','mqtt').lower()
    minutes = min(int(request.args.get('minutes',1440)), 1440)
    if src == 'db':
        now = datetime.now()
        query_now = now.replace(year=2025, month=4)
        start_date = (query_now - timedelta(minutes=1440)).strftime("%Y-%m-%d %H:%M:%S")
        df_hist = fetch_sensor_data_from_db(1440, start_date=start_date)
    else:
        df_hist = pd.DataFrame(list(sensor_buffer))
        df_hist['timestamp']=pd.to_datetime(df_hist['timestamp'],errors='coerce')
        df_hist.sort_values('timestamp',inplace=True)
        df_hist.reset_index(drop=True,inplace=True)
    if len(df_hist)<1440:
        return jsonify(error=f"{src} only {len(df_hist)} rows"),400
    diffs = df_hist['timestamp'].diff().dropna()
    delta = diffs.mode()[0] if not diffs.empty else timedelta(minutes=1)
    preds = predict_1440_minutes(df_hist, total_minutes=minutes)
    last_ts = df_hist['timestamp'].iloc[-1]
    rec, result = [], []
    for idx, v in enumerate(preds, start=1):
        t = last_ts + delta*idx
        ts = t.strftime("%Y-%m-%d %H:%M:%S")
        result.append({"timestamp":ts,"predicted_co2":float(v)})
        rec.append((ts, float(v)))
    conn = open_conn(EDDY_CONFIG)
    try:
        with conn.cursor() as cur:
            execute_values(cur,
                "INSERT INTO co2_predicted_cp(timestamp,predicted_co2) VALUES %s",
                rec, template="(%s,%s)")
    finally:
        conn.close()
    print(f"[DB] Inserted {len(rec)} records")
    return jsonify(result)

@app.route('/api/predict/history')
def get_history():
    limit = int(request.args.get('limit',100))
    return jsonify(fetch_prediction_history(limit))

if __name__=='__main__':
    print("Flask running on 0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001)
