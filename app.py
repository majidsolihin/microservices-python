from flask import Flask, jsonify
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone, timedelta
import psycopg2
from psycopg2.extras import execute_values
from tensorflow.keras.models import load_model
import time
import threading
from dotenv import load_dotenv
import pytz

# Memuat variabel lingkungan dari .env
load_dotenv()

# Inisialisasi Flask app
app = Flask(__name__)

# Memuat model dan scaler dari path yang disediakan dalam environment
MODEL_PATH = os.getenv('MODEL_PATH', 'my_model.h5')
SCALER_PATH = os.getenv('SCALER_PATH', 'scaler.save')
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# Konfigurasi database untuk Eddy (CO2) dari .env
EDDY_CONFIG = {
    'host': os.getenv('PGHOST_EDDY'),
    'port': int(os.getenv('PGPORT_EDDY', 5432)),
    'database': os.getenv('PGDATABASE_EDDY'),
    'user': os.getenv('PGUSER_EDDY'),
    'password': os.getenv('PGPASSWORD_EDDY')
}
CO2_TABLE = "co2_backend"  # Menggunakan tabel `co2_backend` sebagai sumber data

# Fungsi untuk membuka koneksi DB
def open_conn(config):
    conn = psycopg2.connect(
        host=config['host'], port=config['port'],
        database=config['database'], user=config['user'],
        password=config['password']
    )
    conn.autocommit = True
    return conn

# Fungsi untuk mengambil data terakhir sebanyak n_past untuk prediksi (48 hari ke belakang)
def fetch_data_for_prediction(n_past=1440):
    date_48_days_ago = datetime.now(timezone.utc) - timedelta(days=48)
    start_date = date_48_days_ago.strftime("%Y-%m-%d %H:%M:%S")
    
    conn = open_conn(EDDY_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT timestamp, co2, temperature, humidity, rainfall, pyrano
                FROM {CO2_TABLE}
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

# Fungsi untuk melakukan prediksi setiap 30 menit dan memasukkan hasilnya ke dalam database
def predict_and_insert():
    df_hist = fetch_data_for_prediction(n_past=1440)
    
    if len(df_hist) < 1440:
        return {"error": "Data tidak cukup untuk prediksi."}

    # Melakukan scaling pada data
    feature_cols = ['co2', 'temperature', 'humidity', 'rainfall', 'pyrano']
    df_scaled = df_hist.copy()
    df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])

    # Menyiapkan data untuk model
    X = df_scaled[['temperature', 'humidity', 'rainfall', 'pyrano']].values[-1440:]
    X = X.reshape(1, 1440, 4)

    # Melakukan prediksi untuk 30 menit ke depan
    predicted_scaled = model.predict(X)

    # Mengonversi prediksi CO₂ kembali ke skala aslinya
    dummy_full = np.zeros((predicted_scaled.shape[1], len(feature_cols)))
    dummy_full[:, feature_cols.index('co2')] = predicted_scaled[0]

    predicted_original = scaler.inverse_transform(dummy_full)[:, feature_cols.index('co2')]

    # Mendapatkan tanggal dan waktu lokal saat ini, kemudian mengurangi 48 hari
    local_tz = pytz.timezone('Asia/Jakarta')  # Ganti dengan zona waktu sesuai kebutuhan
    current_local_date = datetime.now(local_tz)
    last_timestamp = current_local_date - timedelta(days=48)

    # Membuat future timestamps yang disesuaikan dengan zona waktu lokal
    future_timestamps = [
        last_timestamp + timedelta(minutes=i)
        for i in range(1, len(predicted_original) + 1)
    ]

    # Menambahkan 7 jam pada setiap timestamp untuk menyesuaikan dengan waktu lokal
    future_timestamps = [ts + timedelta(hours=7) for ts in future_timestamps]

    # Memastikan jumlah timestamps dan prediksi memiliki panjang yang sama
    if len(future_timestamps) != len(predicted_original):
        print(f"❌ ERROR: Ketidaksesuaian panjang timestamp dan prediksi ({len(future_timestamps)} vs {len(predicted_original)})")
        return {"error": "Ketidaksesuaian panjang timestamp dan prediksi"}

    # Membuat DataFrame untuk prediksi
    df_prediksi = pd.DataFrame({
        'timestamp': future_timestamps,
        'prediksi_co2': predicted_original
    })

    # Memasukkan hasil prediksi ke dalam database
    conn = open_conn(EDDY_CONFIG)
    try:
        with conn.cursor() as cur:
            execute_values(cur, f"""
                INSERT INTO co2_predicted_cp (timestamp, predicted_co2)
                VALUES %s
            """, list(zip(df_prediksi['timestamp'], df_prediksi['prediksi_co2'])))
    finally:
        conn.close()

    return df_prediksi.to_dict(orient='records')

# Fungsi untuk menjadwalkan prediksi pada menit 00:00 dan 30:00
def schedule_predictions():
    last_executed_time = None  # Melacak waktu eksekusi terakhir untuk mencegah eksekusi duplikat

    while True:
        current_time = datetime.now(timezone.utc)
        if current_time.minute in [0, 30] and current_time.second == 0:
            if last_executed_time and current_time.strftime("%H:%M") == last_executed_time:
                print("[Jadwal] Melewati eksekusi duplikat...")
            else:
                print("[Jadwal] Menjalankan prediksi...")
                predict_and_insert()
                last_executed_time = current_time.strftime("%H:%M")  # Memperbarui waktu eksekusi terakhir
        time.sleep(1)

# Menjalankan penjadwalan prediksi dalam thread terpisah
threading.Thread(target=schedule_predictions, daemon=True).start()

# Endpoint API untuk mendapatkan prediksi secara manual
@app.route('/api/predict', methods=['GET'])
def predict():
    result = predict_and_insert()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
