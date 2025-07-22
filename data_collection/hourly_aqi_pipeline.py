import os
import pandas as pd
from datetime import datetime
from src.api_client import APIClient
from src.unit_conversion import standardize_row

def append_to_csv(df, csv_path):
    required_cols = [
        'temperature', 'humidity', 'wind_speed', 'wind_direction',
        'hour', 'day', 'weekday', 'pm2_5', 'pm10',
        'co', 'so2', 'o3', 'no2', 'aqi', 'date', 'source'
    ]
    # Standardize column names before saving
    df.rename(columns={
        'PM2.5': 'pm2_5', 'pm2.5': 'pm2_5', 'PM25': 'pm2_5', 'PM10': 'pm10', 'pm10': 'pm10',
        'CO': 'co', 'SO2': 'so2', 'O3': 'o3', 'NO2': 'no2', 'AQI': 'aqi'
    }, inplace=True)
    df.columns = [col.lower() for col in df.columns]
    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    # Always fetch the latest historical data before appending
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing.rename(columns={
            'PM2.5': 'pm2_5', 'pm2.5': 'pm2_5', 'PM25': 'pm2_5', 'PM10': 'pm10', 'pm10': 'pm10',
            'CO': 'co', 'SO2': 'so2', 'O3': 'o3', 'NO2': 'no2', 'AQI': 'aqi'
        }, inplace=True)
        existing.columns = [col.lower() for col in existing.columns]
        # Add source column if missing
        if 'source' not in existing.columns:
            existing['source'] = 'unknown'
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined[[col for col in required_cols if col in combined.columns]]
        combined.drop_duplicates(subset=["date"], keep="last", inplace=True)
        combined.to_csv(csv_path, index=False)
    else:
        # Add source column if missing
        if 'source' not in df.columns:
            df['source'] = 'unknown'
        df = df[[col for col in required_cols if col in df.columns]]
        df.to_csv(csv_path, index=False)

def collect_and_store():
    lat, lon = 24.8608, 67.0104
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    api_client = APIClient()
    data = api_client.get_aqi_data("Karachi", lat, lon)
    if data:
        data['date'] = now
        data_std = standardize_row(data, source="aqicn")  # or "openweather" if that's the source
        data_std['source'] = 'merged'
        df = pd.DataFrame([data_std])
        df.rename(columns={'pm2.5': 'pm2_5', 'PM2.5': 'pm2_5', 'PM25': 'pm2_5', 'PM10': 'pm10', 'pm10': 'pm10',
            'CO': 'co', 'SO2': 'so2', 'O3': 'o3', 'NO2': 'no2', 'AQI': 'aqi'}, inplace=True)
        df.columns = [col.lower() for col in df.columns]
        required_cols = [
            'temperature', 'humidity', 'wind_speed', 'wind_direction',
            'hour', 'day', 'weekday', 'pm2_5', 'pm10',
            'co', 'so2', 'o3', 'no2', 'aqi', 'date', 'source'
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        hopsworks_csv = "Resources/karachi_merged_data_aqi.csv"
        append_to_csv(df, hopsworks_csv)
        print("Inserted hourly data for", now)
    else:
        print("No data collected.")

if __name__ == "__main__":
    collect_and_store()