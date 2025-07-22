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
    df = df[[col for col in required_cols if col in df.columns]]
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[[col for col in required_cols if col in existing.columns]]
        combined = pd.concat([existing, df], ignore_index=True)
        combined.drop_duplicates(subset=["date"], keep="last", inplace=True)
        combined.to_csv(csv_path, index=False)
    else:
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
        df.rename(columns={'pm2.5': 'pm2_5', 'PM2.5': 'pm2_5', 'PM10': 'pm10'}, inplace=True)
        df.columns = [col.lower() for col in df.columns]
        required_cols = [
            'temperature', 'humidity', 'wind_speed', 'wind_direction',
            'hour', 'day', 'weekday', 'pm2_5', 'pm10',
            'co', 'so2', 'o3', 'no2', 'aqi', 'date', 'source'
        ]
        df = df[[col for col in required_cols if col in df.columns]]
        numeric_cols = [
            'temperature', 'humidity', 'wind_speed', 'wind_direction',
            'hour', 'day', 'weekday', 'pm2_5', 'pm10',
            'co', 'so2', 'o3', 'no2', 'aqi'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype('float64')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        local_csv = "karachi_merged_data_aqi.csv"
        hopsworks_csv = "Resources/karachi_merged_data_aqi.csv"
        append_to_csv(df, local_csv)
        append_to_csv(df, hopsworks_csv)
        print("Inserted hourly data for", now)
    else:
        print("No data collected.")

if __name__ == "__main__":
    collect_and_store()