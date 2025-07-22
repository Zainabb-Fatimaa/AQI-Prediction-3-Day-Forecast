import os
import pandas as pd
from datetime import datetime
import hopsworks
from src.api_client import APIClient
from src.unit_conversion import standardize_row

# Hopsworks setup
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="karachi_raw_data_store",
    version=1,
    description="Hourly AQI and weather for Karachi, standardized units",
    primary_key=["date"],
    event_time="date",
    online_enabled=True
)

# Initial bulk upload if feature group is empty
def initial_bulk_upload():
    try:
        fg_df = fg.read()
        if fg_df.empty:
            print("Feature group is empty. Uploading historical data...")
            csv_path = "karachi_merged_data_aqi.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Rename columns for Hopsworks compatibility
                df.rename(columns={'pm2.5': 'pm2_5', 'PM2.5': 'pm2_5', 'PM10': 'pm10'}, inplace=True)
                df.columns = [col.lower() for col in df.columns]
                # Only keep required columns
                required_cols = [
                    'temperature', 'humidity', 'wind_speed', 'wind_direction',
                    'hour', 'day', 'weekday', 'pm2_5', 'pm10',
                    'co', 'so2', 'o3', 'no2', 'aqi', 'date'
                ]
                df = df[[col for col in required_cols if col in df.columns]]
                # Convert numeric columns to float64
                numeric_cols = [
                    'temperature', 'humidity', 'wind_speed', 'wind_direction',
                    'hour', 'day', 'weekday', 'pm2_5', 'pm10',
                    'co', 'so2', 'o3', 'no2', 'aqi'
                ]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = df[col].astype('float64')
                # Ensure date is datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                fg.insert(df, write_options={"wait_for_job": True})
                print(f"Uploaded {len(df)} historical rows to feature group.")
            else:
                print("No historical CSV found for initial upload.")
        else:
            print("Feature group already has data.")
    except Exception as e:
        print(f"Error checking/uploading initial data: {e}")


def append_to_csv(df, csv_path):
    # Append new rows to the CSV, creating it if needed
    try:
        # Only keep required columns
        required_cols = [
            'temperature', 'humidity', 'wind_speed', 'wind_direction',
            'hour', 'day', 'weekday', 'pm2_5', 'pm10',
            'co', 'so2', 'o3', 'no2', 'aqi', 'date'
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
    except Exception as e:
        print(f"Error appending to CSV {csv_path}: {e}")


def collect_and_store():
    lat, lon = 24.8608, 67.0104
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    api_client = APIClient()
    data = api_client.get_aqi_data(None, lat, lon)
    if data:
        data['date'] = now
        data_std = standardize_row(data, source="aqicn")  # or "openweather" if that's the source
        df = pd.DataFrame([data_std])
        # Rename columns for Hopsworks compatibility
        df.rename(columns={'pm2.5': 'pm2_5', 'PM2.5': 'pm2_5', 'PM10': 'pm10'}, inplace=True)
        df.columns = [col.lower() for col in df.columns]
        # Only keep required columns
        required_cols = [
            'temperature', 'humidity', 'wind_speed', 'wind_direction',
            'hour', 'day', 'weekday', 'pm2_5', 'pm10',
            'co', 'so2', 'o3', 'no2', 'aqi', 'date'
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
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        fg.insert(df, write_options={"wait_for_job": True})
        print("Inserted hourly data for", now)
        # Append to CSV (local and Hopsworks path)
        local_csv = "karachi_merged_data_aqi.csv"
        hopsworks_csv = "resources/karachi_merged_data_aqi.csv/karachi_merged_data_aqi.csv"
        append_to_csv(df, local_csv)
        append_to_csv(df, hopsworks_csv)
    else:
        print("No data collected.")

if __name__ == "__main__":
    initial_bulk_upload()
    collect_and_store()