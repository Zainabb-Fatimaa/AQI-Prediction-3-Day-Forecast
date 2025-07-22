import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hopsworks
import openmeteo_requests
import requests_cache
from retry_requests import retry
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

CSV_PATH_LOCAL = "karachi_merged_data_aqi.csv"
CSV_PATH_HOPS = "resources/karachi_merged_data_aqi.csv/karachi_merged_data_aqi.csv"


def fetch_open_meteo_data(lat, lon, start, end):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide",
            "nitrogen_dioxide", "sulphur_dioxide", "ozone"
        ],
        "timezone": "UTC",
        "start_date": str(start),
        "end_date": str(end)
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "pm10": hourly.Variables(0).ValuesAsNumpy(),
        "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
        "carbon_monoxide": hourly.Variables(2).ValuesAsNumpy(),
        "carbon_dioxide": hourly.Variables(3).ValuesAsNumpy(),
        "nitrogen_dioxide": hourly.Variables(4).ValuesAsNumpy(),
        "sulphur_dioxide": hourly.Variables(5).ValuesAsNumpy(),
        "ozone": hourly.Variables(6).ValuesAsNumpy(),
    }
    df = pd.DataFrame(hourly_data)
    df = df.apply(lambda row: standardize_row(row, source="open-meteo"), axis=1)
    # Standardize column names for feature group
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
    return df


def append_to_csv(df, csv_path):
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
            combined.drop_duplicates(subset="date", keep="last", inplace=True)
            combined.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error appending to CSV {csv_path}: {e}")


def compare_and_overwrite(openmeteo_df, fg, threshold=2):
    fg_df = fg.read()
    fg_df["date"] = pd.to_datetime(fg_df["date"], errors='coerce')
    openmeteo_df["date"] = pd.to_datetime(openmeteo_df["date"], errors='coerce')

    merged = pd.merge(
        openmeteo_df,
        fg_df,
        on="date",
        how="left",
        suffixes=("_openmeteo", "_api")
    )

    fields = ["pm2_5", "pm10", "co", "no2", "so2", "o3"]
    result_rows = []
    for _, row in merged.iterrows():
        new_row = row.filter(like="_openmeteo").to_dict()
        new_row = {k.replace("_openmeteo", ""): v for k, v in new_row.items()}
        overwrite = False
        for field in fields:
            openmeteo_val = row.get(f"{field}_openmeteo")
            api_val = row.get(f"{field}_api")
            if pd.notnull(api_val) and pd.notnull(openmeteo_val):
                try:
                    diff = abs(api_val - openmeteo_val)
                except Exception:
                    diff = None
                if diff is not None and diff > threshold:
                    new_row[field] = api_val
                    overwrite = True
        result_rows.append(new_row)
    result_df = pd.DataFrame(result_rows)
    # Only keep required columns
    required_cols = [
        'temperature', 'humidity', 'wind_speed', 'wind_direction',
        'hour', 'day', 'weekday', 'pm2_5', 'pm10',
        'co', 'so2', 'o3', 'no2', 'aqi', 'date'
    ]
    result_df = result_df[[col for col in required_cols if col in result_df.columns]]
    fg.insert(result_df, write_options={"wait_for_job": True})
    # Append/overwrite only the affected day's rows in both CSVs
    for csv_path in [CSV_PATH_LOCAL, CSV_PATH_HOPS]:
        try:
            if os.path.exists(csv_path):
                csv_df = pd.read_csv(csv_path)
                csv_df = csv_df[[col for col in required_cols if col in csv_df.columns]]
                result_dates = pd.to_datetime(result_df["date"], errors='coerce').dt.date.unique()
                csv_df["date"] = pd.to_datetime(csv_df["date"], errors='coerce')
                csv_df = csv_df[~csv_df["date"].dt.date.isin(result_dates)]
                combined = pd.concat([csv_df, result_df], ignore_index=True)
                combined.drop_duplicates(subset="date", keep="last", inplace=True)
                combined.to_csv(csv_path, index=False)
            else:
                result_df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error updating CSV {csv_path}: {e}")


def daily_backfill():
    lat, lon = 24.8608, 67.0104
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    df_hist = fetch_open_meteo_data(lat, lon, start=yesterday, end=today)
    if not df_hist.empty:
        compare_and_overwrite(df_hist, fg, threshold=2)
        print("Open-Meteo backfill and overwrite complete for", yesterday)
    else:
        print("No Open-Meteo data for", yesterday)

if __name__ == "__main__":
    daily_backfill()