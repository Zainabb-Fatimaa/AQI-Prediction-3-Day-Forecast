# -*- coding: utf-8 -*-
"""
Script: data_collect_aqi.py
Description: Production-ready script to collect AQI and weather data, process, and upload to Hopsworks feature store. Designed for local or server execution (not Colab).
"""
import os
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests
import hopsworks
from hsfs.feature import Feature
from hsfs.feature_group import FeatureGroup
from hsfs.client.exceptions import FeatureStoreException
from datetime import datetime, timedelta

# --- Utility Functions ---
def get_hopsworks_api_key():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise EnvironmentError("HOPSWORKS_API_KEY environment variable not set.")
    return api_key

def get_hopsworks_project():
    project = os.getenv("HOPSWORKS_PROJECT")
    if not project:
        raise EnvironmentError("HOPSWORKS_PROJECT environment variable not set.")
    return project

# --- Hopsworks Login ---
api_key = get_hopsworks_api_key()
project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()

# --- Fetch existing feature group data ---
feature_group_name = "karachi_raw_data_store"
feature_group_version = 1
try:
    fg = fs.get_feature_group(name=feature_group_name, version=feature_group_version)
    existing_df = fg.read()
except Exception:
    existing_df = pd.DataFrame()

# --- Open-Meteo API Setup ---
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --- Date Setup ---
DAYS_BACK = 92
END_DATE = (datetime.utcnow() - timedelta(days=1)).date()
START_DATE = END_DATE - timedelta(days=DAYS_BACK - 1)
START_DATE_STR = START_DATE.strftime('%Y-%m-%d')
END_DATE_STR = END_DATE.strftime('%Y-%m-%d')

# --- Air Quality API ---
aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aqi_params = {
    "latitude": 24.8608,
    "longitude": 67.0104,
    "start_date": START_DATE_STR,
    "end_date": END_DATE_STR,
    "hourly": ["pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
    "current": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
    "timezone": "UTC"
}
aqi_responses = openmeteo.weather_api(aqi_url, params=aqi_params)
aqi_response = aqi_responses[0]

# --- Air Quality DataFrame ---
hourly = aqi_response.Hourly()
hourly_data = {"date": pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
)}
hourly_data["pm10"] = hourly.Variables(0).ValuesAsNumpy()
hourly_data["pm2_5"] = hourly.Variables(1).ValuesAsNumpy()
hourly_data["carbon_monoxide"] = hourly.Variables(2).ValuesAsNumpy()
hourly_data["carbon_dioxide"] = hourly.Variables(3).ValuesAsNumpy()
hourly_data["nitrogen_dioxide"] = hourly.Variables(4).ValuesAsNumpy()
hourly_data["sulphur_dioxide"] = hourly.Variables(5).ValuesAsNumpy()
hourly_data["ozone"] = hourly.Variables(6).ValuesAsNumpy()
air_quality_df = pd.DataFrame(data=hourly_data)

# --- Conversion Factors ---
conversion_factors = {
    'carbon_monoxide': {'factor_ppb': 28.01 / 24.45},
    'nitrogen_dioxide': {'factor_ppb': 46.01 / 24.45},
    'sulphur_dioxide': {'factor_ppb': 64.07 / 24.45},
    'ozone': {'factor_ppb': 48.00 / 24.45},
    'carbon_dioxide': {'factor_ppb': 44.01 / 24.45}
}

# --- Rolling Averages and Unit Conversion ---
air_quality_df['PM2.5 (μg/m³) 24h'] = air_quality_df['pm2_5'].rolling(window=24, min_periods=1).mean()
air_quality_df['PM10 (μg/m³) 24h'] = air_quality_df['pm10'].rolling(window=24, min_periods=1).mean()
if 'carbon_monoxide' in air_quality_df.columns:
    air_quality_df['Carbon Monoxide (ppb) Hourly'] = air_quality_df['carbon_monoxide'] / conversion_factors['carbon_monoxide']['factor_ppb']
    air_quality_df['CO (ppm) 8h'] = (air_quality_df['Carbon Monoxide (ppb) Hourly'] / 1000).rolling(window=8, min_periods=1).mean()
if 'nitrogen_dioxide' in air_quality_df.columns:
    air_quality_df['NO2 (ppb) 1h'] = air_quality_df['nitrogen_dioxide'] / conversion_factors['nitrogen_dioxide']['factor_ppb']
if 'sulphur_dioxide' in air_quality_df.columns:
    air_quality_df['Sulphur Dioxide (ppb) Hourly'] = air_quality_df['sulphur_dioxide'] / conversion_factors['sulphur_dioxide']['factor_ppb']
    air_quality_df['SO2 (ppb) 1h'] = air_quality_df['Sulphur Dioxide (ppb) Hourly']
    air_quality_df['SO2 (ppb) 24h'] = air_quality_df['Sulphur Dioxide (ppb) Hourly'].rolling(window=24, min_periods=1).mean()
if 'ozone' in air_quality_df.columns:
    air_quality_df['Ozone (ppb) Hourly'] = air_quality_df['ozone'] / conversion_factors['ozone']['factor_ppb']
    air_quality_df['O3 (ppb) 1h'] = air_quality_df['Ozone (ppb) Hourly']
    air_quality_df['O3 (ppb) 8h'] = air_quality_df['Ozone (ppb) Hourly'].rolling(window=8, min_periods=1).mean()
if 'carbon_dioxide' in air_quality_df.columns:
    air_quality_df['Carbon Dioxide (ppb) Hourly'] = air_quality_df['carbon_dioxide'] / conversion_factors['carbon_dioxide']['factor_ppb']

# --- AQI Calculation Function ---
def calculate_epa_aqi(pollutant, concentration, unit):
    aqi_breakpoints = {
        'PM2.5': [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)],
        'PM10': [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150), (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400), (505, 604, 401, 500)],
        'CO': [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500)],
        'SO2': [(0.0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200), (305, 604, 201, 300), (605, 804, 301, 400), (805, 1004, 401, 500)],
        'O3': [(0.0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200), (106, 200, 201, 300)],
        'NO2': [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 1649, 301, 400), (1650, 2049, 401, 500)]
    }
    if pollutant not in aqi_breakpoints:
        return np.nan
    breakpoints = aqi_breakpoints[pollutant]
    if pollutant in ['O3', 'SO2', 'NO2'] and unit.lower() != 'ppb':
        return np.nan
    elif pollutant == 'CO' and unit.lower() != 'ppm':
        return np.nan
    elif pollutant in ['PM2.5', 'PM10'] and unit.lower() != 'ug/m3':
        return np.nan
    if isinstance(concentration, pd.Series):
        concentration = concentration.values
    aqi_values = []
    if isinstance(concentration, np.ndarray):
        for c in concentration:
            aqi = np.nan
            for c_low, c_high, i_low, i_high in breakpoints:
                if c_low <= c and c <= c_high:
                    aqi = ((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low
                    break
            aqi_values.append(aqi)
        return np.array(aqi_values)
    else:
        aqi = np.nan
        for c_low, c_high, i_low, i_high in breakpoints:
            if c_low <= concentration and concentration <= c_high:
                aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
                break
        return aqi

# --- Calculate AQI Columns ---
air_quality_df['Calculated AQI PM2.5'] = calculate_epa_aqi('PM2.5', air_quality_df['PM2.5 (μg/m³) 24h'], 'ug/m3')
air_quality_df['Calculated AQI PM10'] = calculate_epa_aqi('PM10', air_quality_df['PM10 (μg/m³) 24h'], 'ug/m3')
air_quality_df['Calculated AQI CO'] = calculate_epa_aqi('CO', air_quality_df['CO (ppm) 8h'], 'ppm')
air_quality_df['Calculated AQI SO2'] = calculate_epa_aqi('SO2', air_quality_df['SO2 (ppb) 1h'], 'ppb')
air_quality_df['Calculated AQI O3'] = calculate_epa_aqi('O3', air_quality_df['O3 (ppb) 8h'], 'ppb')
air_quality_df['Calculated AQI NO2'] = calculate_epa_aqi('NO2', air_quality_df['NO2 (ppb) 1h'], 'ppb')
pollutant_aqi_columns = ['Calculated AQI PM2.5', 'Calculated AQI PM10', 'Calculated AQI CO',
                         'Calculated AQI SO2', 'Calculated AQI O3', 'Calculated AQI NO2']
air_quality_df[pollutant_aqi_columns] = air_quality_df[pollutant_aqi_columns].astype(float)
air_quality_df['Calculated Overall AQI'] = air_quality_df[pollutant_aqi_columns].max(axis=1)  # type: ignore

# --- Weather Archive API ---
weather_url = "https://archive-api.open-meteo.com/v1/archive"
weather_params = {
    "latitude": 24.8608,
    "longitude": 67.0104,
    "start_date": START_DATE_STR,
    "end_date": END_DATE_STR,
    "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_10m", "wind_direction_10m", "wind_speed_100m", "wind_direction_100m"],
    "timezone": "UTC"
}
weather_responses = openmeteo.weather_api(weather_url, params=weather_params)
weather_response = weather_responses[0]
hourly = weather_response.Hourly()
hourly_data = {"date": pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
)}
hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
hourly_data["relative_humidity_2m"] = hourly.Variables(1).ValuesAsNumpy()
hourly_data["rain"] = hourly.Variables(2).ValuesAsNumpy()
hourly_data["wind_speed_10m"] = hourly.Variables(3).ValuesAsNumpy()
hourly_data["wind_direction_10m"] = hourly.Variables(4).ValuesAsNumpy()
hourly_data["wind_speed_100m"] = hourly.Variables(5).ValuesAsNumpy()
hourly_data["wind_direction_100m"] = hourly.Variables(6).ValuesAsNumpy()
hourly_df = pd.DataFrame(data=hourly_data)
hourly_df['wind_speed_10m'] = hourly_df['wind_speed_10m'] * 3.6
hourly_df['wind_speed_100m'] = hourly_df['wind_speed_100m'] * 3.6
hourly_df['hour'] = hourly_df['date'].dt.hour
hourly_df['day'] = hourly_df['date'].dt.day
hourly_df['weekday'] = hourly_df['date'].dt.weekday
hourly_df['temperature_change_1h'] = hourly_df['temperature_2m'].diff()
hourly_df['relative_humidity_2m_24h'] = hourly_df['relative_humidity_2m'].rolling(window=24, min_periods=1).mean()

# --- Merge DataFrames ---
merged_df = pd.merge(hourly_df, air_quality_df, on='date', how='inner')
# STEP 1: Rename and filter required columns only (IMPORTANT)
merged_df = merged_df.rename(columns={
    'temperature_2m': 'temperature',
    'relative_humidity_2m': 'humidity',
    'wind_speed_10m': 'wind_speed',
    'wind_direction_10m': 'wind_direction',
    'PM2.5 (μg/m³) 24h': 'pm2_5',
    'PM10 (μg/m³) 24h': 'pm10',
    'Carbon Monoxide (ppb) Hourly': 'co',
    'Sulphur Dioxide (ppb) Hourly': 'so2',
    'Ozone (ppb) Hourly': 'o3',
    'NO2 (ppb) 1h': 'no2',
    'Calculated Overall AQI': 'aqi'
}, errors='ignore')

merged_df = merged_df[[
    'date', 'temperature', 'humidity', 'wind_speed', 'wind_direction',
    'pm2_5', 'pm10', 'co', 'so2', 'o3', 'no2', 'aqi'
]]
# --- Add date_str for online primary key ---
if pd.api.types.is_datetime64_any_dtype(merged_df['date']):
    merged_df['date_str'] = [x.strftime("%Y-%m-%d %H:%M:%S") if hasattr(x, 'strftime') else str(x) for x in merged_df['date']]
else:
    merged_df['date_str'] = [pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S") for x in merged_df['date']]

# --- Fetch existing feature group data and merge ---
try:
    existing_df['date_str'] = existing_df['date_str'].astype(str)
    merged_df['date_str'] = merged_df['date_str'].astype(str)
    final_df = pd.concat([merged_df, existing_df], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['date_str'], keep='first')
except Exception:
    final_df = merged_df.copy()

# --- Save to CSV and upload to Hopsworks Resources ---
final_df.to_csv("karachi_merged_data_aqi.csv", index=False)
dataset_api.upload("karachi_merged_data_aqi.csv", "Resources", overwrite=True)

# --- Convert numeric columns to float64 ---
numeric_cols = final_df.select_dtypes(include='number').columns.tolist()
for col in final_df.columns:
    if col not in ['date', 'date_str']:
        try:
            final_df[col] = final_df[col].astype(float)
        except ValueError:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')


# --- Convert 'date' column to Python date objects ---
final_df['date'] = [x.date() if hasattr(x, 'date') else pd.to_datetime(x).date() for x in final_df['date']]

# --- Define schema ---
feature_group_schema = []
for col in final_df.columns:
    if col == 'date':
        feature_group_schema.append(Feature(name=col, type="date"))
    elif col == 'date_str':
        feature_group_schema.append(Feature(name=col, type="string"))
    elif pd.api.types.is_numeric_dtype(final_df[col]):
        feature_group_schema.append(Feature(name=col, type="double"))
    else:
        feature_group_schema.append(Feature(name=col, type="string"))

# --- Create or update feature group ---
try:
    fg = fs.get_feature_group(name=feature_group_name, version=feature_group_version)
    print("Using existing feature group")
except FeatureStoreException:
    print("Creating new feature group")
    fg = fs.create_feature_group(
        name=feature_group_name,
        version=feature_group_version,
        description="Final features for Karachi AQI model (online + offline) - Overwritten",
        primary_key=["date_str"],
        event_time="date",
        features=feature_group_schema,
        online_enabled=True
    )
except Exception as e:
    print(f"Unexpected error: {e}")
    fg = None

# --- Insert merged data into feature group ---
if fg is not None:
    try:
        fg.insert(final_df, write_options={"wait_for_job": True, "overwrite": True})
        print("Data inserted and overwritten successfully.")
    except Exception as e:
        print(f"Error inserting data into feature group: {e}")
else:
    print("Feature group operation failed. Please check your Hopsworks connection and parameters.")
