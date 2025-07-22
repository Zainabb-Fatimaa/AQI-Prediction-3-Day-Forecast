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
    name="karachi_aqi_hourly",
    version=1,
    description="Hourly AQI and weather for Karachi, standardized units",
    primary_key=["city", "timestamp"],
    event_time="timestamp",
    online_enabled=True
)

def collect_and_store():
    city = "Karachi"
    lat, lon = 24.8608, 67.0104
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    api_client = APIClient()
    data = api_client.get_aqi_data(city, lat, lon)
    if data:
        data['city'] = city
        data['timestamp'] = now
        data_std = standardize_row(data, source="aqicn")  # or "openweather" if that's the source
        data_std['source'] = "api_client"
        df = pd.DataFrame([data_std])
        fg.insert(df, write_options={"wait_for_job": True})
        print("Inserted hourly data for", now)
    else:
        print("No data collected.")

if __name__ == "__main__":
    collect_and_store()