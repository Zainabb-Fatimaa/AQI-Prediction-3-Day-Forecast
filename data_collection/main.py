#!/usr/bin/env python3
"""
Simple AQI Data Collection System for Karachi.
"""

import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.append('src')

from src.data_collector import get_data_collector

# City configuration for Karachi only
CITY = {
    "latitude": 24.8607,
    "longitude": 67.0011,
    "state": "Sindh",
    "country": "Pakistan"
}

def setup_environment():
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

    aqicn_key = os.getenv('AQICN_API_KEY')
    openweather_key = os.getenv('OPENWEATHER_API_KEY')
    airvisual_key = os.getenv('AIRVISUAL_API_KEY')

    return aqicn_key, openweather_key, airvisual_key

def preload_from_hopsworks():
    print("=" * 60)
    print("PRELOAD: Restoring local DB from Hopsworks CSV")
    print("=" * 60)

    import hopsworks
    import pandas as pd
    import sqlite3

    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        print("HOPSWORKS_API_KEY not set. Skipping preload.")
        return

    try:
        project = hopsworks.login(api_key_value=api_key)
        dataset_api = project.get_dataset_api()
        dataset_api.download("Resources/karachi_merged_data_aqi.csv/karachi_merged_data_aqi.csv", "karachi_merged_data_aqi.csv", overwrite=True)

        df = pd.read_csv("karachi_merged_data_aqi.csv")
        os.makedirs("data", exist_ok=True)
        conn = sqlite3.connect("data/aqi_data.db")
        df.to_sql("aqi_data", conn, if_exists="replace", index=False)
        conn.close()

        print(f"✅ Restored {len(df)} rows into local SQLite DB")
    except Exception as e:
        print(f"❌ Preload failed: {e}")

def test_system(aqicn_key=None, openweather_key=None, airvisual_key=None):
    print("=" * 60)
    print("AQI DATA COLLECTION SYSTEM - TEST")
    print("=" * 60)

    collector = get_data_collector()
    test_results = collector.test_api_connections(aqicn_key, openweather_key, airvisual_key)
    print("\nSystem test completed!")
    return test_results

def collect_data(aqicn_key=None, openweather_key=None, airvisual_key=None):
    print("=" * 60)
    print("AQI DATA COLLECTION SYSTEM - DATA COLLECTION")
    print("=" * 60)

    collector = get_data_collector()
    result = collector.collect_data_for_city(
        "karachi",
        CITY["latitude"],
        CITY["longitude"],
        CITY["state"],
        CITY["country"],
        aqicn_key=aqicn_key,
        openweather_key=openweather_key,
        airvisual_key=airvisual_key
    )

    print("\nData collection completed!")

    # Append new records to CSV
    from src.database import get_db_manager
    db_manager = get_db_manager()
    csv_path = "Resources/karachi_merged_data_aqi.csv"
    os.makedirs("Resources", exist_ok=True)
    db_manager.export_all_aqi_data_to_csv(csv_path)

    # Upload to Hopsworks Dataset storage and Feature Store
    import hopsworks
    import pandas as pd
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if api_key:
        try:
            project = hopsworks.login(api_key_value=api_key)
            dataset_api = project.get_dataset_api()
            # Validate file path before upload
            if os.path.isfile(csv_path):
                dataset_api.upload(csv_path, "Resources/karachi_merged_data_aqi.csv", overwrite=True)
                print("✅ Uploaded AQI data CSV to Hopsworks Dataset storage.")
            else:
                print("❌ CSV file not found for upload.")

            # --- Feature group creation and insertion ---
            df = pd.read_csv(csv_path)

            # Rename columns for Hopsworks compatibility
            df.rename(columns={'pm2.5': 'pm2_5', 'PM2.5': 'pm2_5', 'PM10': 'pm10'}, inplace=True)
            df.columns = [col.lower() for col in df.columns]

            # Convert numeric columns to float64
            numeric_cols = [
                'temperature', 'humidity', 'wind_speed', 'wind_direction',
                'hour', 'day', 'weekday', 'pm2_5', 'pm10',
                'co', 'so2', 'o3', 'no2', 'aqi'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].astype('float64')

            # Convert 'date' column to Python date objects
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date

            # Add string version of timestamp for online primary key
            df["date_str"] = pd.to_datetime(df["date"], errors='coerce').astype(str)

            from hsfs.feature import Feature
            feature_group_schema = (
                [Feature(name=col, type="double") for col in df.select_dtypes(include='number').columns if col != "date"] +
                [Feature(name="date", type="date")] +
                [Feature(name="date_str", type="string")]
            )

            fs = project.get_feature_store()
            feature_group_name = "karachi_raw_data_store"
            feature_group_version = 1

            try:
                fg = fs.get_feature_group(name=feature_group_name, version=feature_group_version)
                if fg is None:
                    raise Exception("Feature group not found, will create.")
                print("Using existing feature group")
            except Exception:
                print("Creating new feature group")
                fg = fs.create_feature_group(
                    name=feature_group_name,
                    version=feature_group_version,
                    description="Final features for Karachi AQI model (online + offline)",
                    primary_key=["date_str"],
                    event_time="date",
                    features=feature_group_schema,
                    online_enabled=True
                )

            # Insert data
            if fg is not None:
                print(fg)
                try:
                    fg.insert(df, write_options={"wait_for_job": True})
                    print(fg._feature_group_engine.__class__.__name__)
                except Exception:
                    print("❌ Feature group insert failed.")
            else:
                print("Feature group creation failed. Please check your Hopsworks connection and parameters.")

        except Exception:
            print("❌ Failed to upload to Hopsworks.")
    else:
        print("HOPSWORKS_API_KEY not set. Skipping upload to Hopsworks.")

    return result

def show_latest_data():
    print("=" * 60)
    print("AQI DATA COLLECTION SYSTEM - LATEST DATA")
    print("=" * 60)

    collector = get_data_collector()
    data = collector.get_latest_data_for_city("karachi")

    if data:
        print(f"\nKARACHI:")
        print(f"  AQI: {data.get('aqi', 'N/A')}")
        print(f"  PM2.5: {data.get('pm25', 'N/A')} μg/m³")
        print(f"  PM10: {data.get('pm10', 'N/A')} μg/m³")
        print(f"  Temperature: {data.get('temperature', 'N/A')}°C")
        print(f"  Humidity: {data.get('humidity', 'N/A')}%")
        print(f"  Source: {data.get('source', 'N/A')}")
        print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
    else:
        print(f"\nKARACHI: No data available")
    print("=" * 60)

def show_statistics(days=30):
    print("=" * 60)
    print(f"AQI DATA COLLECTION SYSTEM - STATISTICS (Last {days} days)")
    print("=" * 60)

    collector = get_data_collector()
    stats = collector.get_city_statistics("karachi", days)
    if stats:
        print(f"\nKARACHI:")
        print(f"  Total records: {stats.get('total_records', 0)}")
        print(f"  Average AQI: {stats.get('avg_aqi', 'N/A')}")
        print(f"  Min AQI: {stats.get('min_aqi', 'N/A')}")
        print(f"  Max AQI: {stats.get('max_aqi', 'N/A')}")
        print(f"  Average temperature: {stats.get('avg_temperature', 'N/A')}°C")
        print(f"  Average humidity: {stats.get('avg_humidity', 'N/A')}%")
        print(f"  Average PM2.5: {stats.get('avg_pm25', 'N/A')} μg/m³")
        print(f"  Average PM10: {stats.get('avg_pm10', 'N/A')} μg/m³")
    else:
        print(f"\nKARACHI: No statistics available")
    print("=" * 60)

def cleanup_old_data(days=90):
    print("=" * 60)
    print(f"AQI DATA COLLECTION SYSTEM - CLEANUP (Older than {days} days)")
    print("=" * 60)

    collector = get_data_collector()
    deleted_count = collector.cleanup_old_data(days)
    print(f"Deleted {deleted_count} old records")
    print("=" * 60)

def main():
    print("AQI Data Collection System for Karachi")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    aqicn_key, openweather_key, airvisual_key = setup_environment()

    if not aqicn_key and not openweather_key and not airvisual_key:
        print("Warning: No API keys found. Create a .env file with your API keys:")
        print("AQICN_API_KEY=your_aqicn_key_here")
        print("OPENWEATHER_API_KEY=your_openweather_key_here")
        print("AIRVISUAL_API_KEY=your_airvisual_key_here")
        print()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == 'test':
            test_system(aqicn_key, openweather_key, airvisual_key)
        elif command == 'preload':
            preload_from_hopsworks()
        elif command == 'collect':
            collect_data(aqicn_key, openweather_key, airvisual_key)
        elif command == 'latest':
            show_latest_data()
        elif command == 'stats':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            show_statistics(days)
        elif command == 'cleanup':
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
            cleanup_old_data(days)
        else:
            print(f"Unknown command: {command}")
            print("Available commands: test, preload, collect, latest, stats, cleanup")
    else:
        print("No command specified. Running default sequence...")
        test_system(aqicn_key, openweather_key, airvisual_key)
        preload_from_hopsworks()
        collect_data(aqicn_key, openweather_key, airvisual_key)
        show_latest_data()

    print("Operation completed!")

if __name__ == "__main__":
    main()
