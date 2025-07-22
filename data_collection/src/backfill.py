import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from src.api_client import APIClient
from src.feature_engineering import full_feature_engineering_pipeline

PROGRESS_FILE = 'backfill_progress.json'
EXPORT_DIR = 'exports'

logger = logging.getLogger("Backfill")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def save_progress(progress: dict):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {}

def validate_record(record: Dict) -> bool:
    # Basic validation: required fields and value ranges
    required = ['city', 'timestamp', 'aqi']
    for r in required:
        if r not in record or record[r] is None:
            return False
    if not (0 <= record['aqi'] <= 500):
        return False
    return True

def detect_gaps(existing_timestamps: set, all_timestamps: List[str]) -> List[str]:
    # Return missing timestamps as ISO strings
    return [ts for ts in all_timestamps if ts not in existing_timestamps]

def backfill_city(
    city: str,
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    aqicn_key: Optional[str] = None,
    openweather_key: Optional[str] = None,
    airvisual_key: Optional[str] = None,
    output_format: str = 'parquet',
    output_dir: str = 'data/backfill',
    state: Optional[str] = None,
    country: Optional[str] = None
):
    os.makedirs(output_dir, exist_ok=True)
    progress = load_progress()
    city_key = city.lower().replace(' ', '_')
    city_progress = progress.get(city_key, {'last_timestamp': None, 'completed': False})

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    timestamps = [start_dt + timedelta(hours=i) for i in range(int((end_dt - start_dt).total_seconds() // 3600) + 1)]
    timestamps_iso = [dt.isoformat() for dt in timestamps]

    # Resume support: skip already completed timestamps
    completed_ts = set(city_progress.get('completed_timestamps', []))
    to_process = [ts for ts in timestamps_iso if ts not in completed_ts]

    api_client = APIClient()
    records = []
    invalid_records = []
    for ts in to_process:
        print(f"Collecting {city} at {ts} ...")
        dt = datetime.fromisoformat(ts)
        # Use the timestamp as a hint for the API (if supported)
        data = api_client.get_aqi_data(
            city, latitude, longitude, aqicn_key, openweather_key, airvisual_key
        )
        data['timestamp'] = ts
        data['state'] = state or data.get('state')
        data['country'] = country or data.get('country')
        if validate_record(data):
            records.append(data)
            completed_ts.add(ts)
        else:
            print(f"Invalid data for {city} at {ts}: {data}")
            invalid_records.append({'timestamp': ts, 'data': data})
        # Save progress every 10 records
        if len(completed_ts) % 10 == 0:
            progress[city_key] = {
                'last_timestamp': ts,
                'completed_timestamps': list(completed_ts),
                'completed': False
            }
            save_progress(progress)
    # Final progress save
    progress[city_key] = {
        'last_timestamp': to_process[-1] if to_process else city_progress.get('last_timestamp'),
        'completed_timestamps': list(completed_ts),
        'completed': True
    }
    save_progress(progress)

    # Gap detection
    gaps = detect_gaps(completed_ts, timestamps_iso)
    if gaps:
        print(f"Gaps detected for {city}: {gaps}")
        # Optionally, re-query or impute here

    # DataFrame and feature engineering
    df = pd.DataFrame(records)
    if not df.empty:
        df = full_feature_engineering_pipeline(df)
        out_path = os.path.join(output_dir, f"{city_key}_{start[:10]}_{end[:10]}.{output_format}")
        if output_format == 'csv':
            df.to_csv(out_path, index=False)
        else:
            df.to_parquet(out_path, index=False)
        print(f"Saved {len(df)} records to {out_path}")
    else:
        print(f"No valid records collected for {city}.")
    # Save invalid records for review
    if invalid_records:
        with open(os.path.join(output_dir, f"{city_key}_invalid.json"), 'w') as f:
            json.dump(invalid_records, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Historical AQI Data Backfill")
    parser.add_argument('--city', required=True)
    parser.add_argument('--latitude', type=float, required=True)
    parser.add_argument('--longitude', type=float, required=True)
    parser.add_argument('--start', required=True, help='Start datetime (ISO format)')
    parser.add_argument('--end', required=True, help='End datetime (ISO format)')
    parser.add_argument('--aqicn_key')
    parser.add_argument('--openweather_key')
    parser.add_argument('--airvisual_key')
    parser.add_argument('--output_format', choices=['csv', 'parquet'], default='parquet')
    parser.add_argument('--output_dir', default='data/backfill')
    parser.add_argument('--state')
    parser.add_argument('--country')
    args = parser.parse_args()
    backfill_city(
        city=args.city,
        latitude=args.latitude,
        longitude=args.longitude,
        start=args.start,
        end=args.end,
        aqicn_key=args.aqicn_key,
        openweather_key=args.openweather_key,
        airvisual_key=args.airvisual_key,
        output_format=args.output_format,
        output_dir=args.output_dir,
        state=args.state,
        country=args.country
    ) 