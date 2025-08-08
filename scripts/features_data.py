import os
import json
import pandas as pd
import hopsworks
import sys
from datetime import datetime

def ensure_directories():
    directories = [
        'backend/models',
        'backend/data',
        'backend/config'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_hopsworks_project():
    try:
        project = hopsworks.login(
            api_key_value=os.environ["HOPSWORKS_API_KEY"],
            project=os.environ["HOPSWORKS_PROJECT"]
        )
        return project
    except Exception as e:
        print(f" Failed to login to Hopsworks: {e}")
        sys.exit(1)

def load_selected_features(horizon):
    config_path = f'backend/config/selected_features_{horizon}h.json'
    try:
        with open(config_path, 'r') as f:
            selected_features = json.load(f)
        return selected_features
    except FileNotFoundError:
        print(f" Config file not found: {config_path}")
        return None
    except json.JSONDecodeError:
        print(f" Invalid JSON in config file: {config_path}")
        return None

def extract_horizon_data(horizon, project):
    print(f"--- Processing {horizon}h horizon ---")
    
    try:
        fs = project.get_feature_store()
        
        selected_features = load_selected_features(horizon)
        if not selected_features:
            print(f" No selected features found for {horizon}h. Skipping.")
            return False
        
        print(f"Loading {len(selected_features)} selected features...")
        
        fg_name = f"aqi_features_{horizon}h_prod"
        fg = fs.get_feature_group(name=fg_name, version=1)
        
        print(f"Reading data from feature group: {fg_name}")
        all_data_df = fg.read()
        
        available_features = [feature for feature in selected_features if feature in all_data_df.columns]
        
        if len(available_features) != len(selected_features):
            missing_features = set(selected_features) - set(available_features)
            print(f" Features not found in dataframe: {missing_features}")
        
        if not available_features:
            print(f" No valid features found for {horizon}h horizon")
            return False
        
        print(f"Found {len(available_features)} valid features")
        
        filtered_df = all_data_df[available_features].copy()
        latest_72_rows = filtered_df.tail(72)
        csv_filename = f'backend/data/horizon_{horizon}h_data.csv'
        latest_72_rows.to_csv(csv_filename, index=False)
        
        print(f" Saved latest 72 rows for {horizon}h to {csv_filename}")
        print(f"Data shape: {latest_72_rows.shape}")
        print(f"Features: {list(latest_72_rows.columns)}")
        
        return True
        
    except Exception as e:
        print(f" Failed to process {horizon}h horizon: {e}")
        return False

def main():
    print(f" Starting feature data extraction at {datetime.now()}")
    
    ensure_directories()
    
    print(" Connecting to Hopsworks...")
    project = get_hopsworks_project()
    
    horizons = [24, 48, 72]
    success_count = 0
    
    for horizon in horizons:
        if extract_horizon_data(horizon, project):
            success_count += 1
    
    print(f"\n--- Extraction Summary ---")
    print(f"Successful extractions: {success_count}/{len(horizons)}")
    
    if success_count == 0:
        print(" No data was successfully extracted")
        sys.exit(1)
    elif success_count < len(horizons):
        print(" Some extractions failed, but continuing...")
        sys.exit(0)
    else:
        print(" All data extracted successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()
