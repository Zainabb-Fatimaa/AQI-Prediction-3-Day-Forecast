# scripts/update_data.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hopsworks

def ensure_directories():
    """Create necessary directories"""
    os.makedirs('backend/data', exist_ok=True)

def update_aqi_data():
    """Download and update AQI data from Hopsworks"""
    try:
        # Connect to Hopsworks
        project = hopsworks.login(
            api_key_value=os.environ.get("HOPSWORKS_API_KEY"),
            project=os.environ.get("HOPSWORKS_PROJECT")
        )
        print("Successfully connected to Hopsworks.")

        # Download AQI data
        dataset_api = project.get_dataset_api()
        local_temp_path = "temp_aqi_data.csv"
        
        # Download the data
        dataset_api.download("Resources/aqi_data.csv/aqi_data.csv", local_temp_path, overwrite=True)
        
        # Read and process the data
        df = pd.read_csv(local_temp_path)
        
        # Basic data cleaning
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)
        
        # Save to backend directory
        output_path = "backend/data/aqi_data.csv"
        df.to_csv(output_path, index=False)
        
        # Clean up temp file
        if os.path.exists(local_temp_path):
            os.remove(local_temp_path)
        
        print(f"Successfully updated AQI data: {len(df)} records")
        print(f"Data saved to: {output_path}")
        
        # Create data summary
        summary = {
            "last_updated": datetime.now().isoformat(),
            "total_records": len(df),
            "cities": df['city'].unique().tolist() if 'city' in df.columns else [],
            "date_range": {
                "start": df['timestamp'].min() if 'timestamp' in df.columns else None,
                "end": df['timestamp'].max() if 'timestamp' in df.columns else None
            }
        }
        
        # Save summary
        import json
        with open("backend/data/data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return True
        
    except Exception as e:
        print(f"Error updating AQI data: {e}")
        import traceback
        traceback.print_exc()
        return False

def update_feature_data():
    """Update feature group data for predictions"""
    try:
        # Connect to Hopsworks
        project = hopsworks.login(
            api_key_value=os.environ.get("HOPSWORKS_API_KEY"),
            project=os.environ.get("HOPSWORKS_PROJECT")
        )
        fs = project.get_feature_store()
        
        # Update features for each horizon
        for horizon in [24, 48, 72]:
            try:
                fg_name = f"aqi_features_{horizon}h_prod"
                fg = fs.get_feature_group(name=fg_name)
                
                # Get the latest features
                feature_data = fg.read()
                
                if not feature_data.empty:
                    output_path = f"backend/data/features_{horizon}h.csv"
                    feature_data.to_csv(output_path, index=False)
                    print(f"Updated features for {horizon}h: {len(feature_data)} records")
                else:
                    print(f"No feature data found for {horizon}h")
                    
            except Exception as e:
                print(f"Error updating features for {horizon}h: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error updating feature data: {e}")
        return False

def main():
    ensure_directories()
    
    # Update AQI data (always)
    aqi_success = update_aqi_data()
    
    # Update feature data (for predictions)
    feature_success = update_feature_data()
    
    if aqi_success:
        print("AQI data update completed successfully")
    else:
        print("AQI data update failed")
    
    if feature_success:
        print("Feature data update completed successfully")
    else:
        print("Feature data update failed")

if __name__ == "__main__":
    main()