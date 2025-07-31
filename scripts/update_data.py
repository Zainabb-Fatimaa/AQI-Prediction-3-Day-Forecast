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
        
        # Read and process the downloaded data
        new_df = pd.read_csv(local_temp_path)
        
        # Basic data cleaning for new data
        new_df.ffill(inplace=True)
        new_df.bfill(inplace=True)
        new_df.fillna(new_df.mean(numeric_only=True), inplace=True)
        
        # Define output path
        output_path = "backend/data/aqi_data.csv"
        
        # Check if existing file exists
        if os.path.exists(output_path):
            # File exists - update logic
            print("Existing AQI data file found. Updating...")
            
            try:
                existing_df = pd.read_csv(output_path)
                
                # Combine existing and new data
                # Remove duplicates based on timestamp and city (if these columns exist)
                if 'timestamp' in new_df.columns and 'city' in new_df.columns:
                    # Concatenate and remove duplicates
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['timestamp', 'city'], keep='last')
                    combined_df = combined_df.sort_values('timestamp') if 'timestamp' in combined_df.columns else combined_df
                else:
                    # If no timestamp/city columns, just append new data
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                
                # Save updated data
                combined_df.to_csv(output_path, index=False)
                print(f"Successfully updated AQI data: {len(combined_df)} total records ({len(new_df)} new records)")
                
            except Exception as e:
                print(f"Error reading existing file, replacing with new data: {e}")
                # If error reading existing file, just save new data
                new_df.to_csv(output_path, index=False)
                print(f"Replaced AQI data with new file: {len(new_df)} records")
                
        else:
            # File doesn't exist - create new file
            print("No existing AQI data file found. Creating new file...")
            new_df.to_csv(output_path, index=False)
            print(f"Successfully created new AQI data file: {len(new_df)} records")
        
        # Clean up temp file
        if os.path.exists(local_temp_path):
            os.remove(local_temp_path)
        
        print(f"Data saved to: {output_path}")
        
        # Read final data for summary
        final_df = pd.read_csv(output_path)
        
        # Create data summary
        summary = {
            "last_updated": datetime.now().isoformat(),
            "total_records": len(final_df),
            "cities": final_df['city'].unique().tolist() if 'city' in final_df.columns else [],
            "date_range": {
                "start": final_df['timestamp'].min() if 'timestamp' in final_df.columns else None,
                "end": final_df['timestamp'].max() if 'timestamp' in final_df.columns else None
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
                    
                    # Check if feature file exists
                    if os.path.exists(output_path):
                        print(f"Updating existing features for {horizon}h...")
                        try:
                            existing_features = pd.read_csv(output_path)
                            # Combine and deduplicate if needed
                            combined_features = pd.concat([existing_features, feature_data], ignore_index=True)
                            # Remove duplicates if timestamp column exists
                            if 'timestamp' in combined_features.columns:
                                combined_features = combined_features.drop_duplicates(subset=['timestamp'], keep='last')
                            combined_features.to_csv(output_path, index=False)
                            print(f"Updated features for {horizon}h: {len(combined_features)} total records")
                        except Exception as e:
                            print(f"Error updating existing features, replacing: {e}")
                            feature_data.to_csv(output_path, index=False)
                            print(f"Replaced features for {horizon}h: {len(feature_data)} records")
                    else:
                        print(f"Creating new features file for {horizon}h...")
                        feature_data.to_csv(output_path, index=False)
                        print(f"Created features for {horizon}h: {len(feature_data)} records")
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
