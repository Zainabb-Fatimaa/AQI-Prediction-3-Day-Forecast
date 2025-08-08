import os
import hopsworks
import pandas as pd
from aqi_preprocessor import AQIDataPreprocessor 

def get_hopsworks_project():
    """Connects to Hopsworks and returns the project object."""
    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ["HOPSWORKS_PROJECT"]
    )
    return project

def create_feature_groups(fs):
    for horizon in [24, 48, 72]:
        fg_name = f"aqi_features_{horizon}h_prod"
        try:
            fs.get_feature_group(name=fg_name, version=1)
            print(f"Feature group '{fg_name}' already exists.")
        except:
            print(f"Creating feature group: {fg_name}")
            fg = fs.get_or_create_feature_group(
                name=fg_name,
                version=1,
                description=f"Preprocessed AQI features for {horizon}h forecast",
                primary_key=["unique_id"],
                event_time="event_time",
                online_enabled=True,
            )

def run_feature_pipeline():
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    create_feature_groups(fs)
    
    fg_old = fs.get_feature_group(name="karachi_raw_data_store", version=1)
    raw_df = fg_old.read()
    
    for horizon in [24, 48, 72]:
        print(f"\n--- Processing for {horizon}h horizon ---")
        
        preprocessor = AQIDataPreprocessor(dataframe=raw_df)
        success = preprocessor.run_full_preprocessing(
            dataframe=raw_df,
            forecast_horizon=horizon
        )
        
        if success:
            processed_data = preprocessor.get_processed_data()
            if processed_data and 'full_data' in processed_data:
                features_df = processed_data['full_data'].copy()
                features_df["event_time"] = pd.to_datetime(features_df.index)
                features_df['unique_id'] = range(len(features_df))
                
                fg = fs.get_feature_group(name=f"aqi_features_{horizon}h_prod", version=1)
                
                schema_cols = [feature.name for feature in fg.schema]
                
                matching_cols = [col for col in features_df.columns if col in schema_cols]
                dropped_cols = [col for col in features_df.columns if col not in schema_cols]
                
                if dropped_cols:
                    print(f"Dropping {len(dropped_cols)} columns not in schema: {dropped_cols[:5]}{'...' if len(dropped_cols) > 5 else ''}")
                
                features_df = features_df[matching_cols]
                print(f"Final DataFrame shape after schema alignment: {features_df.shape}")
                
                fg.insert(features_df, write_options={"wait_for_job": True})
                print(f"Inserted {len(features_df)} rows into '{fg.name}'")
                
if __name__ == "__main__":
    run_feature_pipeline()
