# scripts/download_models.py
import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import hopsworks
import glob

def ensure_directories():
    """Create necessary directories"""
    directories = [
        'backend/models',
        'backend/data',
        'backend/config'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def cleanup_old_models(horizon):
    """Remove all existing model files for a specific horizon before saving the new one."""
    model_patterns = [
        f"backend/models/model_{horizon}h.*",
        f"backend/models/*_{horizon}h.*"
    ]
    
    removed_files = []
    for pattern in model_patterns:
        old_files = glob.glob(pattern)
        for old_file in old_files:
            try:
                os.remove(old_file)
                removed_files.append(old_file)
                print(f"Removed old model file: {old_file}")
            except Exception as e:
                print(f"Warning: Could not remove {old_file}: {e}")
    
    if removed_files:
        print(f"Cleaned up {len(removed_files)} old model files for {horizon}h")
    
    return removed_files

def get_latest_model_version(mr, model_name):
    """Get the latest version of a model, preferring production-tagged versions."""
    try:
        # First try to get production version
        try:
            model_info = mr.get_model(name=model_name, tag="production")
            print(f"Using production-tagged version {model_info.version} for {model_name}")
            return model_info
        except:
            pass

        # If no production tag, try "latest" tag
        try:
            model_info = mr.get_model(name=model_name, tag="latest")
            print(f"Using latest-tagged version {model_info.version} for {model_name}")
            return model_info
        except:
            pass

        # If no tags, get all versions and pick the highest
        models_list = mr.get_models(name=model_name)
        if not models_list:
            raise Exception(f"No models found with name {model_name}")

        # Sort by version number (descending) and get the first one
        latest_model = max(models_list, key=lambda x: x.version)
        print(f"Using highest version {latest_model.version} for {model_name}")
        return latest_model

    except Exception as e:
        print(f"Error getting latest model version for {model_name}: {e}")
        return None

def parse_best_model_from_description(description):
    """Parse the best model name from description."""
    if not description:
        return None

    # Updated patterns to handle various model names
    patterns = [
        r"The best model for \d+h is ([A-Za-z]+)",
        r"Best model: ([A-Za-z]+)",
        r"Model: ([A-Za-z]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            model_name = match.group(1).lower()
            
            # Normalize model names to standard format
            model_mapping = {
                'extratrees': 'ExtraTrees',
                'extra_trees': 'ExtraTrees',
                'extratreesregressor': 'ExtraTrees',
                'catboost': 'CatBoost',
                'catboostregressor': 'CatBoost',
                'xgboost': 'XGBoost',
                'xgbregressor': 'XGBoost',
                'lightgbm': 'LightGBM',
                'lgbmregressor': 'LightGBM',
                'gradientboostingregressor': 'GradientBoosting',
                'gradientboosting': 'GradientBoosting',
                'randomforest': 'RandomForest',
                'randomforestregressor': 'RandomForest',
                'decisiontree': 'DecisionTree',
                'decisiontreeregressor': 'DecisionTree'
            }
            
            return model_mapping.get(model_name, match.group(1))

    print(f"Could not parse model name from description: {description}")
    return None

def find_model_file(model_dir, best_model_name, horizon):
    """Find the correct model file path based on model type."""
    # Define search names for each model type
    model_search_mapping = {
        'ExtraTrees': ['ExtraTrees', 'extratrees', 'extra_trees', 'ExtraTreesRegressor'],
        'CatBoost': ['CatBoost', 'catboost', 'CatBoostRegressor'],
        'XGBoost': ['XGBoost', 'xgboost', 'XGBRegressor'],
        'LightGBM': ['LightGBM', 'lightgbm', 'LGBMRegressor'],
        'GradientBoosting': ['GradientBoosting', 'gradientboosting', 'GradientBoostingRegressor'],
        'RandomForest': ['RandomForest', 'randomforest', 'RandomForestRegressor'],
        'DecisionTree': ['DecisionTree', 'decisiontree', 'DecisionTreeRegressor']
    }
    
    search_names = model_search_mapping.get(best_model_name, [best_model_name])
    
    # Define file extensions based on model type
    if best_model_name == 'CatBoost':
        extensions = ['.cbm', '.pkl']
    elif best_model_name == 'XGBoost':
        extensions = ['.json', '.pkl']
    elif best_model_name == 'LightGBM':
        extensions = ['.txt', '.pkl']
    else:
        # Sklearn models (ExtraTrees, GradientBoosting, RandomForest, DecisionTree)
        extensions = ['.pkl', '.joblib']

    possible_paths = []
    for name in search_names:
        for ext in extensions:
            possible_paths.extend([
                os.path.join(model_dir, "individual_models", name, f"{name}_{horizon}h{ext}"),
                os.path.join(model_dir, "individual_models", f"{name}_{horizon}h{ext}"),
                os.path.join(model_dir, f"{name}_{horizon}h{ext}")
            ])

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found model file at: {path}")
            return path
    
    print(f"Could not find model file for {best_model_name}_{horizon}h")
    print(f"Searched paths: {possible_paths[:5]}...")  # Show first 5 paths for debugging
    return None

def load_selected_features(model_dir, horizon):
    """Load selected features from the JSON file in the model directory."""
    features_file = os.path.join(model_dir, "selected_features.json")

    if not os.path.exists(features_file):
        print(f"Selected features file not found at: {features_file}")
        return None

    try:
        with open(features_file, 'r') as f:
            features = json.load(f)
        print(f"Loaded {len(features)} selected features for {horizon}h model")
        return features
    except Exception as e:
        print(f"Error loading selected features: {e}")
        return None

def get_model_format_and_extension(model_type):
    """Get the appropriate format and file extension for each model type."""
    format_mapping = {
        'CatBoost': ('cbm', '.cbm'),
        'XGBoost': ('json', '.json'),
        'LightGBM': ('txt', '.txt'),
        'GradientBoosting': ('pkl', '.pkl'),
        'ExtraTrees': ('pkl', '.pkl'),
        'RandomForest': ('pkl', '.pkl'),
        'DecisionTree': ('pkl', '.pkl')
    }
    return format_mapping.get(model_type, ('pkl', '.pkl'))

def convert_model_to_standard_format(model_obj, model_type, output_path, horizon):
    """Convert different model types to their appropriate native format and clean up old files."""
    # First, clean up old model files for this horizon
    cleanup_old_models(horizon)
    
    # Get the appropriate format and extension
    model_format, extension = get_model_format_and_extension(model_type)
    output_path = output_path.replace('.pkl', extension)
    
    try:
        if model_type == 'CatBoost':
            # CatBoost native format (.cbm)
            # Contains: model structure, feature importance, training info
            model_obj.save_model(output_path)
            print(f"Saved CatBoost model in native .cbm format")
            
        elif model_type == 'XGBoost':
            # XGBoost native format (.json)
            # Contains: learner, feature_names, feature_types, model structure
            model_obj.save_model(output_path)
            print(f"Saved XGBoost model in native .json format")
            
        elif model_type == 'LightGBM':
            # LightGBM native format (.txt)
            # Contains: features and model weights in text format
            if hasattr(model_obj, 'save_model'):
                model_obj.save_model(output_path)
            else:
                # If it's a Booster object
                model_obj.save_model(output_path)
            print(f"Saved LightGBM model in native .txt format")
            
        else:
            # Sklearn models: GradientBoosting, ExtraTrees, RandomForest, DecisionTree
            # Contains: weights, loss_changes, parameters, iterations, etc.
            joblib.dump(model_obj, output_path)
            print(f"Saved {model_type} model in .pkl format with joblib")
            
        return output_path, model_format
        
    except Exception as e:
        print(f"Error converting {model_type} model: {e}")
        # Fallback to joblib for sklearn-compatible models
        try:
            fallback_path = output_path.replace(extension, '.pkl')
            joblib.dump(model_obj, fallback_path)
            print(f"Fallback: Saved {model_type} model as .pkl")
            return fallback_path, 'pkl'
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            raise

def load_model_by_type(model_file_path, best_model_name):
    """Load model based on its type and file extension."""
    try:
        if best_model_name == 'CatBoost':
            from catboost import CatBoostRegressor
            model_obj = CatBoostRegressor()
            model_obj.load_model(model_file_path)
            print(f"Loaded CatBoost model from {model_file_path}")
            return model_obj
            
        elif best_model_name == 'XGBoost':
            from xgboost import XGBRegressor
            model_obj = XGBRegressor()
            model_obj.load_model(model_file_path)
            print(f"Loaded XGBoost model from {model_file_path}")
            return model_obj
            
        elif best_model_name == 'LightGBM':
            import lightgbm as lgb
            model_obj = lgb.Booster(model_file=model_file_path)
            print(f"Loaded LightGBM model from {model_file_path}")
            return model_obj
            
        else:
            # Sklearn models: GradientBoosting, ExtraTrees, RandomForest, DecisionTree
            model_obj = joblib.load(model_file_path)
            print(f"Loaded {best_model_name} model from {model_file_path}")
            return model_obj
            
    except Exception as e:
        print(f"Error loading {best_model_name} model from {model_file_path}: {e}")
        # Fallback: try joblib load
        try:
            model_obj = joblib.load(model_file_path)
            print(f"Fallback: Loaded {best_model_name} model with joblib")
            return model_obj
        except Exception as fallback_error:
            raise Exception(f"Could not load model with any method. Original error: {e}, Fallback error: {fallback_error}")

def main():
    try:
        # Connect to Hopsworks
        project = hopsworks.login(
            api_key_value=os.environ.get("HOPSWORKS_API_KEY"),
            project=os.environ.get("HOPSWORKS_PROJECT")
        )
        mr = project.get_model_registry()
        print("Successfully connected to Hopsworks.")

        ensure_directories()

        model_info = {}
        
        for horizon in [24, 48, 72]:
            try:
                model_name = f"aqi_forecast_model_{horizon}h"
                print(f"\n{'='*50}")
                print(f"Processing {model_name}...")
                print(f"{'='*50}")

                # Get the latest model version
                model_info_obj = get_latest_model_version(mr, model_name)
                if not model_info_obj:
                    print(f"Could not find any version of {model_name}")
                    continue

                # Download model
                print("Downloading model from Hopsworks...")
                model_dir = model_info_obj.download()

                # Load selected features
                features = load_selected_features(model_dir, horizon)
                if not features:
                    print(f"Could not load selected features for {horizon}h model")
                    continue

                # Save selected features to backend/config
                features_output_path = f"backend/config/selected_features_{horizon}h.json"
                with open(features_output_path, 'w') as f:
                    json.dump(features, f, indent=2)
                print(f"Saved selected features to {features_output_path}")

                # Parse best model name from description
                description = model_info_obj.description
                best_model_name = parse_best_model_from_description(description)

                if not best_model_name:
                    print(f"Could not find best model name for {horizon}h model")
                    print(f"Description: {description}")
                    continue

                print(f"Best model for {horizon}h: {best_model_name}")

                # Find model file
                model_file_path = find_model_file(model_dir, best_model_name, horizon)
                if not model_file_path:
                    print(f"Could not find model file for {best_model_name}_{horizon}h")
                    continue

                # Load the model
                print(f"Loading {best_model_name} model...")
                model_obj = load_model_by_type(model_file_path, best_model_name)

                # Save model to backend/models (this will clean up old files first)
                output_model_path = f"backend/models/model_{horizon}h.pkl"
                print(f"Converting and saving model...")
                final_path, model_format = convert_model_to_standard_format(
                    model_obj, best_model_name, output_model_path, horizon
                )

                # Store model metadata
                model_info[horizon] = {
                    "model_type": best_model_name,
                    "model_file": os.path.basename(final_path),
                    "model_format": model_format,
                    "version": model_info_obj.version,
                    "features_count": len(features),
                    "updated_at": datetime.now().isoformat(),
                    "description": description
                }

                print(f"Successfully processed {horizon}h model: {best_model_name} -> {model_format}")

            except Exception as e:
                print(f"Failed to process {horizon}h model: {e}")
                import traceback
                traceback.print_exc()

        # Save model metadata
        metadata_path = "backend/config/model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"\n{'='*60}")
        print("MODEL DOWNLOAD COMPLETED")
        print(f"{'='*60}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Total models processed: {len(model_info)}")

        # Show final model summary
        print(f"\n{'='*60}")
        print("FINAL MODEL SUMMARY")
        print(f"{'='*60}")
        for horizon, info in model_info.items():
            print(f"{horizon}h: {info['model_type']} ({info['model_file']}) - Version {info['version']}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
