import os
import re
import json
from fastapi import FastAPI, HTTPException, Query, Depends, Request
from pydantic import BaseModel, conint
import hopsworks
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI(title="AQI Forecast API",
    description="Air Quality Index forecasting service for Karachi",
    version="1.0.0")

# --- Rate Limiting ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8501", # Default Streamlit port
    # Add deployed Streamlit URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@limiter.limit("20/minute")
async def root(request: Request):
    return {"message": "Welcome to the Public AQI Forecast API!"}

# --- 1. Hopsworks Connection ---
try:
    project = hopsworks.login(
        api_key_value=os.environ.get("HOPSWORKS_API_KEY"),
        project=os.environ.get("HOPSWORKS_PROJECT")
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    print("Successfully connected to Hopsworks.")
except Exception as e:
    print(f"Error connecting to Hopsworks: {e}")
    project = None
    fs = None
    mr = None

# --- 2. Load Models, Selected Features and Set Up Caching ---
models = {}
model_versions = {}
selected_features = {}  # Store selected features for each horizon
feature_group_cache = {}  # Cache feature groups to avoid repeated loading
forecast_cache = {}
cache_timestamp = None

def get_latest_model_version(model_name):
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

def get_feature_group_for_horizon(horizon):
    """Get the specific feature group for a given horizon."""
    try:
        fg_name = f"aqi_features_{horizon}h_prod"
        fg = fs.get_feature_group(name=fg_name)
        print(f"Found feature group: {fg_name}")
        return fg
    except Exception as e:
        print(f"Error getting feature group {fg_name}: {e}")
        return None

def get_features_for_horizon(horizon, selected_features_list, location="Karachi"):
    """Retrieve feature values for a specific horizon."""
    feature_vector = {}

    try:
        fg = get_feature_group_for_horizon(horizon)
        if not fg:
            raise Exception(f"Could not get feature group for {horizon}h")

        # Create query for the features we need
        query = fg.select(selected_features_list)

        # Get the latest data point
        feature_data = query.read()

        if not feature_data.empty:
            for feature in selected_features_list:
                if feature in feature_data.columns:
                    feature_vector[feature] = feature_data[feature].iloc[0]
                else:
                    print(f"Feature {feature} not found in feature group")
                    # Use default value for missing feature
                    feature_vector[feature] = get_default_value(feature)
        else:
            print(f"No data returned from feature group")
            # Use default values for all features
            for feature in selected_features_list:
                feature_vector[feature] = get_default_value(feature)

    except Exception as e:
        print(f"Error retrieving features for {horizon}h: {e}")
        # Fill with default values for all features
        for feature in selected_features_list:
            feature_vector[feature] = get_default_value(feature)

    return feature_vector

def get_default_value(feature_name):
    """Get appropriate default value based on feature name."""
    feature_lower = feature_name.lower()
    if 'aqi' in feature_lower:
        return 50.0  # Default AQI
    elif 'temp' in feature_lower:
        return 25.0  # Default temperature
    elif 'humidity' in feature_lower:
        return 60.0  # Default humidity
    elif 'wind' in feature_lower:
        return 5.0   # Default wind speed
    elif any(pollutant in feature_lower for pollutant in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']):
        return 10.0  # Default pollutant concentration
    else:
        return 0.0   # Generic default

def create_prediction_dataframe(feature_vector, selected_features_list):
    """Create a properly ordered DataFrame for model prediction."""
    # Create DataFrame with features in the exact order expected by the model
    ordered_data = []
    for feature in selected_features_list:
        if feature in feature_vector:
            ordered_data.append(feature_vector[feature])
        else:
            # Handle missing features with appropriate defaults
            if 'aqi' in feature.lower():
                default_value = 50.0  # Default AQI
            elif 'temp' in feature.lower():
                default_value = 25.0  # Default temperature
            elif 'humidity' in feature.lower():
                default_value = 60.0  # Default humidity
            elif 'wind' in feature.lower():
                default_value = 5.0   # Default wind speed
            else:
                default_value = 0.0   # Generic default
            ordered_data.append(default_value)
            print(f"Using default value {default_value} for missing feature: {feature}")

    # Create DataFrame
    prediction_df = pd.DataFrame([ordered_data], columns=selected_features_list)
    return prediction_df

def find_model_file(model_dir, best_model_name, horizon):
    """Find the correct model file path based on your specific structure."""
    possible_paths = [
        # Your structure: individual_models/ModelName/ModelName_XXh.pkl
        os.path.join(model_dir, "individual_models", best_model_name, f"{best_model_name}_{horizon}h.pkl"),
        os.path.join(model_dir, "individual_models", best_model_name, f"{best_model_name}_{horizon}h.cbm"),
        os.path.join(model_dir, "individual_models", best_model_name, f"{best_model_name}_{horizon}h.json"),

        # Alternative extensions
        os.path.join(model_dir, "individual_models", best_model_name, f"{best_model_name}_{horizon}h.joblib"),
        os.path.join(model_dir, "individual_models", best_model_name, f"{best_model_name}_{horizon}h.txt"),

        # Fallback: flat structure
        os.path.join(model_dir, "individual_models", f"{best_model_name}_{horizon}h.pkl"),
        os.path.join(model_dir, "individual_models", f"{best_model_name}_{horizon}h.cbm"),
        os.path.join(model_dir, "individual_models", f"{best_model_name}_{horizon}h.json"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found model file at: {path}")
            return path

    # If no file found, list directory contents for debugging
    print(f"Model file not found for {best_model_name}_{horizon}h. Directory contents of {model_dir}:")
    try:
        for root, dirs, files in os.walk(model_dir):
            level = root.replace(model_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        print()
    except Exception as e:
        print(f"Error listing directory: {e}")

    return None

def parse_best_model_from_description(description):
    """Parse the best model name from description with your specific format."""
    if not description:
        return None

    # Your specific pattern: "The best model for 48h is ExtraTrees"
    pattern = r"The best model for \d+h is ([A-Za-z]+)"

    match = re.search(pattern, description, re.IGNORECASE)
    if match:
        return match.group(1)

    print(f"Description content: {description}")
    print(f"Could not parse model name from description. Expected format: 'The best model for XXh is ModelName'")
    return None

def load_models():
    """
    Loads the latest version of the models for each horizon from the registry.
    It dynamically finds and loads the best model based on the description.
    Also loads selected features for each model.
    """
    if not mr:
        print("Model Registry not available. Skipping model loading.")
        return

    for horizon in [24, 48, 72]:
        try:
            model_name = f"aqi_forecast_model_{horizon}h"

            # Get the latest version (with fallback strategies)
            model_info = get_latest_model_version(model_name)
            if not model_info:
                print(f"Could not find any version of {model_name}")
                continue

            # Store the version for later use in feature views
            model_versions[horizon] = model_info.version

            model_dir = model_info.download()

            # Load selected features for this model
            features = load_selected_features(model_dir, horizon)
            if not features:
                print(f"Could not load selected features for {horizon}h model")
                continue
            selected_features[horizon] = features

            description = model_info.description
            best_model_name = parse_best_model_from_description(description)

            if not best_model_name:
                print(f"Could not find best model name in description for {horizon}h model, version {model_info.version}.")
                print(f"Description: {description}")
                continue

            print(f"🔍 Best model for {horizon}h (Version {model_info.version}) is '{best_model_name}'. Loading corresponding file...")

            # Find the model file with multiple strategies
            model_file_path = find_model_file(model_dir, best_model_name, horizon)

            if not model_file_path:
                print(f"Could not find any model file for {best_model_name}_{horizon}h")
                continue

            # Load the model based on type
            model_obj = None
            if best_model_name in ['CatBoost', 'catboost']:
                from catboost import CatBoostRegressor
                model_obj = CatBoostRegressor()
                model_obj.load_model(model_file_path)
            elif best_model_name in ['XGBoost', 'xgboost']:
                from xgboost import XGBRegressor
                model_obj = XGBRegressor()
                model_obj.load_model(model_file_path)
            elif best_model_name in ['LightGBM', 'lightgbm']:
                import lightgbm as lgb
                model_obj = lgb.Booster(model_file=model_file_path)
            elif best_model_name == 'ExtraTrees':
                # ExtraTrees is typically saved with joblib/pickle
                model_obj = joblib.load(model_file_path)
            elif best_model_name in ['RandomForest', 'randomforest']:
                model_obj = joblib.load(model_file_path)
            elif best_model_name in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
                model_obj = joblib.load(model_file_path)
            else:
                # Default: try joblib first, then pickle
                try:
                    model_obj = joblib.load(model_file_path)
                except:
                    import pickle
                    with open(model_file_path, 'rb') as f:
                        model_obj = pickle.load(f)

            models[horizon] = model_obj
            print(f"Loaded model for {horizon}h horizon (Version: {model_info.version}, Type: {best_model_name})")

        except Exception as e:
            print(f"Failed to load model for {horizon}h horizon: {e}")
            import traceback
            traceback.print_exc()

# Load models
load_models()

# --- 3. Pydantic Models for Response Schemas ---
class Forecast(BaseModel):
    horizon_hours: int
    predicted_aqi: float
    risk_level: str

class ForecastResponse(BaseModel):
    location: str
    forecast_generated_at: str
    forecasts: List[Forecast]

class HourlyForecast(BaseModel):
    timestamp: datetime
    horizon_hours: int
    predicted_aqi: float
    risk_level: str

class HourlyForecastResponse(BaseModel):
    location: str
    forecast_generated_at: str
    forecasts: List[HourlyForecast]

class HistoricalData(BaseModel):
    timestamp: datetime
    aqi: float
    pm25: float
    pm10: float
    o3: float
    no2: float
    so2: float
    co: float
    temperature: float
    humidity: float
    wind_direction: float
    wind_speed: float

class HistoricalResponse(BaseModel):
    location: str
    data: List[HistoricalData]

class LocationInfo(BaseModel):
    city: str
    state: str
    country: str
    current_aqi: Optional[float]
    last_updated: Optional[str]

class LocationsResponse(BaseModel):
    locations: List[LocationInfo]

class TrendData(BaseModel):
    date: str
    avg_aqi: float
    min_aqi: float
    max_aqi: float
    dominant_pollutant: str

class TrendsResponse(BaseModel):
    location: str
    period: str
    data: List[TrendData]

class DashboardOverview(BaseModel):
    location: str
    current_aqi: float
    current_risk_level: str

    weekly_avg: float
    trend_direction: str  # "improving", "worsening", "stable"
    last_updated: str


# --- 4. Helper Functions ---
def get_aqi_risk_level(aqi: float) -> str:
    if 0 <= aqi <= 50: return "Good"
    if 51 <= aqi <= 100: return "Moderate"
    if 101 <= aqi <= 150: return "Unhealthy for Sensitive Groups"
    if 151 <= aqi <= 200: return "Unhealthy"
    if 201 <= aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

def get_dominant_pollutant(row):
    """Determine the dominant pollutant based on highest concentration."""
    pollutants = {'pm25': row['pm25'], 'pm10': row['pm10'], 'o3': row['o3'],
                  'no2': row['no2'], 'so2': row['so2'], 'co': row['co']}
    return max(pollutants.items(), key=lambda x: x[1])[0].upper()

def calculate_trend_direction(recent_avg, older_avg):
    """Calculate if air quality is improving, worsening, or stable."""
    if recent_avg < older_avg * 0.9:
        return "improving"
    elif recent_avg > older_avg * 1.1:
        return "worsening"
    else:
        return "stable"

def safe_parse_datetime(df, column_name):
    """Safely parse datetime column with various formats."""
    try:
        # First try the standard pandas to_datetime
        df[column_name] = pd.to_datetime(df[column_name])
        return df
    except Exception as e1:
        print(f"First attempt failed: {e1}")
        try:
            # Try with format='mixed' (pandas 2.0+)
            df[column_name] = pd.to_datetime(df[column_name], format='mixed')
            return df
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            try:
                # Try with ISO8601 format
                df[column_name] = pd.to_datetime(df[column_name], format='ISO8601')
                return df
            except Exception as e3:
                print(f"Third attempt failed: {e3}")
                try:
                    # Manual parsing for common timestamp formats with microseconds
                    def parse_timestamp(ts_str):
                        if pd.isna(ts_str):
                            return pd.NaT
                        # Handle various timestamp formats
                        ts_str = str(ts_str).strip()

                        # Common formats to try
                        formats = [
                            '%Y-%m-%d %H:%M:%S.%f',
                            '%Y-%m-%dT%H:%M:%S.%f',
                            '%Y-%m-%d %H:%M:%S',
                            '%Y-%m-%dT%H:%M:%S',
                            '%Y-%m-%d',
                        ]

                        for fmt in formats:
                            try:
                                return pd.to_datetime(ts_str, format=fmt)
                            except:
                                continue

                        # If all formats fail, try pandas default parsing
                        return pd.to_datetime(ts_str, errors='coerce')

                    df[column_name] = df[column_name].apply(parse_timestamp)
                    return df
                except Exception as e4:
                    print(f"Final attempt failed: {e4}")
                    # Last resort: convert to string and use errors='coerce'
                    df[column_name] = pd.to_datetime(df[column_name].astype(str), errors='coerce')
                    return df

# --- 5. Updated Forecast Endpoint ---

@app.get("/forecast", response_model=ForecastResponse)
@limiter.limit("10/minute")
async def get_forecast(
    request: Request,
    horizon: Optional[conint(ge=24, le=72)] = Query(None, description="Specific horizon (24, 48, or 72). If not provided, returns all.")
):
    """
    Provides an AQI forecast for Karachi using selected features.
    Can be for a specific horizon or a combined 3-day forecast.
    """
    global forecast_cache, cache_timestamp

    if not models or not fs:
        raise HTTPException(status_code=503, detail="Models or Feature Store are not available.")

    # If a specific horizon is requested, generate it on-demand
    if horizon:
        if horizon not in [24, 48, 72]:
            raise HTTPException(status_code=400, detail="Invalid horizon. Please use 24, 48, or 72.")

        print(f"Generating on-demand forecast for {horizon}h...")
        try:
            model = models[horizon]
            features_needed = selected_features[horizon]

            # Get feature values from the specific feature group for this horizon
            feature_vector = get_features_for_horizon(horizon, features_needed, "Karachi")

            # Create prediction DataFrame with proper feature ordering
            prediction_df = create_prediction_dataframe(feature_vector, features_needed)

            print(f"Prediction DataFrame shape: {prediction_df.shape}")
            print(f"Features used: {list(prediction_df.columns)}")

            # Make prediction
            if hasattr(model, 'predict'):
                aqi_prediction = model.predict(prediction_df)[0]
            else:
                # For LightGBM Booster
                aqi_prediction = model.predict(prediction_df.values)[0]

            single_forecast = Forecast(
                horizon_hours=horizon,
                predicted_aqi=round(float(aqi_prediction), 2),
                risk_level=get_aqi_risk_level(aqi_prediction)
            )

            return ForecastResponse(
                location="Karachi, Sindh, Pakistan",
                forecast_generated_at=datetime.utcnow().isoformat(),
                forecasts=[single_forecast]
            )
        except Exception as e:
            print(f"Error in forecast generation: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to generate forecast for {horizon}h: {e}")

    # If no horizon is specified, use the daily cache for the combined forecast
    cache_key = f"{date.today()}"
    if cache_timestamp and cache_timestamp == date.today() and cache_key in forecast_cache:
        print("Returning cached full forecast.")
        return forecast_cache[cache_key]

    print("Cache is stale or empty. Generating new full forecast...")
    predictions = []
    try:
        for h, model in sorted(models.items()):
            features_needed = selected_features[h]

            # Get feature values from the specific feature group for this horizon
            feature_vector = get_features_for_horizon(h, features_needed, "Karachi")

            # Create prediction DataFrame with proper feature ordering
            prediction_df = create_prediction_dataframe(feature_vector, features_needed)

            # Make prediction
            if hasattr(model, 'predict'):
                aqi_prediction = model.predict(prediction_df)[0]
            else:
                # For LightGBM Booster
                aqi_prediction = model.predict(prediction_df.values)[0]

            predictions.append(
                Forecast(
                    horizon_hours=h,
                    predicted_aqi=round(float(aqi_prediction), 2),
                    risk_level=get_aqi_risk_level(aqi_prediction)
                )
            )

        forecast_response = ForecastResponse(
            location="Karachi, Sindh, Pakistan",
            forecast_generated_at=datetime.utcnow().isoformat(),
            forecasts=predictions
        )

        forecast_cache[cache_key] = forecast_response
        cache_timestamp = date.today()

        print("New full forecast generated and cached.")
        return forecast_response

    except Exception as e:
        print(f"Error in full forecast generation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate full forecast: {e}")

@app.get("/forecast/hourly", response_model=HourlyForecastResponse)
@limiter.limit("10/minute")
async def get_hourly_forecast(
    request: Request,
    horizon: Optional[str] = Query("Next 3 days", description="Specify a forecast horizon (e.g., 24, 48, 72) or 'Next 3 days'.")
):
    """
    Provides an hourly AQI forecast for Karachi.
    """
    if not models or not fs:
        raise HTTPException(status_code=503, detail="Models or Feature Store are not available.")

    try:
        # 1. Get current AQI value
        dataset_api = project.get_dataset_api()
        local_path = "aqi_data_downloaded.csv"
        dataset_api.download("Resources/aqi_data.csv/aqi_data.csv", local_path, overwrite=True)
        df = pd.read_csv(local_path)
        df_location = df[df['city'].str.lower() == "karachi"]
        if df_location.empty:
            raise HTTPException(status_code=404, detail="No historical data for Karachi")

        df_location['timestamp'] = pd.to_datetime(df_location['timestamp'], format='mixed')
        latest_data = df_location.loc[df_location['timestamp'].idxmax()]
        current_aqi = float(latest_data['aqi'])
        start_time = latest_data['timestamp']

        # 2. Get 24, 48, 72h forecasts
        major_forecasts = {}
        for h in [24, 48, 72]:
            forecast_response = await get_forecast(request, horizon=h)
            major_forecasts[h] = forecast_response.forecasts[0].predicted_aqi

        # 3. Create points for interpolation
        known_points_x = [0, 24, 48, 72]
        known_points_y = [current_aqi, major_forecasts[24], major_forecasts[48], major_forecasts[72]]

        # 4. Create target hours for interpolation
        if horizon.lower() == "next 3 days":
            target_hours = np.arange(1, 25) # Changed to 25 to include 24
        else:
            try:
                horizon_hours = int(horizon)
                if horizon_hours > 72:
                    raise HTTPException(status_code=400, detail="Maximum horizon is 72 hours.")
                target_hours = np.arange(1, horizon_hours + 1)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid horizon specified.")

        # 5. Interpolate
        hourly_aqi_values = np.interp(target_hours, known_points_x, known_points_y)

        # 6. Format response
        hourly_forecasts = []
        for i, hour in enumerate(target_hours):
            forecast_time = start_time + timedelta(hours=int(hour))
            predicted_aqi = round(float(hourly_aqi_values[i]), 2)
            hourly_forecasts.append(
                HourlyForecast(
                    timestamp=forecast_time,
                    horizon_hours=int(hour),
                    predicted_aqi=predicted_aqi,
                    risk_level=get_aqi_risk_level(predicted_aqi)
                )
            )

        return HourlyForecastResponse(
            location="Karachi",
            forecast_generated_at=datetime.utcnow().isoformat(),
            forecasts=hourly_forecasts
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate hourly forecast: {e}")


@app.get("/locations", response_model=LocationsResponse)
@limiter.limit("15/minute")
async def get_locations(request: Request):
    """
    Get all available locations with their current AQI status.
    """
    if not project:
        raise HTTPException(status_code=503, detail="Hopsworks connection is not available.")

    try:
        dataset_api = project.get_dataset_api()
        local_path = "aqi_data_downloaded.csv"

        dataset_api.download("Resources/aqi_data.csv/aqi_data.csv", local_path, overwrite=True)
        df = pd.read_csv(local_path)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)


        # Safe datetime parsing
        df = safe_parse_datetime(df, 'timestamp')

        # Get latest data for each city
        latest_data = df.loc[df.groupby('city')['timestamp'].idxmax()]

        locations = []
        for _, row in latest_data.iterrows():
            # Handle potential NaT (Not a Time) values
            last_updated = None
            if pd.notna(row['timestamp']):
                try:
                    last_updated = row['timestamp'].strftime("%Y-%m-%d %H:%M")
                except:
                    last_updated = str(row['timestamp'])

            locations.append(
                LocationInfo(
                    city=row['city'],
                    state=row['state'],
                    country=row['country'],
                    current_aqi=float(row['aqi']) if pd.notna(row['aqi']) else None,
                    last_updated=last_updated
                )
            )

        return LocationsResponse(locations=locations)

    except Exception as e:
        print(f"Error in get_locations: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not retrieve locations: {e}")

@app.get("/historical/{location}", response_model=HistoricalResponse)
@limiter.limit("10/minute")
async def get_historical_data(request: Request, location: str, days: conint(ge=1, le=30) = Query(default=7, description="Number of days (max 30)")):
    """
    Provides historical weather and pollutant data for a specific city.
    """
    if not project:
        raise HTTPException(status_code=503, detail="Hopsworks connection is not available.")

    try:
        dataset_api = project.get_dataset_api()
        local_path = "aqi_data_downloaded.csv"

        dataset_api.download("Resources/aqi_data.csv/aqi_data.csv", local_path, overwrite=True)
        df = pd.read_csv(local_path)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)

        df_location = df[df['city'].str.lower() == location.lower()]
        if df_location.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for city: {location}")

        # Safe datetime parsing
        df_location = safe_parse_datetime(df_location, 'timestamp')

        # Determine date range based on period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df_filtered = df_location[df_location['timestamp'] >= start_date].copy()

        # Fix column mapping - use the correct CSV column names
        required_columns = ['timestamp', 'aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'temperature', 'humidity', 'wind_direction', 'wind_speed']

        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df_filtered.columns]
        if missing_columns:
            raise HTTPException(status_code=500, detail=f"Missing columns in data: {missing_columns}")

        df_response = df_filtered[required_columns].copy()

        # Fill missing values
        df_response = df_response.fillna(method='ffill').fillna(method='bfill')

        # Sort by timestamp
        df_response = df_response.sort_values('timestamp')

        # Convert to dict and ensure column names match Pydantic model exactly
        records = df_response.to_dict(orient='records')

        historical_data = []
        for record in records:
            try:
                historical_data.append(HistoricalData(**record))
            except Exception as e:
                print(f"Error creating HistoricalData: {e}")
                print(f"Record: {record}")
                raise

        return HistoricalResponse(location=location, data=historical_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve historical data: {e}")

@app.get("/trends/{location}", response_model=TrendsResponse)
@limiter.limit("10/minute")
async def get_trends(request: Request, location: str, period: str = Query(default="weekly", description="weekly or monthly")):
    """
    Get air quality trends for a specific location.
    """
    if not project:
        raise HTTPException(status_code=503, detail="Hopsworks connection is not available.")

    try:
        dataset_api = project.get_dataset_api()
        local_path = "aqi_data_downloaded.csv"

        dataset_api.download("Resources/aqi_data.csv/aqi_data.csv", local_path, overwrite=True)
        df = pd.read_csv(local_path)

        df_location = df[df['city'].str.lower() == location.lower()]
        if df_location.empty:
            raise HTTPException(status_code=404, detail=f"No data found for city: {location}")

        # Safe datetime parsing
        df_location = safe_parse_datetime(df_location, 'timestamp')
        # Determine date range based on period
        end_date = datetime.now()
        if period == "weekly":
            start_date = end_date - timedelta(weeks=4)  # 4 weeks of data
            freq = 'D'  # Daily aggregation
        else:  # monthly
            start_date = end_date - timedelta(days=180)  # 6 months of data
            freq = 'W'  # Weekly aggregation

        df_filtered = df_location[df_location['timestamp'] >= start_date].copy()

        # Group by period and calculate statistics
        df_filtered.set_index('timestamp', inplace=True)
        df_filtered.ffill(inplace=True)
        df_filtered.bfill(inplace=True)
        df_filtered.fillna(df_filtered.mean(numeric_only=True), inplace=True)
        grouped = df_filtered.resample(freq).agg({
            'aqi': ['mean', 'min', 'max'],
            'pm25': 'mean',
            'pm10': 'mean',
            'o3': 'mean',
            'no2': 'mean',
            'so2': 'mean',
            'co': 'mean'
        }).round(2)


        trends = []
        for date_idx, row in grouped.iterrows():
            # Determine dominant pollutant
            pollutant_values = {
                'PM2.5': row[('pm25', 'mean')],
                'PM10': row[('pm10', 'mean')],
                'O3': row[('o3', 'mean')],
                'NO2': row[('no2', 'mean')],
                'SO2': row[('so2', 'mean')],
                'CO': row[('co', 'mean')]
            }
            dominant = max(pollutant_values.items(), key=lambda x: x[1] if pd.notna(x[1]) else 0)[0]

            trends.append(
                TrendData(
                    date=date_idx.strftime("%Y-%m-%d"),
                    avg_aqi=float(row[('aqi', 'mean')]) if pd.notna(row[('aqi', 'mean')]) else 0.0,
                    min_aqi=float(row[('aqi', 'min')]) if pd.notna(row[('aqi', 'min')]) else 0.0,
                    max_aqi=float(row[('aqi', 'max')]) if pd.notna(row[('aqi', 'max')]) else 0.0,
                    dominant_pollutant=dominant
                )
            )

        return TrendsResponse(location=location, period=period, data=trends)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not retrieve trends: {e}")

@app.get("/dashboard/overview", response_model=DashboardOverview)
@limiter.limit("10/minute")
async def get_dashboard_overview(request: Request, location: str = Query(default="Karachi", description="City name")):
    """
    Get comprehensive dashboard data for a location in a single call.
    """
    if not project:
        raise HTTPException(status_code=503, detail="Services not available.")

    try:
        # Get current data
        dataset_api = project.get_dataset_api()
        local_path = "aqi_data_downloaded.csv"
        dataset_api.download("Resources/aqi_data.csv/aqi_data.csv", local_path, overwrite=True)
        df = pd.read_csv(local_path)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)

        df_location = df[df['city'].str.lower() == location.lower()]
        if df_location.empty:
            raise HTTPException(status_code=404, detail=f"No data found for city: {location}")

        # Safe datetime parsing
        df_location = safe_parse_datetime(df_location, 'timestamp')

        # Get latest data point
        latest_data = df_location.loc[df_location['timestamp'].idxmax()]
        current_aqi = float(latest_data['aqi'])

        # # Get forecasts only for Karachi
        # forecasts = None
        # if location.lower() == "karachi":
        #     forecast_response = await get_forecast()
        #     forecasts = forecast_response.forecasts

        # Calculate weekly average
        week_ago = datetime.now() - timedelta(days=7)
        recent_data = df_location[df_location['timestamp'] >= week_ago]
        weekly_avg = float(recent_data['aqi'].mean()) if not recent_data.empty else current_aqi

        # Calculate trend direction
        two_weeks_ago = datetime.now() - timedelta(days=14)
        older_data = df_location[(df_location['timestamp'] >= two_weeks_ago) & (df_location['timestamp'] < week_ago)]
        older_avg = float(older_data['aqi'].mean()) if not older_data.empty else weekly_avg

        trend_direction = calculate_trend_direction(weekly_avg, older_avg)

        # Handle potential NaT timestamps
        last_updated = "Unknown"
        if pd.notna(latest_data['timestamp']):
            try:
                last_updated = latest_data['timestamp'].strftime("%Y-%m-%d %H:%M")
            except:
                last_updated = str(latest_data['timestamp'])

        return DashboardOverview(
            location=location,
            current_aqi=current_aqi,
            current_risk_level=get_aqi_risk_level(current_aqi),
            # forecasts=forecasts,
            weekly_avg=round(weekly_avg, 2),
            trend_direction=trend_direction,
            last_updated=last_updated
        )

    except Exception as e:
        print(f"Error in get_dashboard_overview: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not retrieve dashboard overview: {e}")
