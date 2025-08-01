import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict
import logging

from fastapi import FastAPI, HTTPException, Query, Depends, Request
from pydantic import BaseModel, conint
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="AQI Forecast API",
    description="Air Quality Index forecasting service for Karachi",
    version="2.0.0"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"}
    )

# CORS Middleware
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8501",
    "https://*.vercel.app",
    "https://*.netlify.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
models = {}
selected_features = {}
model_metadata = {}
aqi_data = None
feature_data = {}
data_loaded = False

# Load models and data on startup
def load_models_and_data():
    """Load models, features, and data from local files"""
    global models, selected_features, model_metadata, aqi_data, feature_data, data_loaded
    
    try:
        # Get the base directory (parent of current directory to go back to root)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load model metadata
        metadata_path = os.path.join(base_dir, "config", "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"Loaded model metadata for {len(model_metadata)} models")
        
        # Load models and selected features for each horizon
        for horizon in [24, 48, 72]:
            try:
                # Load selected features
                features_path = os.path.join(base_dir, "config", f"selected_features_{horizon}h.json")
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        selected_features[horizon] = json.load(f)
                    logger.info(f"Loaded {len(selected_features[horizon])} features for {horizon}h")
                
                # Load model
                if str(horizon) in model_metadata:
                    model_info = model_metadata[str(horizon)]
                    model_file = model_info["model_file"]
                    model_format = model_info["model_format"]
                    model_type = model_info["model_type"]
                    
                    model_path = os.path.join(base_dir, "models", model_file)
                    
                    if os.path.exists(model_path):
                        # Load model based on format
                        if model_format == 'cbm':  # CatBoost
                            from catboost import CatBoostRegressor
                            model_obj = CatBoostRegressor()
                            model_obj.load_model(model_path)
                        elif model_format == 'json':  # XGBoost
                            from xgboost import XGBRegressor
                            model_obj = XGBRegressor()
                            model_obj.load_model(model_path)
                        elif model_format == 'txt':  # LightGBM
                            import lightgbm as lgb
                            model_obj = lgb.Booster(model_file=model_path)
                        else:  # joblib/pickle
                            model_obj = joblib.load(model_path)
                        
                        models[horizon] = {
                            'model': model_obj,
                            'type': model_type,
                            'format': model_format
                        }
                        logger.info(f"Loaded {model_type} model for {horizon}h")
                    else:
                        logger.warning(f"Model file not found: {model_path}")
                
            except Exception as e:
                logger.error(f"Error loading model for {horizon}h: {e}")
        
        # Load AQI data
        aqi_data_path = os.path.join(base_dir, "data", "aqi_data.csv")
        if os.path.exists(aqi_data_path):
            aqi_data = pd.read_csv(aqi_data_path)
            aqi_data['timestamp'] = pd.to_datetime(aqi_data['timestamp'], format='mixed', errors='coerce')
            logger.info(f"Loaded AQI data: {len(aqi_data)} records")
        
        # Load feature data for each horizon
        for horizon in [24, 48, 72]:
            features_data_path = os.path.join(base_dir, "data", f"features_{horizon}h.csv")
            if os.path.exists(features_data_path):
                feature_data[horizon] = pd.read_csv(features_data_path)
                logger.info(f"Loaded feature data for {horizon}h: {len(feature_data[horizon])} records")
        
        data_loaded = len(models) > 0 and aqi_data is not None
        logger.info(f"Data loading completed. Models loaded: {len(models)}, Data loaded: {data_loaded}")
        
    except Exception as e:
        logger.error(f"Error loading models and data: {e}")
        data_loaded = False

# Load data on startup
load_models_and_data()

# Pydantic Models for Response Schemas
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

class DashboardOverview(BaseModel):
    location: str
    current_aqi: float
    current_risk_level: str
    weekly_avg: float
    trend_direction: str
    last_updated: str

# Helper Functions
def get_aqi_risk_level(aqi: float) -> str:
    if 0 <= aqi <= 50: return "Good"
    if 51 <= aqi <= 100: return "Moderate"
    if 101 <= aqi <= 150: return "Unhealthy for Sensitive Groups"
    if 151 <= aqi <= 200: return "Unhealthy"
    if 201 <= aqi <= 300: return "Very Unhealthy"
    return "Hazardous"

def get_default_value(feature_name):
    """Get appropriate default value based on feature name."""
    feature_lower = feature_name.lower()
    if 'aqi' in feature_lower:
        return 50.0
    elif 'temp' in feature_lower:
        return 25.0
    elif 'humidity' in feature_lower:
        return 60.0
    elif 'wind' in feature_lower:
        return 5.0
    elif any(pollutant in feature_lower for pollutant in ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']):
        return 10.0
    else:
        return 0.0

def get_features_for_prediction(horizon, location="Karachi"):
    """Get feature values for prediction from loaded data"""
    try:
        if horizon not in feature_data or horizon not in selected_features:
            raise Exception(f"No feature data available for {horizon}h")
        
        features_needed = selected_features[horizon]
        df = feature_data[horizon]
        
        if df.empty:
            raise Exception(f"Feature data is empty for {horizon}h")
        
        # Get the latest row
        latest_row = df.iloc[0]
        feature_vector = {}
        
        for feature in features_needed:
            if feature in df.columns:
                value = latest_row[feature]
                feature_vector[feature] = float(value) if pd.notna(value) else get_default_value(feature)
            else:
                feature_vector[feature] = get_default_value(feature)
        
        return feature_vector
        
    except Exception as e:
        logger.error(f"Error getting features for {horizon}h: {e}")
        # Return default values
        features_needed = selected_features.get(horizon, [])
        return {feature: get_default_value(feature) for feature in features_needed}

def make_prediction(horizon, feature_vector):
    """Make prediction using the loaded model"""
    try:
        if horizon not in models:
            raise Exception(f"No model available for {horizon}h")
        
        model_info = models[horizon]
        model_obj = model_info['model']
        features_needed = selected_features[horizon]
        
        # Create prediction DataFrame
        ordered_data = [feature_vector.get(feature, get_default_value(feature)) for feature in features_needed]
        prediction_df = pd.DataFrame([ordered_data], columns=features_needed)
        
        # Make prediction based on model type
        if model_info['format'] == 'txt':  # LightGBM Booster
            prediction = model_obj.predict(prediction_df.values)[0]
        else:
            prediction = model_obj.predict(prediction_df)[0]
        
        return float(prediction)
        
    except Exception as e:
        logger.error(f"Error making prediction for {horizon}h: {e}")
        raise

def calculate_trend_direction(recent_avg, older_avg):
    """Calculate if air quality is improving, worsening, or stable."""
    if recent_avg < older_avg * 0.9:
        return "improving"
    elif recent_avg > older_avg * 1.1:
        return "worsening"
    else:
        return "stable"

# API Endpoints
@app.get("/")
@limiter.limit("20/minute")
async def root(request: Request):
    return {
        "message": "Welcome to the Lightweight AQI Forecast API!",
        "version": "2.0.0",
        "models_loaded": len(models),
        "data_available": data_loaded
    }

@app.get("/forecast", response_model=ForecastResponse)
@limiter.limit("10/minute")
async def get_forecast(
    request: Request,
    horizon: Optional[int] = Query(None, ge=24, le=72, description="Specific horizon (24, 48, or 72). If not provided, returns all.")
):
    """
    Provides an AQI forecast for Karachi using pre-loaded models and features.
    """
    if not data_loaded:
        raise HTTPException(status_code=503, detail="Models or data are not available.")

    try:
        forecasts = []
        
        # If specific horizon requested
        if horizon:
            if horizon not in models:
                raise HTTPException(status_code=400, detail=f"Model not available for {horizon}h horizon")
            
            feature_vector = get_features_for_prediction(horizon, "Karachi")
            prediction = make_prediction(horizon, feature_vector)
            
            forecasts.append(Forecast(
                horizon_hours=horizon,
                predicted_aqi=round(prediction, 2),
                risk_level=get_aqi_risk_level(prediction)
            ))
        else:
            # Generate forecasts for all available horizons
            for h in sorted(models.keys()):
                feature_vector = get_features_for_prediction(h, "Karachi")
                prediction = make_prediction(h, feature_vector)
                
                forecasts.append(Forecast(
                    horizon_hours=h,
                    predicted_aqi=round(prediction, 2),
                    risk_level=get_aqi_risk_level(prediction)
                ))

        return ForecastResponse(
            location="Karachi, Sindh, Pakistan",
            forecast_generated_at=datetime.utcnow().isoformat(),
            forecasts=forecasts
        )

    except Exception as e:
        logger.error(f"Error in forecast generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate forecast: {str(e)}")

@app.get("/forecast/hourly", response_model=HourlyForecastResponse)
@limiter.limit("10/minute")
async def get_hourly_forecast(
    request: Request,
    horizon: Optional[str] = Query("Next 3 days", description="Specify a forecast horizon (e.g., 24, 48, 72) or 'Next 3 days'.")
):
    """
    Provides an hourly AQI forecast for Karachi using interpolation.
    """
    if not data_loaded or aqi_data is None:
        raise HTTPException(status_code=503, detail="Data not available.")

    try:
        # Get current AQI value
        df_karachi = aqi_data[aqi_data['city'].str.lower() == "karachi"]
        if df_karachi.empty:
            raise HTTPException(status_code=404, detail="No historical data for Karachi")

        latest_data = df_karachi.loc[df_karachi['timestamp'].idxmax()]
        
        # Safe conversion that handles both Series and scalar values
        aqi_value = latest_data['aqi']
        if isinstance(aqi_value, pd.Series):
            current_aqi = float(aqi_value.iloc[0])
        else:
            current_aqi = float(aqi_value)
        
        # Convert start_time to proper Python datetime
        start_time = latest_data['timestamp']
        if isinstance(start_time, pd.Series):
            start_time = start_time.iloc[0]
        
        if hasattr(start_time, 'to_pydatetime'):
            base_time = start_time.to_pydatetime()
        elif isinstance(start_time, pd.Timestamp):
            base_time = start_time.to_pydatetime()
        else:
            base_time = start_time

        # Get major forecasts (24, 48, 72h)
        major_forecasts = {}
        for h in [24, 48, 72]:
            if h in models:
                feature_vector = get_features_for_prediction(h, "Karachi")
                prediction = make_prediction(h, feature_vector)
                major_forecasts[h] = prediction

        # Create interpolation points
        known_points_x = [0] + list(major_forecasts.keys())
        known_points_y = [current_aqi] + list(major_forecasts.values())

        # Determine target hours
        if horizon is None or horizon.lower() == "next 3 days":
            target_hours = np.arange(1, 73)
        else:
            try:
                horizon_hours = int(horizon)
                if horizon_hours > 72:
                    raise HTTPException(status_code=400, detail="Maximum horizon is 72 hours.")
                target_hours = np.arange(1, horizon_hours + 1)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid horizon specified.")

        # Interpolate
        hourly_aqi_values = np.interp(target_hours, known_points_x, known_points_y)

        # Format response
        hourly_forecasts = []
        for i, hour in enumerate(target_hours):
            # Use base_time (converted datetime) instead of start_time
            forecast_time = base_time + timedelta(hours=int(hour))
            predicted_aqi = round(float(hourly_aqi_values[i]), 2)
            
            hourly_forecasts.append(HourlyForecast(
                timestamp=forecast_time,
                horizon_hours=int(hour),
                predicted_aqi=predicted_aqi,
                risk_level=get_aqi_risk_level(predicted_aqi)
            ))

        return HourlyForecastResponse(
            location="Karachi",
            forecast_generated_at=datetime.utcnow().isoformat(),
            forecasts=hourly_forecasts
        )

    except Exception as e:
        logger.error(f"Error in hourly forecast: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate hourly forecast: {str(e)}")

@app.get("/locations", response_model=LocationsResponse)
@limiter.limit("15/minute")
async def get_locations(request: Request):
    """
    Get all available locations with their current AQI status.
    """
    if aqi_data is None:
        raise HTTPException(status_code=503, detail="AQI data not available.")

    try:
        # Get latest data for each city
        latest_data = aqi_data.loc[aqi_data.groupby('city')['timestamp'].idxmax()]

        locations = []
        for _, row in latest_data.iterrows():
            last_updated = None
            if pd.notna(row['timestamp']):
                try:
                    last_updated = row['timestamp'].strftime("%Y-%m-%d %H:%M")
                except:
                    last_updated = str(row['timestamp'])

            locations.append(LocationInfo(
                city=row['city'],
                state=row['state'],
                country=row['country'],
                current_aqi=float(row['aqi']) if pd.notna(row['aqi']) else None,
                last_updated=last_updated
            ))

        return LocationsResponse(locations=locations)

    except Exception as e:
        logger.error(f"Error in get_locations: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve locations: {str(e)}")

@app.get("/historical/{location}", response_model=HistoricalResponse)
@limiter.limit("10/minute")
async def get_historical_data(
    request: Request, 
    location: str, 
    days: int = Query(default=7, ge=1, le=30, description="Number of days (max 30)")  
):
    """
    Provides historical weather and pollutant data for a specific city.
    """
    if aqi_data is None:
        raise HTTPException(status_code=503, detail="AQI data not available.")

    try:
        df_location = aqi_data[aqi_data['city'].str.lower() == location.lower()]
        if df_location.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for city: {location}")

        # Filter by date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df_filtered = df_location[df_location['timestamp'] >= start_date].copy()

        # Required columns
        required_columns = ['timestamp', 'aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'temperature', 'humidity', 'wind_direction', 'wind_speed']
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df_filtered.columns]
        if missing_columns:
            raise HTTPException(status_code=500, detail=f"Missing columns in data: {missing_columns}")

        df_response = df_filtered[required_columns].copy()
        
        df_response = df_response.ffill().bfill()
        
        df_response = df_response.sort_values('timestamp')

        # Convert pandas timestamps to Python datetimes for Pydantic compatibility
        df_response['timestamp'] = pd.to_datetime(df_response['timestamp'])
        
        # Convert to records and handle timestamp conversion
        records = []
        for _, row in df_response.iterrows():
            record = row.to_dict()
            # Convert pandas timestamp to Python datetime
            timestamp_val = record['timestamp']
            if hasattr(timestamp_val, 'to_pydatetime'):
                record['timestamp'] = timestamp_val.to_pydatetime()
            elif isinstance(timestamp_val, pd.Timestamp):
                record['timestamp'] = timestamp_val.to_pydatetime()
            records.append(record)

        historical_data = [HistoricalData(**record) for record in records]

        return HistoricalResponse(location=location, data=historical_data)

    except Exception as e:
        logger.error(f"Error in get_historical_data: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve historical data: {str(e)}")
    
@app.get("/dashboard/overview", response_model=DashboardOverview)
@limiter.limit("10/minute")
async def get_dashboard_overview(
    request: Request, 
    location: str = Query(default="Karachi", description="City name")
):
    """
    Get comprehensive dashboard data for a location.
    """
    if aqi_data is None:
        raise HTTPException(status_code=503, detail="Data not available.")

    try:
        df_location = aqi_data[aqi_data['city'].str.lower() == location.lower()]
        if df_location.empty:
            raise HTTPException(status_code=404, detail=f"No data found for city: {location}")

        # Get latest data point - fix the indexing issue
        latest_idx = df_location['timestamp'].idxmax()
        latest_data = df_location.loc[latest_idx]
        
        # Handle AQI value properly
        aqi_value = latest_data['aqi']
        if isinstance(aqi_value, pd.Series):
            current_aqi = float(aqi_value.iloc[0])
        else:
            current_aqi = float(aqi_value)  

        # Calculate weekly average
        week_ago = datetime.now() - timedelta(days=7)
        recent_data = df_location[df_location['timestamp'] >= week_ago]
        weekly_avg = float(recent_data['aqi'].mean()) if not recent_data.empty else current_aqi

        # Calculate trend direction
        two_weeks_ago = datetime.now() - timedelta(days=14)
        older_data = df_location[(df_location['timestamp'] >= two_weeks_ago) & (df_location['timestamp'] < week_ago)]
        older_avg = float(older_data['aqi'].mean()) if not older_data.empty else weekly_avg

        trend_direction = calculate_trend_direction(weekly_avg, older_avg)

        # Format last updated - fix pandas timestamp handling
        last_updated = "Unknown"
        timestamp_value = latest_data['timestamp']
        
        # Handle both scalar and Series cases for timestamp
        if isinstance(timestamp_value, pd.Series):
            timestamp_value = timestamp_value.iloc[0]
        
        if pd.notna(timestamp_value):
            try:
                # Convert pandas timestamp to Python datetime if needed
                if hasattr(timestamp_value, 'to_pydatetime'):
                    dt = timestamp_value.to_pydatetime()
                elif isinstance(timestamp_value, pd.Timestamp):
                    dt = timestamp_value.to_pydatetime()
                else:
                    dt = timestamp_value
                last_updated = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                last_updated = str(timestamp_value)

        return DashboardOverview(
            location=location,
            current_aqi=current_aqi,
            current_risk_level=get_aqi_risk_level(current_aqi),
            weekly_avg=round(weekly_avg, 2),
            trend_direction=trend_direction,
            last_updated=last_updated
        )

    except Exception as e:
        logger.error(f"Error in get_dashboard_overview: {e}")
        raise HTTPException(status_code=500, detail=f"Could not retrieve dashboard overview: {str(e)}")
# For Vercel deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)