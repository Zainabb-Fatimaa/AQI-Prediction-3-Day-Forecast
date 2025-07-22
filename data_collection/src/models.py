"""
Simple data structures for AQI data collection.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List


def create_aqi_record(
    city: str,
    source: str,
    state: Optional[str] = None,
    country: Optional[str] = None,
    aqi: Optional[int] = None,
    pm25: Optional[float] = None,
    pm10: Optional[float] = None,
    no2: Optional[float] = None,
    so2: Optional[float] = None,
    co: Optional[float] = None,
    o3: Optional[float] = None,
    temperature: Optional[float] = None,
    humidity: Optional[float] = None,
    pressure: Optional[float] = None,
    wind_speed: Optional[float] = None,
    wind_direction: Optional[float] = None,
    visibility: Optional[float] = None,
    cloud_cover: Optional[float] = None,
    precipitation: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    timestamp: Optional[str] = None,
    created_at: Optional[str] = None
) -> Dict[str, Any]:
    """Create a simple AQI data record with essential fields only."""
    now = datetime.now().isoformat()
    return {
        'city': city,
        'state': state,
        'country': country,
        'timestamp': timestamp or now,
        'source': source,
        'aqi': aqi,
        'pm25': pm25,
        'pm10': pm10,
        'no2': no2,
        'so2': so2,
        'co': co,
        'o3': o3,
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'visibility': visibility,
        'cloud_cover': cloud_cover,
        'precipitation': precipitation,
        'latitude': latitude,
        'longitude': longitude,
        'created_at': created_at or now
    }


def merge_aqi_data(data_sources: list) -> dict:
    """
    Merge data from multiple sources.
    - Always take AQI and Temperature from AirVisual.
    - All other features (pollutants, weather) from AQICN or OpenWeather if available.
    - Preserve as much metadata as possible (state, country, lat/lon, timestamp, etc.).
    """
    merged = {
        "city": None,
        "state": None,
        "country": None,
        "timestamp": None,
        "source": "merged",
        "aqi": None,
        "pm25": None,
        "pm10": None,
        "no2": None,
        "so2": None,
        "co": None,
        "o3": None,
        "temperature": None,
        "humidity": None,
        "pressure": None,
        "wind_speed": None,
        "wind_direction": None,
        "visibility": None,
        "cloud_cover": None,
        "precipitation": None,
        "latitude": None,
        "longitude": None,
        "created_at": None
    }

    for source in data_sources:
        # Always fill in identity fields if not set
        for field in ["city", "state", "country", "latitude", "longitude", "timestamp", "created_at"]:
            if merged[field] is None and source.get(field) is not None:
                merged[field] = source[field]

        # AirVisual overrides for AQI and temp
        if source.get("source") == "airvisual":
            if source.get("aqi") is not None:
                merged["aqi"] = source["aqi"]
            if source.get("temperature") is not None:
                merged["temperature"] = source["temperature"]

        # Other fields from AQICN/OpenWeather only
        if source.get("source") != "airvisual":
            for field in [
                "pm25", "pm10", "no2", "so2", "co", "o3",
                "humidity", "pressure", "wind_speed", "wind_direction",
                "visibility", "cloud_cover", "precipitation"
            ]:
                if merged[field] is None and source.get(field) is not None:
                    merged[field] = source[field]

    return merged

def get_aqi_category(aqi: int) -> str:
    """Get AQI category based on EPA standards."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous" 
