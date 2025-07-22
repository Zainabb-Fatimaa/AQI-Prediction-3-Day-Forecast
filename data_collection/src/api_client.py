"""
Simple API client for AQI data collection with fallback mechanism.
"""

import requests
import time
from datetime import datetime
from typing import Dict, Any, Optional
from requests.exceptions import RequestException
import logging
import threading
import pickle
import os
from collections import defaultdict, deque

from .models import create_aqi_record, merge_aqi_data
from .unit_conversion import standardize_row


class APIClient:
    """Enhanced API client for handling multiple data sources with robust fallback, health monitoring, rate limiting, and caching."""

    def __init__(self, timeout: int = 30, rate_limit_delay: float = 1.0, max_retries: int = 3, cache_file: str = 'api_cache.pkl'):
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.session = requests.Session()

        self.request_count = 0
        self.last_request_time = 0
        self.failed_requests = 0
        self.successful_requests = 0
        self.logger = logging.getLogger("APIClient")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.api_health = defaultdict(lambda: {'status': 'unknown', 'last_success': None, 'last_failure': None, 'fail_count': 0})
        self.api_last_request = defaultdict(float)
        self.api_rate_limit_delay = defaultdict(lambda: self.rate_limit_delay)
        self.cache = {}
        self.cache_file = cache_file
        self.cache_lock = threading.Lock()
        self._load_cache()
        self.request_queue = deque()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                self.logger.info(f"Loaded API cache from {self.cache_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            self.logger.info(f"Saved API cache to {self.cache_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def _cache_key(self, city, timestamp=None):
        if not timestamp:
            timestamp = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        city_str = city.lower() if city is not None else "unknown"
        return f"{city_str}_{timestamp.isoformat()}"

    def _get_airvisual_data(self, city: str, latitude: float, longitude: float, api_key: str) -> Optional[Dict[str, Any]]:
        try:
            url = "http://api.airvisual.com/v2/nearest_city"
            params = {'lat': latitude, 'lon': longitude, 'key': api_key}
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data.get('status') != 'success':
                return None

            current = data['data']['current']
            pollution = current['pollution']
            weather = current['weather']

            return create_aqi_record(
                city=city,
                source="airvisual",
                aqi=pollution.get('aqius'),
                pm25=None,
                pm10=None,
                no2=None,
                so2=None,
                co=None,
                o3=None,
                temperature=weather.get('tp'),
                humidity=weather.get('hu'),
                pressure=weather.get('pr'),
                wind_speed=weather.get('ws'),
                wind_direction=weather.get('wd'),
                latitude=latitude,
                longitude=longitude
            )
        except Exception:
            return None

    def _get_aqicn_data(self, city: str, latitude: float, longitude: float, api_key: str) -> Optional[Dict[str, Any]]:
        try:
            url = f"https://api.waqi.info/feed/geo:{latitude};{longitude}/"
            params = {'token': api_key}
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data.get('status') != 'ok':
                return None
            aqi_data = data.get('data', {})
            iaqi = aqi_data.get('iaqi', {})
            return create_aqi_record(
                city=city,
                source="aqicn",
                aqi=aqi_data.get('aqi'),
                pm25=iaqi.get('pm25', {}).get('v'),
                pm10=iaqi.get('pm10', {}).get('v'),
                no2=iaqi.get('no2', {}).get('v'),
                so2=iaqi.get('so2', {}).get('v'),
                co=iaqi.get('co', {}).get('v'),
                o3=iaqi.get('o3', {}).get('v'),
                temperature=iaqi.get('t', {}).get('v'),
                humidity=iaqi.get('h', {}).get('v'),
                pressure=iaqi.get('p', {}).get('v'),
                wind_speed=iaqi.get('w', {}).get('v'),
                wind_direction=iaqi.get('wd', {}).get('v'),
                latitude=latitude,
                longitude=longitude
            )
        except Exception:
            return None

    def _get_openweather_data(self, city: str, latitude: float, longitude: float, api_key: str) -> Optional[Dict[str, Any]]:
        try:
            weather_url = "https://api.openweathermap.org/data/2.5/weather"
            weather_params = {'lat': latitude, 'lon': longitude, 'appid': api_key, 'units': 'metric'}
            weather_response = self.session.get(weather_url, params=weather_params, timeout=self.timeout)
            weather_response.raise_for_status()
            weather_data = weather_response.json()

            pollution_url = "https://api.openweathermap.org/data/2.5/air_pollution"
            pollution_params = {'lat': latitude, 'lon': longitude, 'appid': api_key}
            pollution_response = self.session.get(pollution_url, params=pollution_params, timeout=self.timeout)
            pollution_response.raise_for_status()
            pollution_data = pollution_response.json()

            main = weather_data.get('main', {})
            wind = weather_data.get('wind', {})
            components = pollution_data.get('list', [{}])[0].get('components', {})
            aqi_level = pollution_data.get('list', [{}])[0].get('main', {}).get('aqi')
            aqi_value = {1: 50, 2: 100, 3: 150, 4: 200, 5: 300}.get(aqi_level, None)

            return create_aqi_record(
                city=city,
                source="openweather",
                aqi=aqi_value,
                pm25=components.get('pm2_5'),
                pm10=components.get('pm10'),
                no2=components.get('no2'),
                so2=components.get('so2'),
                co=components.get('co'),
                o3=components.get('o3'),
                temperature=main.get('temp'),
                humidity=main.get('humidity'),
                pressure=main.get('pressure'),
                wind_speed=wind.get('speed'),
                wind_direction=wind.get('deg'),
                latitude=latitude,
                longitude=longitude
            )
        except Exception:
            return None

    def get_aqi_data(self, city: str, latitude: float, longitude: float,
                     aqicn_key: Optional[str] = None,
                     openweather_key: Optional[str] = None,
                     airvisual_key: Optional[str] = None) -> Dict[str, Any]:
        cache_key = self._cache_key(city)
        data_sources = []

        if airvisual_key:
            self.logger.info(f"Fetching data from AirVisual for {city}")
            airvisual_data = self._get_airvisual_data(city, latitude, longitude, airvisual_key)
            if airvisual_data:
                data_sources.append(airvisual_data)
                self.logger.info("✓ AirVisual data collected")
            else:
                self.logger.warning("✗ AirVisual data collection failed")

        if aqicn_key:
            self.logger.info(f"Fetching data from AQICN for {city}")
            aqicn_data = self._get_aqicn_data(city, latitude, longitude, aqicn_key)
            if aqicn_data:
                data_sources.append(aqicn_data)
                self.logger.info("✓ AQICN data collected")
            else:
                self.logger.warning("✗ AQICN data collection failed")
                
        if openweather_key:
            self.logger.info(f"Fetching data from OpenWeather for {city}")
            openweather_data = self._get_openweather_data(city, latitude, longitude, openweather_key)
            if openweather_data:
                data_sources.append(openweather_data)
                self.logger.info("✓ OpenWeather data collected")
            else:
                self.logger.warning("✗ OpenWeather data collection failed")

        if not data_sources:
            with self.cache_lock:
                cached = self.cache.get(cache_key)
            if cached:
                self.logger.warning(f"Using cached data for {city}")
                return cached
            else:
                self.logger.error(f"No data collected for {city} and no cache available")
                return create_aqi_record(city, "none")

        merged_data = merge_aqi_data(data_sources)
        with self.cache_lock:
            self.cache[cache_key] = merged_data
            self._save_cache()
        self.logger.info(f"✓ Merged data for {city}")
        return merged_data
