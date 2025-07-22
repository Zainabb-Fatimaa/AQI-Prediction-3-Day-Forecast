import pandas as pd
import numpy as np
from datetime import datetime

CONVERSION_FACTORS = {
    'carbon_monoxide': 28.01 / 24.45,
    'nitrogen_dioxide': 46.01 / 24.45,
    'sulphur_dioxide': 64.07 / 24.45,
    'ozone': 48.00 / 24.45,
    'carbon_dioxide': 44.01 / 24.45
}

# EPA AQI breakpoints for PM2.5 and PM10 (μg/m³)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)
]
PM10_BREAKPOINTS = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400), (505, 604, 401, 500)
]

# EPA AQI breakpoints for all pollutants
AQI_BREAKPOINTS = {
    'PM2.5': PM25_BREAKPOINTS,
    'PM10': PM10_BREAKPOINTS,
    'CO': [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150), (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500)],
    'SO2': [(0.0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200), (305, 604, 201, 300), (605, 804, 301, 400), (805, 1004, 401, 500)],
    'O3': [(0.0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150), (86, 105, 151, 200), (106, 200, 201, 300)],
    'NO2': [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150), (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 1649, 301, 400), (1650, 2049, 401, 500)]
}

# Calculate AQI using EPA method
def calculate_epa_aqi(pollutant, concentration, unit):
    breakpoints = AQI_BREAKPOINTS.get(pollutant)
    if breakpoints is None or concentration is None or np.isnan(concentration):
        return np.nan
    # Unit checks
    if pollutant in ['O3', 'SO2', 'NO2'] and unit.lower() != 'ppb':
        return np.nan
    elif pollutant == 'CO' and unit.lower() != 'ppm':
        return np.nan
    elif pollutant in ['PM2.5', 'PM10'] and unit.lower() != 'ug/m3':
        return np.nan
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
    return np.nan

def aqi_to_ugm3(aqi, breakpoints):
    for c_low, c_high, i_low, i_high in breakpoints:
        if i_low <= aqi <= i_high:
            return ((aqi - i_low) * (c_high - c_low) / (i_high - i_low)) + c_low
    return None

def convert_gas_ugm3_to_ppb(value, gas):
    if gas in CONVERSION_FACTORS:
        return value / CONVERSION_FACTORS[gas]
    return value

def extract_temporal_features(row):
    # Accepts a dict with a 'timestamp' or 'date' field (datetime or string)
    ts = row.get('timestamp') or row.get('date')
    if ts is not None:
        if isinstance(ts, str):
            try:
                ts = pd.to_datetime(ts)
            except Exception:
                return row
        row['hour'] = ts.hour
        row['day'] = ts.day
        row['weekday'] = ts.weekday()
    return row

def standardize_row(row, source):
    # Only for Karachi
    if row.get('city', '').lower() != 'karachi':
        return row
    # AQICN: Convert PM2.5/PM10 AQI to μg/m³
    if source == 'aqicn':
        if 'pm25' in row:
            row['pm25_ugm3'] = aqi_to_ugm3(row['pm25'], PM25_BREAKPOINTS)
        if 'pm10' in row:
            row['pm10_ugm3'] = aqi_to_ugm3(row['pm10'], PM10_BREAKPOINTS)
        # Gaseous elements: pass through if already in ppb/ppm
        for gas in ['co', 'no2', 'so2', 'o3']:
            if gas in row:
                row[f'{gas}_ppb'] = row[gas]  # Assume already in ppb/ppm
    # OpenWeather: wind speed (m/s) to km/h, wind direction as-is
    if source == 'openweather':
        if 'wind_speed' in row:
            row['wind_speed_kmh'] = row['wind_speed'] * 3.6
        if 'wind_direction' in row:
            row['wind_direction_deg'] = row['wind_direction']
    # Open-Meteo: assume already in correct units
    # --- Calculate AQI for all pollutants if possible ---
    # PM2.5
    if 'pm25_ugm3' in row:
        row['aqi_pm25'] = calculate_epa_aqi('PM2.5', row['pm25_ugm3'], 'ug/m3')
    # PM10
    if 'pm10_ugm3' in row:
        row['aqi_pm10'] = calculate_epa_aqi('PM10', row['pm10_ugm3'], 'ug/m3')
    # CO (assume CO in ppm if present)
    if 'co_ppb' in row:
        row['aqi_co'] = calculate_epa_aqi('CO', row['co_ppb'] / 1000, 'ppm')
    # SO2
    if 'so2_ppb' in row:
        row['aqi_so2'] = calculate_epa_aqi('SO2', row['so2_ppb'], 'ppb')
    # O3
    if 'o3_ppb' in row:
        row['aqi_o3'] = calculate_epa_aqi('O3', row['o3_ppb'], 'ppb')
    # NO2
    if 'no2_ppb' in row:
        row['aqi_no2'] = calculate_epa_aqi('NO2', row['no2_ppb'], 'ppb')
    # Overall AQI
    aqi_fields = [row.get('aqi_pm25'), row.get('aqi_pm10'), row.get('aqi_co'), row.get('aqi_so2'), row.get('aqi_o3'), row.get('aqi_no2')]
    aqi_fields = [aqi for aqi in aqi_fields if aqi is not None and not np.isnan(aqi)]
    if aqi_fields:
        row['aqi_overall'] = max(aqi_fields)
    # --- Temporal features ---
    row = extract_temporal_features(row)
    return row 