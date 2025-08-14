import requests
import streamlit as st
import os

API_BASE_URL = os.getenv("BACKEND_URL", "http://backend:8000")
# API_BASE_URL =  "http://backend:8000"

def fetch_api_data(endpoint):
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to API. Please ensure the FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def get_current_aqi(city="Karachi"):
    data = fetch_api_data(f"/dashboard/overview?location={city}")
    return data['current_aqi'] if data else None

def get_forecast_data(horizon="Next 3 days"):
    return fetch_api_data(f"/forecast/hourly?horizon={horizon}")

def get_historical_data(city, days=7):
    return fetch_api_data(f"/historical/{city}?days={days}")

def get_all_locations():
    return fetch_api_data("/locations")