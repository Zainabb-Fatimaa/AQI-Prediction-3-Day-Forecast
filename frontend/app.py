import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from pages import dashboard, prediction, city_comparison, analytics
from utils.alerts import (
    show_alerts, 
    add_success_alert, 
    add_warning_alert, 
    add_error_alert, 
    add_info_alert,
    check_and_alert_hazardous_aqi,  
    add_hazardous_aqi_alert
)

st.set_page_config(page_title="AirLens", page_icon="🌬️", layout="wide")

if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'alerts_initialized' not in st.session_state:
    st.session_state.alerts_initialized = False
if 'current_aqi' not in st.session_state:
    st.session_state.current_aqi = None
if 'is_hazardous_conditions' not in st.session_state:
    st.session_state.is_hazardous_conditions = False
if 'last_hazard_check' not in st.session_state:
    st.session_state.last_hazard_check = 0

API_BASE_URL = os.getenv("BACKEND_URL", "http://backend:8000")
# API_BASE_URL = "http://localhost:8000"
try:
    current_aqi, is_hazardous = check_and_alert_hazardous_aqi(API_BASE_URL)
    if current_aqi is not None:
        st.session_state.current_aqi = current_aqi
        st.session_state.is_hazardous_conditions = is_hazardous
        
except Exception as e:
    st.error(f"Failed to check hazardous conditions: {str(e)}")

show_alerts()

if not st.session_state.alerts_initialized:
    add_info_alert("Welcome!", "Welcome to AirLens - Your Smart Air Quality Companion", 10)
    st.session_state.alerts_initialized = True

import time
current_time = time.time()
if current_time - st.session_state.last_hazard_check > 300:  
    try:
        check_and_alert_hazardous_aqi(API_BASE_URL)
        st.session_state.last_hazard_check = current_time
    except Exception as e:
        pass  
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Header styles */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: fadeInUp 1s ease-out;
    }
    
    .sub-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        font-weight: 400;
        color: rgba(255,255,255,0.9);
        margin-bottom: 1.5rem;
        animation: fadeInUp 1s ease-out 0.2s both;
    }
    
    /* Enhanced info section with hazard indicator */
    .info-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
    }
    
    .info-container.hazardous {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid #ef4444;
        box-shadow: 0 5px 15px rgba(239, 68, 68, 0.2);
    }
    
    .info-text {
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        line-height: 1.7;
        color: #2c3e50;
        margin: 0;
    }
    
    .highlight-text {
        color: #667eea;
        font-weight: 600;
    }
    
    .hazard-text {
        color: #dc2626;
        font-weight: 700;
    }
    
    /* Rest of your existing CSS styles... */
    .button-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .features-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Custom button styling */
    .stButton > button {
        font-family: 'Poppins', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 12px !important;
        border: none !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        margin-bottom: 0.8rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin: 3rem 0 2rem 0;
        padding: 1.5rem;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(240, 147, 251, 0.3);
    }
    
    .footer-text {
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: white;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        .sub-title {
            font-size: 1.2rem;
        }
        .info-container {
            padding: 1.5rem;
        }
        .button-container {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 class="main-title">🌬️ AirLens</h1>
    <h2 class="sub-title">Your Smart Air Quality Companion for Pakistan</h2>
</div>
""", unsafe_allow_html=True)

hazardous_class = "hazardous" if st.session_state.get('is_hazardous_conditions', False) else ""
hazard_text = "hazard-text" if st.session_state.get('is_hazardous_conditions', False) else "highlight-text"

st.markdown(f"""
<div class="info-container {hazardous_class}">
    <p class="info-text">
        <span class="{hazard_text}">Why does AQI matter?</span><br><br>
        The Air Quality Index (AQI) is a vital measure of the air we breathe. Poor air quality can cause 
        <strong>headaches, fatigue, breathing problems</strong>, and even serious <strong>heart and lung diseases</strong>. 
        Children, the elderly, and those with health conditions are especially at risk.<br></p
        {f'<br><strong class="hazard-text">🚨 Current conditions may be hazardous to your health!</strong>' if st.session_state.get('is_hazardous_conditions', False) else ''}
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h3 class="features-title">✨ Explore Features</h3>', unsafe_allow_html=True)
    
    btn_dashboard = st.button("🏠 Real-time Dashboard", use_container_width=True)
    btn_prediction = st.button("🔮 AI Hourly Prediction", use_container_width=True)
    btn_city = st.button("🗺️ Multi-City Comparison", use_container_width=True)
    btn_analytics = st.button("📊 Detailed Analytics", use_container_width=True)

if btn_dashboard:
    dashboard.show()
elif btn_prediction:
    prediction.show()
elif btn_city:
    city_comparison.show()
elif btn_analytics:
    analytics.show()

# Footer
st.markdown("""
<div class="footer">
    <p class="footer-text">
        🌟 Stay informed. Breathe easy. Protect your health. 🌟
    </p>
</div>
""", unsafe_allow_html=True)