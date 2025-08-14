import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

API_URL = os.getenv("BACKEND_URL", "http://backend:8000") + "/forecast/hourly" 
# API_URL = "http://localhost:8000/forecast/hourly" 

def get_hourly_forecast():
    try:
        resp = requests.get(API_URL)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API error: {resp.status_code}")
            return None
    except Exception as e:
        st.error(f"Failed to fetch forecast: {e}")
        return None

def aqi_recommendation(aqi):
    if aqi <= 50:
        return ("🌟 Excellent", "Enjoy outdoor activities! Air quality is good.", "#22c55e", "#15803d")
    elif aqi <= 100:
        return ("😊 Moderate", "Safe for most, but sensitive groups should monitor symptoms.", "#eab308", "#a16207")
    elif aqi <= 150:
        return ("⚠️ Unhealthy for Sensitive Groups", "Limit prolonged outdoor exertion, especially for children, elderly, and those with respiratory issues.", "#f97316", "#c2410c")
    elif aqi <= 200:
        return ("🚨 Unhealthy", "Everyone should reduce outdoor activities. Sensitive groups should avoid them.", "#ef4444", "#dc2626")
    elif aqi <= 300:
        return ("☠️ Very Unhealthy", "Health warnings of emergency conditions. Avoid all outdoor exertion.", "#a855f7", "#7c3aed")
    else:
        return ("💀 Hazardous", "Serious health effects. Remain indoors and keep windows closed.", "#7f1d1d", "#991b1b")

def get_custom_css():
    return """
    <style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.3rem;
        margin: 0;
    }
    
    /* AQI Card Styling */
    .aqi-card {
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 2px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .aqi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    .aqi-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .aqi-status {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0.5rem 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .aqi-time {
        font-size: 0.9rem;
        opacity: 0.8;
        font-weight: 500;
    }
    
    .alert-box {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    .alert-box h3 {
        margin-top: 0;
        font-size: 1.8rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .alert-aqi {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .recommendations-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    .recommendations-box h4 {
        margin-top: 0;
        font-size: 1.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .recommendations-box ul {
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    .recommendations-box li {
        margin-bottom: 0.8rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .chart-container {
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    .chart-title {
        text-align: center;
        font-size: 1.8rem;
        margin-bottom: 1rem;
        color: #333;
        font-weight: 600;
    }
    
    @media (prefers-color-scheme: dark) {
        .chart-title {
            color: #fff;
        }
        
        .aqi-card {
            border: 2px solid rgba(255,255,255,0.1);
        }
    }
    
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .aqi-number {
            font-size: 2rem;
        }
        
        .aqi-status {
            font-size: 1rem;
        }
    }
    </style>
    """

def show():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Hourly AQI Prediction</h1>
        <p>See the next 72 hours of air quality for Karachi, with actionable health tips</p>
    </div>
    """, unsafe_allow_html=True)
    
    data = get_hourly_forecast()
    if not data or 'forecasts' not in data:
        st.error("No forecast data available.")
        return
    
    df = pd.DataFrame(data['forecasts'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">📊 Hourly AQI Forecast (Next 72 Hours)</div>
    </div>
    """, unsafe_allow_html=True)
    
    fig = px.line(df, x='timestamp', y='predicted_aqi', markers=True, 
                  title='', 
                  labels={'timestamp':'Time', 'predicted_aqi':'Predicted AQI'},
                  color_discrete_sequence=['#667eea'])
    
    fig.update_traces(mode='lines+markers', 
                      marker=dict(size=8, color='#667eea'),
                      line=dict(width=3))
    
    fig.update_layout(
        height=450,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.3)',
            title_font=dict(size=14, color='#667eea')
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.3)',
            title_font=dict(size=14, color='#667eea')
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <h3 style="font-size: 1.8rem; color: #667eea; margin-bottom: 2rem;">🕐 Current & Next 6 Hours</h3>
    </div>
    """, unsafe_allow_html=True)
    
    next6 = df.head(6)
    cols = st.columns(6)
    
    for i, (idx, row) in enumerate(next6.iterrows()):
        label, rec, bg_color, text_color = aqi_recommendation(row['predicted_aqi'])
        
        with cols[i]:
            st.markdown(f"""
            <div class="aqi-card" style="background: linear-gradient(135deg, {bg_color}, {text_color}); color: white;">
                <div class="aqi-number">{int(row['predicted_aqi'])}</div>
                <div class="aqi-status">{label}</div>
                <div class="aqi-time">{row['timestamp'].strftime('%H:%M')}</div>
            </div>
            """, unsafe_allow_html=True)
    
    worst = df.loc[df['predicted_aqi'].idxmax()]
    label, rec, bg_color, text_color = aqi_recommendation(worst['predicted_aqi'])
    
    st.markdown(f"""
    <div class="alert-box">
        <h3>⚠️ Highest Risk Hour</h3>
        <div class="alert-aqi">{int(worst['predicted_aqi'])} ({label})</div>
        <p style='font-size: 1.2rem; margin: 1rem 0;'>{worst['timestamp'].strftime('%A %H:%M')}</p>
        <p style='font-size: 1.1rem; line-height: 1.5;'>{rec}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="recommendations-box">
        <h4>💡 General AQI Health Recommendations</h4>
        <ul>
            <li>🕒 Check hourly AQI before planning outdoor activities</li>
            <li>🏃‍♂️ On moderate/unhealthy hours, reduce exertion and time outside</li>
            <li>🏠 Keep windows closed and use air purifiers if AQI is high</li>
            <li>😷 Wear a mask if you must go outside during poor air quality</li>
            <li>🫁 Monitor symptoms if you have asthma or heart/lung conditions</li>
            <li>🚶‍♀️ Consider indoor exercises when air quality is poor</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show()