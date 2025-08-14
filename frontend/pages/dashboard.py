import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils.api_client import fetch_api_data
from utils.helpers import get_aqi_class

def show():
    add_custom_css()
    
    st.markdown("# 🌍 Air Quality Monitor")
    st.markdown('<p class="subtitle">Real-time air quality insights for Karachi</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    aqi = 0
    current_data = None
    risk_level = ""
    aqi_class = ""
    
    current_data = fetch_api_data("/dashboard/overview?location=Karachi")
    if current_data:
        aqi = current_data['current_aqi']
        risk_level = current_data['current_risk_level']
        aqi_class, _ = get_aqi_class(aqi)
    
    if current_data:
        st.markdown('<div class="section-header">📍 Current Air Quality</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            if aqi <= 50:
                gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                shadow_color = "rgba(102, 126, 234, 0.4)"
                emoji = "🌟"
                status_text = "Excellent Air Quality"
                status_desc = "Perfect for all outdoor activities"
            elif aqi <= 100:
                gradient = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
                shadow_color = "rgba(240, 147, 251, 0.4)"
                emoji = "😊"
                status_text = "Good Air Quality"
                status_desc = "Safe for everyone"
            elif aqi <= 150:
                gradient = "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
                shadow_color = "rgba(79, 172, 254, 0.4)"
                emoji = "⚠️"
                status_text = "Moderate Air Quality"
                status_desc = "Sensitive groups be cautious"
            else:
                gradient = "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"
                shadow_color = "rgba(250, 112, 154, 0.4)"
                emoji = "🚨"
                status_text = "Unhealthy Air Quality"
                status_desc = "Limit outdoor exposure"
            
            st.markdown(f"""
            <div class="hero-card" style="
                background: {gradient};
                box-shadow: 0 20px 40px {shadow_color};
            ">
                <div class="aqi-number">{emoji} {aqi:.0f}</div>
                <div class="aqi-label">AQI Index</div>
                <div class="status-text">{status_text}</div>
                <div class="status-desc">{status_desc}</div>
                <div class="last-updated">🕒 Updated: {current_data['last_updated']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        stats_col, rec_col = st.columns([1, 1], gap="large")
        
        with stats_col:
            st.markdown('<div class="section-title">📊 Statistics</div>', unsafe_allow_html=True)
            
            weekly_avg = current_data['weekly_avg']
            trend = current_data['trend_direction'].title()
            
            if trend == "Improving":
                trend_gradient = "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)"
                trend_emoji = "📈"
                trend_color = "#10b981"
            elif trend == "Stable":
                trend_gradient = "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)"
                trend_emoji = "➡️"
                trend_color = "#f59e0b"
            else:
                trend_gradient = "linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%)"
                trend_emoji = "📉"
                trend_color = "#ef4444"
            
            st.markdown(f"""
            <div class="stats-card" style="background: {trend_gradient};">
                <div class="stats-header">
                    <span class="stats-label">7-Day Average</span>
                    <span class="trend-badge" style="color: {trend_color};">
                        {trend_emoji} {trend}
                    </span>
                </div>
                <div class="stats-value">{weekly_avg:.1f}</div>
                <div class="stats-desc">Average AQI this week</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if aqi <= 50:
                indicator_style = "background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); border-left: 4px solid #10b981;"
                st.markdown(f'<div class="quality-indicator" style="{indicator_style}">🌟 <strong>Excellent!</strong><br>Perfect for all outdoor activities</div>', unsafe_allow_html=True)
            elif aqi <= 100:
                indicator_style = "background: linear-gradient(135deg, #a8caba 0%, #5d4e75 100%); border-left: 4px solid #3b82f6;"
                st.markdown(f'<div class="quality-indicator" style="{indicator_style}">😊 <strong>Good</strong><br>Safe for most people</div>', unsafe_allow_html=True)
            elif aqi <= 150:
                indicator_style = "background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); border-left: 4px solid #f59e0b;"
                st.markdown(f'<div class="quality-indicator" style="{indicator_style}">⚠️ <strong>Moderate</strong><br>Sensitive groups be cautious</div>', unsafe_allow_html=True)
            else:
                indicator_style = "background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); border-left: 4px solid #ef4444;"
                st.markdown(f'<div class="quality-indicator" style="{indicator_style}">🚨 <strong>Unhealthy</strong><br>Limit outdoor exposure</div>', unsafe_allow_html=True)
        
        with rec_col:
            st.markdown('<div class="section-title">💡 Recommendations</div>', unsafe_allow_html=True)
            
            recommendations = []
            
            if aqi <= 50:
                recommendations = [
                    ("🏃‍♂️", "Exercise", "Perfect for outdoor running and sports"),
                    ("🚪", "Windows", "Open windows for fresh air circulation"),
                    ("👶", "Children", "Safe for extended outdoor play"),
                    ("🌱", "Activities", "Ideal for gardening and outdoor work")
                ]
            elif aqi <= 100:
                recommendations = [
                    ("🚶‍♀️", "Exercise", "Good for walking and light activities"),
                    ("🏠", "Indoor", "Consider indoor activities for sensitive people"),
                    ("👥", "General", "Safe for most healthy adults"),
                    ("⏰", "Timing", "Best outdoor time is early morning")
                ]
            elif aqi <= 150:
                recommendations = [
                    ("⚠️", "Exercise", "Limit intense outdoor activities"),
                    ("😷", "Masks", "Consider N95 masks when outside"),
                    ("👶", "Sensitive Groups", "Children and elderly stay indoors"),
                    ("🏠", "Windows", "Keep windows closed")
                ]
            else:
                recommendations = [
                    ("🚨", "Stay Indoors", "Avoid outdoor activities"),
                    ("😷", "Masks", "Essential when going outside"),
                    ("🌱", "Air Purifiers", "Use HEPA filters at home"),
                    ("🏥", "Health", "Monitor respiratory symptoms")
                ]
            
            for emoji, title, desc in recommendations:
                st.markdown(f"""
                <div class="recommendation-item">
                    <div class="rec-emoji">{emoji}</div>
                    <div class="rec-content">
                        <div class="rec-title">{title}</div>
                        <div class="rec-desc">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">📈 Weekly Trend Analysis</div>', unsafe_allow_html=True)
        
        hist_data = fetch_api_data("/historical/Karachi?days=7")
        if hist_data:
            df = pd.DataFrame(hist_data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['aqi'],
                mode='lines+markers',
                name='AQI',
                line=dict(color='#667eea', width=4),
                marker=dict(size=10, color='#667eea', line=dict(width=2, color='white')),
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.2)'
            ))
            
            fig.add_hline(y=50, line_dash="dot", line_color="#10b981", opacity=0.6, 
                         annotation_text="Good", annotation_position="right")
            fig.add_hline(y=100, line_dash="dot", line_color="#f59e0b", opacity=0.6, 
                         annotation_text="Moderate", annotation_position="right")
            fig.add_hline(y=150, line_dash="dot", line_color="#ef4444", opacity=0.6, 
                         annotation_text="Unhealthy", annotation_position="right")
            
            fig.update_layout(
                title="",
                xaxis_title="",
                yaxis_title="AQI Value",
                height=450,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                xaxis=dict(
                    gridcolor='rgba(148, 163, 184, 0.2)',
                    showgrid=True,
                    linecolor='rgba(148, 163, 184, 0.3)'
                ),
                yaxis=dict(
                    gridcolor='rgba(148, 163, 184, 0.2)',
                    showgrid=True,
                    linecolor='rgba(148, 163, 184, 0.3)'
                ),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="info-message">📊 Historical data temporarily unavailable</div>', unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">🔮 3-Day Forecast</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-subtitle">Plan your activities ahead</p>', unsafe_allow_html=True)
        
        # Fetch forecast data using the /forecast endpoint
        forecast_data = fetch_api_data("/forecast")
        
        if forecast_data and forecast_data.get('forecasts'):
            forecasts = forecast_data['forecasts']
            
            # Create mapping for horizon hours to day labels
            day_labels = {
                24: "Today",
                48: "Tomorrow", 
                72: "Day After Tomorrow"
            }
            
            # Get the forecast generation timestamp
            forecast_timestamp = forecast_data.get('forecast_generated_at', '')
            
            forecast_cols = st.columns(3, gap="medium")
            gradients = [
                "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
                "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
            ]
            
            # Sort forecasts by horizon_hours to ensure consistent order
            sorted_forecasts = sorted(forecasts, key=lambda x: x['horizon_hours'])
            
            for i, forecast in enumerate(sorted_forecasts[:3]):
                horizon_hours = forecast['horizon_hours']
                predicted_aqi = forecast['predicted_aqi']
                risk_level = forecast['risk_level']
                
                # Get the day label
                day_label = day_labels.get(horizon_hours, f"{horizon_hours}h")
                
                # Calculate the forecast date based on current time + horizon hours
                from datetime import datetime, timedelta
                forecast_date = (datetime.now() + timedelta(hours=horizon_hours)).strftime("%Y-%m-%d")
                
                if predicted_aqi <= 50:
                    emoji = "🌟"
                    status_color = "#10b981"
                elif predicted_aqi <= 100:
                    emoji = "😊"
                    status_color = "#3b82f6"
                elif predicted_aqi <= 150:
                    emoji = "⚠️"
                    status_color = "#f59e0b"
                else:
                    emoji = "🚨"
                    status_color = "#ef4444"
                
                with forecast_cols[i]:
                    st.markdown(f"""
                    <div class="forecast-card" style="background: {gradients[i]};">
                        <div class="forecast-day">{day_label}</div>
                        <div class="forecast-date">{forecast_date}</div>
                        <div class="forecast-aqi">{emoji} {predicted_aqi:.0f}</div>
                        <div class="forecast-status" style="color: {status_color};">
                            {risk_level}
                        </div>
                        <div class="forecast-horizon">({horizon_hours}h forecast)</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-message">🔮 Forecast data temporarily unavailable</div>', unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="error-card">
            <div class="error-icon">❌</div>
            <div class="error-title">Unable to fetch air quality data</div>
            <div class="error-desc">
                Please check your connection and try refreshing the page.<br>
                If the problem persists, contact our support team.
            </div>
            <div class="error-actions">
                <button onclick="window.location.reload()" class="retry-button">🔄 Retry</button>
            </div>
        </div>
        """, unsafe_allow_html=True)

def add_custom_css():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        /* Custom Typography */
        .subtitle {
            font-size: 1.2rem;
            color: #94a3b8;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .section-header {
            font-size: 2rem;
            font-weight: 600;
            color: #e2e8f0;
            text-align: center;
            margin: 2rem 0 1rem 0;
        }
        
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .section-subtitle {
            color: #94a3b8;
            margin-bottom: 1.5rem;
            font-size: 0.95rem;
            text-align: center;
        }
        
        /* Hero Card */
        .hero-card {
            text-align: center;
            padding: 3rem 2rem;
            border-radius: 24px;
            margin: 2rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transform: translateY(0);
            transition: all 0.3s ease;
        }
        
        .hero-card:hover {
            transform: translateY(-5px);
        }
        
        .aqi-number {
            font-size: 4.5rem;
            font-weight: 700;
            color: white;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .aqi-label {
            font-size: 1.1rem;
            color: rgba(255, 255, 255, 0.9);
            margin: 0.5rem 0;
            font-weight: 500;
        }
        
        .status-text {
            font-size: 1.3rem;
            font-weight: 600;
            color: white;
            margin: 1rem 0 0.5rem 0;
        }
        
        .status-desc {
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 1rem;
        }
        
        .last-updated {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.7);
            margin-top: 1.5rem;
        }
        
        /* Stats Card */
        .stats-card {
            padding: 2rem;
            border-radius: 20px;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .stats-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
        
        .stats-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .stats-label {
            font-size: 1rem;
            font-weight: 500;
            color: #1e293b;
        }
        
        .trend-badge {
            font-size: 0.9rem;
            font-weight: 600;
            padding: 0.25rem 0.5rem;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }
        
        .stats-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1e293b;
            margin: 0.5rem 0;
        }
        
        .stats-desc {
            font-size: 0.9rem;
            color: #475569;
            font-weight: 500;
        }
        
        /* Quality Indicator */
        .quality-indicator {
            padding: 1.5rem;
            border-radius: 16px;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #1e293b;
            font-size: 1rem;
            line-height: 1.5;
            transition: all 0.3s ease;
        }
        
        .quality-indicator:hover {
            transform: translateY(-2px);
        }
        
        /* Recommendations */
        .recommendation-item {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            padding: 1rem 1.5rem;
            background: rgba(30, 41, 59, 0.4);
            border-radius: 16px;
            margin: 0.8rem 0;
            border: 1px solid rgba(148, 163, 184, 0.2);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .recommendation-item:hover {
            background: rgba(30, 41, 59, 0.6);
            transform: translateX(5px);
        }
        
        .rec-emoji {
            font-size: 1.5rem;
            flex-shrink: 0;
        }
        
        .rec-content {
            flex: 1;
        }
        
        .rec-title {
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 0.25rem;
        }
        
        .rec-desc {
            font-size: 0.9rem;
            color: #94a3b8;
            line-height: 1.4;
        }
        
        /* Forecast Cards */
        .forecast-card {
            text-align: center;
            padding: 2rem 1.5rem;
            border-radius: 20px;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
            height: 280px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .forecast-card:hover {
            transform: translateY(-5px) scale(1.02);
        }
        
        .forecast-day {
            font-size: 1.2rem;
            font-weight: 600;
            color: white;
            margin-bottom: 0.5rem;
        }
        
        .forecast-date {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 1.5rem;
        }
        
        .forecast-aqi {
            font-size: 3rem;
            font-weight: 700;
            color: white;
            margin: 1rem 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .forecast-status {
            font-size: 1rem;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.2);
            padding: 0.5rem 1rem;
            border-radius: 12px;
            backdrop-filter: blur(10px);
            margin-bottom: 0.5rem;
        }
        
        .forecast-horizon {
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.7);
            font-style: italic;
        }
        
        /* Messages */
        .info-message {
            text-align: center;
            padding: 2rem;
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 16px;
            color: #93c5fd;
            font-size: 1.1rem;
            margin: 2rem 0;
        }
        
        /* Error Card */
        .error-card {
            text-align: center;
            padding: 3rem 2rem;
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
            border: 2px solid rgba(239, 68, 68, 0.3);
            border-radius: 24px;
            margin: 2rem 0;
            backdrop-filter: blur(10px);
        }
        
        .error-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
        
        .error-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #fca5a5;
            margin-bottom: 1rem;
        }
        
        .error-desc {
            color: #f87171;
            margin-bottom: 2rem;
            line-height: 1.6;
        }
        
        .retry-button {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .retry-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(239, 68, 68, 0.4);
        }
        
        /* Remove Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .hero-card {
                padding: 2rem 1rem;
            }
            
            .aqi-number {
                font-size: 3.5rem;
            }
            
            .forecast-card {
                height: 250px;
                padding: 1.5rem 1rem;
            }
            
            .forecast-aqi {
                font-size: 2.5rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    show()