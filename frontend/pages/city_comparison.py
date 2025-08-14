import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.api_client import fetch_api_data
from utils.helpers import get_aqi_class

def get_custom_css():
    return """
    <style>
    /* Main styling for all themes */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4);
        border: 2px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.2rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.4rem;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .city-card {
        text-align: center;
        padding: 2.5rem 2rem;
        border-radius: 25px;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
        border: 3px solid transparent;
        transition: all 0.4s ease;
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    .city-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: inherit;
        opacity: 0.9;
        z-index: -1;
    }
    
    .city-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0,0,0,0.25);
    }
    
    .city-name {
        font-size: 2.2rem;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .city-aqi {
        font-size: 4.5rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
        line-height: 1;
    }
    
    .city-status {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .city-details {
        font-size: 1.1rem;
        margin: 0.5rem 0;
        opacity: 0.9;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .city-trend {
        font-weight: bold;
        font-size: 1.2rem;
        margin-top: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
     .ranking-card {
         text-align: center;
         padding: 2.5rem 1.5rem;
         border-radius: 35px;
         margin: 1.5rem 0;
         box-shadow: 0 15px 35px rgba(0,0,0,0.15);
         border: 3px solid transparent;
         transition: all 0.4s ease;
         backdrop-filter: blur(20px);
         position: relative;
         overflow: hidden;
     }
    
    .ranking-card:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 25px 50px rgba(0,0,0,0.25);
    }
    
    .ranking-medal {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
        animation: bounce 2s infinite;
    }
    
    .ranking-title {
        font-size: 1.4rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .ranking-city {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .ranking-aqi {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .ranking-risk {
        font-size: 1.1rem;
        opacity: 0.9;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .section-header {
        text-align: center;
        margin: 4rem 0 2rem 0;
        padding: 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 20px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .section-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .section-subtitle {
        font-size: 1.3rem;
        color: #666;
        margin: 0;
        font-weight: 500;
    }
    
    .map-container {
        background: rgba(255,255,255,0.05);
        border-radius: 25px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 2px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(15px);
    }
    
    .error-state {
        text-align: center;
        padding: 4rem 2rem;
        border-radius: 25px;
        background: linear-gradient(135deg, rgba(255, 0, 0, 0.1), rgba(255, 100, 100, 0.05));
        border: 3px solid rgba(255, 0, 0, 0.3);
        margin: 3rem 0;
        box-shadow: 0 15px 35px rgba(255, 0, 0, 0.1);
        backdrop-filter: blur(15px);
    }
    
    .error-icon {
        font-size: 5rem;
        margin-bottom: 2rem;
        animation: pulse 2s infinite;
    }
    
    .error-title {
        color: #dc2626;
        margin-bottom: 1rem;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .error-message {
        color: #666;
        font-size: 1.3rem;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    
    .nav-button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem 2rem;
        border-radius: 50px;
        border: none;
        font-size: 1.1rem;
        font-weight: 600;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .nav-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.4);
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-10px); }
        60% { transform: translateY(-5px); }
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .float-animation {
        animation: float 3s ease-in-out infinite;
    }
    
    @media (prefers-color-scheme: dark) {
        .section-header {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
            border: 2px solid rgba(102, 126, 234, 0.3);
        }
        
        .section-title {
            color: #8b9bff;
        }
        
        .section-subtitle {
            color: #bbb;
        }
        
        .city-details {
            color: rgba(255,255,255,0.9);
        }
        
        .ranking-title {
            color: #ddd;
        }
        
        .ranking-risk {
            color: rgba(255,255,255,0.8);
        }
        
        .error-message {
            color: #bbb;
        }
        
        .map-container {
            background: rgba(0,0,0,0.1);
            border: 2px solid rgba(255,255,255,0.1);
        }
    }
    
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        
        .main-header p {
            font-size: 1.2rem;
        }
        
        .city-aqi {
            font-size: 3.5rem;
        }
        
        .city-name {
            font-size: 1.8rem;
        }
        
        .section-title {
            font-size: 2rem;
        }
        
        .ranking-medal {
            font-size: 3rem;
        }
        
        .ranking-aqi {
            font-size: 2rem;
        }
    }
    </style>
    """

def show():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header float-animation">
        <h1>🗺️ Multi-City Air Quality</h1>
        <p>Real-time air quality comparison across major Pakistani cities</p>
    </div>
    """, unsafe_allow_html=True)
    
    cities = ["Karachi", "Lahore", "Islamabad"]
    
    city_data = {}
    for city in cities:
        data = fetch_api_data(f"/dashboard/overview?location={city}")
        if data:
            city_data[city] = data
    
    if city_data:
        
        st.markdown("""
        <div class="section-header">
            <div class="section-title">🌆 Current Air Quality Status</div>
            <div class="section-subtitle">Live readings from major cities</div>
        </div>
        """, unsafe_allow_html=True)
        
        city_cols = st.columns(3)
        city_emojis = {"Karachi": "🏖️", "Lahore": "🏛️", "Islamabad": "🏔️"}
        
        for i, (city, data) in enumerate(city_data.items()):
            with city_cols[i]:
                aqi = data['current_aqi']
                aqi_class, risk_level = get_aqi_class(aqi)
                
                if aqi <= 50:
                    bg_gradient = "linear-gradient(135deg, #22c55e, #15803d)"
                    border_color = "#22c55e"
                    text_color = "white"
                elif aqi <= 100:
                    bg_gradient = "linear-gradient(135deg, #eab308, #a16207)"
                    border_color = "#eab308"
                    text_color = "white"
                elif aqi <= 150:
                    bg_gradient = "linear-gradient(135deg, #f97316, #c2410c)"
                    border_color = "#f97316"
                    text_color = "white"
                elif aqi <= 200:
                    bg_gradient = "linear-gradient(135deg, #ef4444, #dc2626)"
                    border_color = "#ef4444"
                    text_color = "white"
                elif aqi <= 300:
                    bg_gradient = "linear-gradient(135deg, #a855f7, #7c3aed)"
                    border_color = "#a855f7"
                    text_color = "white"
                else:
                    bg_gradient = "linear-gradient(135deg, #7f1d1d, #991b1b)"
                    border_color = "#7f1d1d"
                    text_color = "white"
                
                trend = data['trend_direction'].title()
                if trend == "Improving":
                    trend_emoji = "📈"
                    trend_color = "#22c55e"
                elif trend == "Worsening":
                    trend_emoji = "📉"
                    trend_color = "#ef4444"
                else:
                    trend_emoji = "➡️"
                    trend_color = "#eab308"
                
                st.markdown(f"""
                <div class="city-card" style="
                    background: {bg_gradient};
                    border-color: {border_color};
                    color: {text_color};
                ">
                    <div class="city-name">
                        {city_emojis[city]} {city}
                    </div>
                    <div class="city-aqi">{aqi:.0f}</div>
                    <div class="city-status">{risk_level}</div>
                    <div class="city-details">
                        📅 Weekly Avg: <strong>{data['weekly_avg']:.1f}</strong>
                    </div>
                    <div class="city-details">
                        📊 Quality Index: <strong>{aqi_class}</strong>
                    </div>
                    <div class="city-trend" style="color: {trend_color};">
                        {trend_emoji} {trend}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-header">
            <div class="section-title">🏆 Air Quality Rankings</div>
            <div class="section-subtitle">Cities ranked by air quality performance</div>
        </div>
        """, unsafe_allow_html=True)
        
        ranked_cities = sorted(city_data.items(), key=lambda x: x[1]['current_aqi'])
        
        rank_cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        rank_gradients = [
            "linear-gradient(135deg, #ffd700, #b8860b)",
            "linear-gradient(135deg, #c0c0c0, #808080)",
            "linear-gradient(135deg, #cd7f32, #8b4513)"
        ]
        rank_titles = ["🌟 Best Air Quality", "😊 Moderate Quality", "⚠️ Needs Attention"]
        
        for i, (city, data) in enumerate(ranked_cities):
            with rank_cols[i]:
                st.markdown(f"""
                <div class="ranking-card" style="
                    background: {rank_gradients[i]};
                    color: white;
                ">
                    <div class="ranking-medal">{medals[i]}</div>
                    <div class="ranking-title">{rank_titles[i]}</div>
                    <div class="ranking-city">
                        {city_emojis[city]} {city}
                    </div>
                    <div class="ranking-aqi">AQI: {data['current_aqi']:.0f}</div>
                    <div class="ranking-risk">{data['current_risk_level']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-header">
            <div class="section-title">🗺️ Interactive Air Quality Map</div>
            <div class="section-subtitle">Explore pollution levels across Pakistan</div>
        </div>
        """, unsafe_allow_html=True)
        
        city_coords = {
            "Karachi": [24.8607, 67.0011],
            "Lahore": [31.5204, 74.3587],
            "Islamabad": [33.6844, 73.0479]
        }
        
        map_data = []
        for city, data in city_data.items():
            coords = city_coords[city]
            map_data.append({
                'lat': coords[0],
                'lon': coords[1],
                'city': city,
                'aqi': data['current_aqi'],
                'risk_level': data['current_risk_level'],
                'weekly_avg': data['weekly_avg'],
                'trend': data['trend_direction']
            })
        
        df_map = pd.DataFrame(map_data)
        
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        
        fig = px.scatter_mapbox(
            df_map, lat="lat", lon="lon", 
            color="aqi", size="aqi",
            hover_name="city", 
            hover_data={
                "aqi": ":,.0f", 
                "risk_level": True, 
                "weekly_avg": ":,.1f",
                "trend": True,
                "lat": False, 
                "lon": False
            },
            color_continuous_scale="RdYlGn_r",
            size_max=50,
            mapbox_style="open-street-map",
            zoom=5, 
            center={"lat": 30, "lon": 70},
            height=650
        )
        
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="AQI Level",
                thickness=25,
                len=0.8,
                x=1.02,
                tickmode="array",
                tickvals=[0, 50, 100, 150, 200, 300],
                ticktext=["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous", "Extreme"],
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        for _, row in df_map.iterrows():
            fig.add_trace(
                go.Scattermapbox(
                    lat=[row['lat']],
                    lon=[row['lon']],
                    mode='markers+text',
                    marker=dict(size=20, color='white', symbol='circle'),
                    text=f"{city_emojis[row['city']]}",
                    textfont=dict(size=16, color='black'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-header">
            <div class="section-title">📊 Quick Statistics</div>
            <div class="section-subtitle">Key metrics at a glance</div>
        </div>
        """, unsafe_allow_html=True)
        
        stats_cols = st.columns(3)
        
        with stats_cols[0]:
            best_city = min(city_data.items(), key=lambda x: x[1]['current_aqi'])
            st.markdown(f"""
            <div class="city-card" style="
                background: linear-gradient(135deg, #22c55e, #15803d);
                color: white;
                padding: 2rem;
            ">
                <div style="font-size: 2.5rem;">{city_emojis[best_city[0]]}</div>
                <div style="font-size: 1.5rem; font-weight: bold; margin: 1rem 0;">{best_city[0]}</div>
                <div style="font-size: 1.2rem;">Best Quality</div>
                <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">AQI: {best_city[1]['current_aqi']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stats_cols[1]:
            worst_city = max(city_data.items(), key=lambda x: x[1]['current_aqi'])
            st.markdown(f"""
            <div class="city-card" style="
                background: linear-gradient(135deg, #ef4444, #dc2626);
                color: white;
                padding: 2rem;
            ">
                <div style="font-size: 2.5rem;">{city_emojis[worst_city[0]]}</div>
                <div style="font-size: 1.5rem; font-weight: bold; margin: 1rem 0;">{worst_city[0]}</div>
                <div style="font-size: 1.2rem;">Needs Attention</div>
                <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">AQI: {worst_city[1]['current_aqi']:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with stats_cols[2]:
            st.markdown(f"""
            <div class="city-card" style="
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                padding: 2rem;
            ">
                <div style="font-size: 2.5rem;">🏙️</div>
                <div style="font-size: 2rem; font-weight: bold; margin: 1rem 0;">{len(city_data)}</div>
                <div style="font-size: 1.2rem;">Cities Monitored</div>
                <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">Real-time Data</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0;">
            <div style="
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
                border-radius: 25px;
                padding: 3rem 2rem;
                border: 2px solid rgba(102, 126, 234, 0.2);
                backdrop-filter: blur(15px);
            ">
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="error-state">
            <div class="error-icon">❌</div>
            <h2 class="error-title">Data Unavailable</h2>
            <p class="error-message">
                We're unable to fetch city comparison data at the moment.<br>
                This could be due to network connectivity or API service issues.
            </p>
            <div style="margin-top: 2rem;">
                <div class="nav-button" onclick="window.location.reload()">
                    🔄 Refresh Page
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 2rem;
            margin: 3rem 0;
            border: 2px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(15px);
        ">
            <h3 style="color: #667eea; margin-bottom: 1.5rem;">🔧 Troubleshooting Steps</h3>
            <div style="color: #666; font-size: 1.1rem; line-height: 1.8;">
                <p><strong>1. Check Internet Connection</strong> - Ensure you have stable connectivity</p>
                <p><strong>2. Refresh the Page</strong> - Try reloading the application</p>
                <p><strong>3. API Status</strong> - Verify that API services are running</p>
                <p><strong>4. Contact Support</strong> - If the issue persists, please reach out for assistance</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    show()