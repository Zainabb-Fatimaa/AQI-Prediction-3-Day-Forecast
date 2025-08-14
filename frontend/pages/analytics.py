import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.api_client import fetch_api_data
from utils.helpers import get_aqi_class

def get_analytics_css():
    return """
    <style>
    .analytics-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4);
        border: 2px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    .analytics-header h1 {
        color: white;
        font-size: 3.2rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-weight: 700;
    }
    
    .analytics-header p {
        color: rgba(255,255,255,0.95);
        font-size: 1.4rem;
        margin: 0;
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
    
    .chart-container {
        background: rgba(255,255,255,0.05);
        border-radius: 25px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 2px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(15px);
    }
    
    .stats-card {
        text-align: center;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border: 2px solid transparent;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .period-selector {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 2rem 0;
        border: 2px solid rgba(102, 126, 234, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .export-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border-radius: 20px;
        padding: 2rem;
        margin: 3rem 0;
        border: 2px solid rgba(102, 126, 234, 0.2);
        backdrop-filter: blur(15px);
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
        
        .chart-container {
            background: rgba(0,0,0,0.1);
            border: 2px solid rgba(255,255,255,0.1);
        }
        
        .period-selector {
            background: rgba(0,0,0,0.1);
            border: 2px solid rgba(102, 126, 234, 0.3);
        }
        
        .export-section {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
            border: 2px solid rgba(102, 126, 234, 0.3);
        }
    }
    
    @media (max-width: 768px) {
        .analytics-header h1 {
            font-size: 2.5rem;
        }
        
        .analytics-header p {
            font-size: 1.2rem;
        }
        
        .section-title {
            font-size: 2rem;
        }
        
        .chart-container {
            padding: 1rem;
        }
    }
    </style>
    """

def show():
    st.markdown(get_analytics_css(), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="analytics-header">
        <h1>📊 Detailed Analytics</h1>
        <p>Comprehensive air quality analysis and trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    cities = ["Karachi", "Lahore", "Islamabad"]
    
    st.markdown('<div class="period-selector">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        days = st.selectbox("📅 Time Period", [7, 14, 21], index=1, format_func=lambda x: f"{x} days")
    with col2:
        chart_type = st.selectbox("📈 Chart Type", ["Line Chart", "Bar Chart", "Area Chart", "Summary Bars"], index=0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    historical_data = {}
    for city in cities:
        hist_data = fetch_api_data(f"/historical/{city}?days={days}")
        if hist_data:
            historical_data[city] = hist_data['data']
    
    if not historical_data:
        st.error("No historical data available for the selected time period.")
        return
    
    for city in historical_data:
        historical_data[city] = pd.DataFrame(historical_data[city])
        historical_data[city]['timestamp'] = pd.to_datetime(historical_data[city]['timestamp'])
    
    st.markdown("""
    <div class="section-header">
        <div class="section-title">📈 Historical Trends Comparison</div>
        <div class="section-subtitle">AQI patterns over the selected time period</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    fig = go.Figure()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    color_rgba = ['255,107,107', '78,205,196', '69,183,209']
    city_emojis = {"Karachi": "🏖️", "Lahore": "🏛️", "Islamabad": "🏔️"}
    
    if chart_type == "Summary Bars":
        summary_data = []
        for city, df in historical_data.items():
            summary_data.append({
                'City': f"{city_emojis[city]} {city}",
                'Average AQI': df['aqi'].mean(),
                'Max AQI': df['aqi'].max(),
                'Min AQI': df['aqi'].min(),
                'Days > 100': (df['aqi'] > 100).sum(),
                'Days > 150': (df['aqi'] > 150).sum()
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        fig.add_trace(go.Bar(
            x=summary_df['City'],
            y=summary_df['Average AQI'],
            name='Average AQI',
            marker_color='#667eea',
            text=summary_df['Average AQI'].round(1),
            textposition='outside'
        ))
        
    else:
        for i, (city, df) in enumerate(historical_data.items()):
            if chart_type == "Line Chart":
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['aqi'],
                    mode='lines+markers',
                    name=f"{city_emojis[city]} {city}",
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=6, symbol='circle')
                ))
            elif chart_type == "Bar Chart":
                daily_avg = df.groupby(df['timestamp'].dt.date)['aqi'].mean().reset_index()
                daily_avg['timestamp'] = pd.to_datetime(daily_avg['timestamp'])
                fig.add_trace(go.Bar(
                    x=daily_avg['timestamp'],
                    y=daily_avg['aqi'],
                    name=f"{city_emojis[city]} {city}",
                    marker_color=colors[i],
                    opacity=0.8
                ))
            elif chart_type == "Area Chart":
                fill_mode = 'tozeroy' if i == 0 else 'tonexty'
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['aqi'],
                    mode='lines',
                    name=f"{city_emojis[city]} {city}",
                    line=dict(color=colors[i], width=2),
                    fill=fill_mode,
                    fillcolor=f'rgba({color_rgba[i]}, 0.3)'
                ))
    
    fig.add_hline(y=50, line_dash="dash", line_color="green", opacity=0.3, 
                  annotation_text="Good", annotation_position="right")
    fig.add_hline(y=100, line_dash="dash", line_color="yellow", opacity=0.3,
                  annotation_text="Moderate", annotation_position="right")
    fig.add_hline(y=150, line_dash="dash", line_color="orange", opacity=0.3,
                  annotation_text="Unhealthy", annotation_position="right")
    
    fig.update_layout(
        title="",
        xaxis_title="Date",
        yaxis_title="AQI Value",
        height=500,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(128,128,128,0.2)',
            showgrid=True
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <div class="section-title">📊 Statistical Summary</div>
        <div class="section-subtitle">Key metrics and performance indicators</div>
    </div>
    """, unsafe_allow_html=True)
    
    stats_data = []
    for city, df in historical_data.items():
        stats_data.append({
            'City': f"{city_emojis[city]} {city}",
            'Mean AQI': f"{df['aqi'].mean():.1f}",
            'Max AQI': f"{df['aqi'].max():.0f}",
            'Min AQI': f"{df['aqi'].min():.0f}",
            'Standard Deviation': f"{df['aqi'].std():.1f}",
            'Days > 100 AQI': f"{(df['aqi'] > 100).sum()}",
            'Days > 150 AQI': f"{(df['aqi'] > 150).sum()}",
            'Improvement Days': f"{(df['aqi'].diff() < 0).sum()}",
            'Worsening Days': f"{(df['aqi'].diff() > 0).sum()}"
        })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True)
    
    st.markdown("""
    <div class="section-header">
        <div class="section-title">🏆 Performance Comparison</div>
        <div class="section-subtitle">Visual comparison of city performance</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        avg_aqi_data = [{'City': city, 'Average AQI': df['aqi'].mean()} 
                        for city, df in historical_data.items()]
        avg_df = pd.DataFrame(avg_aqi_data)
        
        fig_avg = px.bar(avg_df, x='City', y='Average AQI', 
                         color='Average AQI',
                         color_continuous_scale='RdYlGn_r',
                         title="Average AQI by City")
        fig_avg.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_avg, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        threshold_data = []
        for city, df in historical_data.items():
            threshold_data.append({
                'City': city,
                'Days > 100': (df['aqi'] > 100).sum(),
                'Days > 150': (df['aqi'] > 150).sum()
            })
        
        threshold_df = pd.DataFrame(threshold_data)
        fig_threshold = px.bar(threshold_df, x='City', y=['Days > 100', 'Days > 150'],
                              title="Days Above Threshold",
                              barmode='group')
        fig_threshold.update_layout(height=400)
        st.plotly_chart(fig_threshold, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <div class="section-title">📈 Trend Analysis</div>
        <div class="section-subtitle">Direction and magnitude of air quality changes</div>
    </div>
    """, unsafe_allow_html=True)
    
    trend_cols = st.columns(len(cities))
    
    for i, (city, df) in enumerate(historical_data.items()):
        with trend_cols[i]:
            st.markdown(f"""
            <div class="stats-card" style="
                background: linear-gradient(135deg, {colors[i]}, {colors[i]}dd);
                color: white;
            ">
                <h3 style="font-size: 1.5rem; margin-bottom: 1rem;">{city_emojis[city]} {city}</h3>
                <div style="font-size: 2rem; font-weight: bold; margin: 1rem 0;">
                    {df['aqi'].iloc[-1]:.0f}
                </div>
                <div style="font-size: 1.1rem; margin-bottom: 0.5rem;">Current AQI</div>
                <div style="font-size: 1.2rem; font-weight: bold; margin: 1rem 0;">
                    {df['aqi'].diff().mean():+.1f}
                </div>
                <div style="font-size: 1rem;">Daily Change</div>
                <div style="font-size: 1.1rem; margin-top: 1rem;">
                    📈 {(df['aqi'].diff() > 0).sum()} days worsening<br>
                    📉 {(df['aqi'].diff() < 0).sum()} days improving
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="export-section">
        <h3 style="color: #667eea; margin-bottom: 1.5rem;">💾 Export Analytics Data</h3>
        <p style="color: #666; margin-bottom: 2rem;">
            Download comprehensive analytics reports and data for further analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate Analytics Report", type="primary"):
            report = f"""# 📊 Air Quality Analytics Report
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Time Period:** {days} days

## 📈 Summary Statistics

"""
            for city, df in historical_data.items():
                report += f"""
### {city_emojis[city]} {city}
- **Average AQI:** {df['aqi'].mean():.1f}
- **Maximum AQI:** {df['aqi'].max():.0f}
- **Minimum AQI:** {df['aqi'].min():.0f}
- **Standard Deviation:** {df['aqi'].std():.1f}
- **Days > 100 AQI:** {(df['aqi'] > 100).sum()}
- **Days > 150 AQI:** {(df['aqi'] > 150).sum()}
- **Improving Days:** {(df['aqi'].diff() < 0).sum()}
- **Worsening Days:** {(df['aqi'].diff() > 0).sum()}

"""
            
            report += f"""
## 📊 Key Insights
- Best performing city: {min(historical_data.items(), key=lambda x: x[1]['aqi'].mean())[0]}
- Worst performing city: {max(historical_data.items(), key=lambda x: x[1]['aqi'].mean())[0]}
- Total days analyzed: {days}
- Overall trend analysis completed

---
*Report generated by AQI Guardian Analytics System*
"""
            
            st.download_button(
                "📄 Download Report (Markdown)",
                report,
                f"aqi_analytics_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.md",
                "text/markdown"
            )
    
    with col2:
        if st.button("📋 Export Raw Data"):
            all_data = []
            for city, df in historical_data.items():
                df_copy = df.copy()
                df_copy['City'] = city
                all_data.append(df_copy)
            
            combined_df = pd.concat(all_data, ignore_index=True)
            
            csv_data = combined_df.to_csv(index=False)
            st.download_button(
                "📊 Download CSV Data",
                csv_data,
                f"aqi_data_{days}days_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )

if __name__ == "__main__":
    show()