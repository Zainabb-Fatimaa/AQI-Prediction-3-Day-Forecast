import streamlit as st
from datetime import datetime, timedelta
import json
import requests
import logging
from typing import Dict, List, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self):
        self.alerts = []
        self.load_alerts()
        
        self.AQI_THRESHOLDS = {
            'moderate': 100,
            'unhealthy_sensitive': 150,
            'unhealthy': 200,
            'very_unhealthy': 300,
            'hazardous': 301
        }
    
    def load_alerts(self):
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
    
    def add_alert(self, alert_type, title, message, duration_minutes=5):
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
            
        alert = {
            'id': len(st.session_state.alerts) + 1,
            'type': alert_type,
            'title': title,
            'message': message,
            'timestamp': datetime.now(),
            'duration_minutes': duration_minutes,
            'dismissed': False
        }
        st.session_state.alerts.append(alert)
    
    def dismiss_alert(self, alert_id):
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
            return
            
        for alert in st.session_state.alerts:
            if alert['id'] == alert_id:
                alert['dismissed'] = True
                break
    
    def get_active_alerts(self):
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
            return []
            
        current_time = datetime.now()
        active_alerts = []
        
        for alert in st.session_state.alerts:
            if not alert['dismissed']:
                time_diff = current_time - alert['timestamp']
                if time_diff.total_seconds() < alert['duration_minutes'] * 60:
                    active_alerts.append(alert)
        
        return active_alerts
    
    def clear_expired_alerts(self):
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
            return
            
        current_time = datetime.now()
        st.session_state.alerts = [
            alert for alert in st.session_state.alerts
            if not alert['dismissed'] and 
            (current_time - alert['timestamp']).total_seconds() < alert['duration_minutes'] * 60
        ]
    
    def check_hazardous_aqi(self, aqi_value: float, location: str = "Karachi") -> bool:
        try:
            if aqi_value >= self.AQI_THRESHOLDS['hazardous']:
                self.add_hazardous_alert(
                    'error',
                    '🚨 CRITICAL: Hazardous Air Quality',
                    f'{location}: AQI {aqi_value:.0f} - Emergency conditions! Stay indoors immediately.',
                    30
                )
                return True
                
            elif aqi_value >= self.AQI_THRESHOLDS['very_unhealthy']:
                self.add_hazardous_alert(
                    'error',
                    '⚠️ DANGEROUS: Very Unhealthy Air',
                    f'{location}: AQI {aqi_value:.0f} - Health emergency for all! Avoid outdoor activities.',
                    20
                )
                return True
                
            elif aqi_value >= self.AQI_THRESHOLDS['unhealthy']:
                self.add_hazardous_alert(
                    'warning',
                    '🔶 ALERT: Unhealthy Air Quality',
                    f'{location}: AQI {aqi_value:.0f} - Health effects for everyone. Limit outdoor exposure.',
                    15
                )
                return True
                
            elif aqi_value >= self.AQI_THRESHOLDS['unhealthy_sensitive']:
                self.add_hazardous_alert(
                    'warning',
                    '🔸 CAUTION: Unhealthy for Sensitive Groups',
                    f'{location}: AQI {aqi_value:.0f} - Sensitive individuals should avoid outdoor activities.',
                    10
                )
                return True
                
        except Exception as e:
            logger.error(f"Error checking hazardous AQI: {e}")
            self.add_alert('error', 'Alert System Error', f'Failed to check AQI conditions: {str(e)}', 5)
        
        return False
    
    def add_hazardous_alert(self, alert_type, title, message, duration_minutes):
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
            
        alert = {
            'id': len(st.session_state.alerts) + 1,
            'type': alert_type,
            'title': title,
            'message': message,
            'timestamp': datetime.now(),
            'duration_minutes': duration_minutes,
            'dismissed': False,
            'is_hazardous': True
        }
        st.session_state.alerts.append(alert)
    
    def fetch_current_aqi_from_api(self, api_base_url: str = os.getenv("BACKEND_URL", "http://backend:8000")) -> Optional[float]:
        try:
            response = requests.get(f"{api_base_url}/dashboard/overview", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('current_aqi')
            else:
                logger.warning(f"API returned status code: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch AQI from API: {e}")
            return None
    
    def monitor_hazardous_conditions(self, api_base_url: str = os.getenv("BACKEND_URL", "http://backend:8000")):
        try:
            current_aqi = self.fetch_current_aqi_from_api(api_base_url)
            
            if current_aqi is not None:
                is_hazardous = self.check_hazardous_aqi(current_aqi)
                
                if 'last_aqi_check' not in st.session_state:
                    st.session_state.last_aqi_check = {}
                
                st.session_state.last_aqi_check.update({
                    'timestamp': datetime.now(),
                    'aqi_value': current_aqi,
                    'is_hazardous': is_hazardous
                })
                
                return current_aqi, is_hazardous
            else:
                self.add_alert('warning', 'Data Unavailable', 
                             'Unable to fetch current AQI data. Please check your connection.', 5)
                return None, False
                
        except Exception as e:
            logger.error(f"Error monitoring hazardous conditions: {e}")
            self.add_alert('error', 'Monitoring Error', f'Failed to monitor air quality: {str(e)}', 5)
            return None, False
    
    def show_alerts(self):
        self.load_alerts()
        self.clear_expired_alerts()
        active_alerts = self.get_active_alerts()
        
        if not active_alerts:
            return
        
        alerts_container = st.container()
        
        with alerts_container:
            for alert in active_alerts:
                self.render_alert(alert)
    
    def render_alert(self, alert):
        is_hazardous = alert.get('is_hazardous', False)
        
        alert_styles = {
            'success': {
                'bg_color': 'linear-gradient(135deg, #22c55e, #15803d)',
                'border_color': '#22c55e',
                'icon': '✅'
            },
            'warning': {
                'bg_color': 'linear-gradient(135deg, #eab308, #a16207)' if not is_hazardous else 'linear-gradient(135deg, #f59e0b, #d97706)',
                'border_color': '#eab308' if not is_hazardous else '#f59e0b',
                'icon': '⚠️'
            },
            'error': {
                'bg_color': 'linear-gradient(135deg, #ef4444, #dc2626)' if not is_hazardous else 'linear-gradient(135deg, #dc2626, #991b1b)',
                'border_color': '#ef4444' if not is_hazardous else '#dc2626',
                'icon': '❌' if not is_hazardous else '🚨'
            },
            'info': {
                'bg_color': 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                'border_color': '#3b82f6',
                'icon': 'ℹ️'
            }
        }
        
        style = alert_styles.get(alert['type'], alert_styles['info'])
        
        animation_css = ""
        if is_hazardous:
            animation_css = "animation: pulse 2s infinite, slideIn 0.3s ease-out;"
        else:
            animation_css = "animation: slideIn 0.3s ease-out;"
        
        time_elapsed = (datetime.now() - alert['timestamp']).total_seconds() / 60
        time_remaining = max(0, alert['duration_minutes'] - time_elapsed)
        
        st.markdown(f"""
        <div style="
            background: {style['bg_color']};
            border: 2px solid {style['border_color']};
            border-radius: 15px;
            padding: 1rem 1.5rem;
            margin: 0.5rem 0;
            color: white;
            position: relative;
            {animation_css}
        ">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.4rem;">{style['icon']}</span>
                    <div>
                        <h4 style="margin: 0 0 0.25rem 0; font-size: 1.1rem; font-weight: 600;">
                            {alert['title']}
                        </h4>
                        <p style="margin: 0 0 0.25rem 0; font-size: 0.95rem; opacity: 0.9;">
                            {alert['message']}
                        </p>
                        <small style="opacity: 0.7; font-size: 0.8rem;">
                            {time_remaining:.0f} min remaining
                        </small>
                    </div>
                </div>
            </div>
        </div>
        
        <style>
        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(-10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        @keyframes pulse {{
            0% {{ box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4); }}
            50% {{ box-shadow: 0 4px 25px rgba(220, 38, 38, 0.8); }}
            100% {{ box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4); }}
        }}
        </style>
        """, unsafe_allow_html=True)
    
    def success(self, title, message, duration_minutes=5):
        self.add_alert('success', title, message, duration_minutes)
    
    def warning(self, title, message, duration_minutes=5):
        self.add_alert('warning', title, message, duration_minutes)
    
    def error(self, title, message, duration_minutes=5):
        self.add_alert('error', title, message, duration_minutes)
    
    def info(self, title, message, duration_minutes=5):
        self.add_alert('info', title, message, duration_minutes)

alert_system = AlertSystem()

def show_alerts():
    alert_system.show_alerts()

def add_success_alert(title, message, duration_minutes=5):
    alert_system.success(title, message, duration_minutes)

def add_warning_alert(title, message, duration_minutes=5):
    alert_system.warning(title, message, duration_minutes)

def add_error_alert(title, message, duration_minutes=5):
    alert_system.error(title, message, duration_minutes)

def add_info_alert(title, message, duration_minutes=5):
    alert_system.info(title, message, duration_minutes)

def check_and_alert_hazardous_aqi(api_base_url: str = os.getenv("BACKEND_URL", "http://backend:8000")):
    current_aqi, is_hazardous = alert_system.monitor_hazardous_conditions(api_base_url)
    return current_aqi, is_hazardous

def add_hazardous_aqi_alert(aqi_value: float, location: str = "Karachi"):
    return alert_system.check_hazardous_aqi(aqi_value, location)