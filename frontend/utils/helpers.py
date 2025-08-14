import streamlit as st
from datetime import datetime
import numpy as np

def get_aqi_class(aqi):
    if 0 <= aqi <= 50: 
        return "aqi-good", "Good"
    elif 51 <= aqi <= 100: 
        return "aqi-moderate", "Moderate"
    elif 101 <= aqi <= 150: 
        return "aqi-unhealthy-sensitive", "Unhealthy for Sensitive Groups"
    elif 151 <= aqi <= 200: 
        return "aqi-unhealthy", "Unhealthy"
    elif 201 <= aqi <= 300: 
        return "aqi-very-unhealthy", "Very Unhealthy"
    else: 
        return "aqi-hazardous", "Hazardous"

# def get_health_recommendation(aqi, age, health_condition, activity_level):
#     recommendations = []
    
#     if aqi <= 50:
#         recommendations.append("✅ Perfect air quality! All outdoor activities are safe.")
#     elif aqi <= 100:
#         if health_condition in ["Asthma", "Heart Disease"] or age > 65:
#             recommendations.append("⚠️ Consider reducing prolonged outdoor exertion.")
#         else:
#             recommendations.append("✅ Good air quality for most people.")
#     elif aqi <= 150:
#         recommendations.append("🚨 Sensitive individuals should limit outdoor activities.")
#         if activity_level == "High":
#             recommendations.append("🏃‍♂️ Consider indoor exercise instead.")
#     elif aqi <= 200:
#         recommendations.append("🚨 Everyone should reduce outdoor activities.")
#         recommendations.append("😷 Wear a mask when going outside.")
#     else:
#         recommendations.append("🚨 Avoid all outdoor activities!")
#         recommendations.append("🏠 Stay indoors and use air purifiers.")
    
#     return recommendations

# def calculate_health_risk_score(aqi, age, health_condition, activity_level):
#     """Calculate health risk score based on multiple factors"""
#     risk_score = 0
    
#     # Age factor
#     if age > 65 or age < 12:
#         risk_score += 2
#     elif age > 50:
#         risk_score += 1
    
#     # Health condition factor
#     if health_condition in ["Asthma", "COPD"]:
#         risk_score += 3
#     elif health_condition in ["Heart Disease", "Diabetes"]:
#         risk_score += 2
#     elif health_condition == "Pregnancy":
#         risk_score += 1
    
#     if aqi > 150:
#         risk_score += 3
#     elif aqi > 100:
#         risk_score += 2
#     elif aqi > 50:
#         risk_score += 1
    
#     if activity_level == "High" and aqi > 100:
#         risk_score += 1
    
#     return risk_score

def get_risk_level_display(risk_score):
    if risk_score <= 2:
        return "🟢 **Low Risk**", "You're generally safe!", "success"
    elif risk_score <= 4:
        return "🟡 **Moderate Risk**", "Take some precautions", "warning"
    elif risk_score <= 6:
        return "🟠 **High Risk**", "Limit outdoor exposure", "error"
    else:
        return "🔴 **Very High Risk**", "Stay indoors!", "error"


def detect_anomalies(df, column='aqi'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return anomalies