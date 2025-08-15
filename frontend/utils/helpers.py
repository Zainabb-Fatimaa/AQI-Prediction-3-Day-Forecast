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