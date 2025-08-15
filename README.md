# AirLens - Smart Air Quality Companion 🌬️

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68.0+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Getting Started](#getting-started)
- [API Documentation](#api-documentation)
- [Data Sources](#data-sources)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [Future Enhancements](#future-enhancements)

## 🌟 Overview

AirLens is a comprehensive air quality monitoring and prediction platform that provides real-time AQI tracking and AI-powered 72-hour forecasts for major Pakistani cities. Built with FastAPI backend and Streamlit frontend, it combines advanced machine learning with an intuitive user interface to deliver actionable air quality insights.

**Target Cities:** Karachi, Lahore, Islamabad  
**Prediction Horizons:** 24h, 48h, 72h  
**Data Sources:** OpenMeteo, IQAir, AQICN, OpenWeather APIs

## ✨ Key Features

### 🎯 Frontend (Streamlit)
- **Live Dashboard** - Interactive real-time AQI monitoring with high-quality charts
- **AI Predictions** - 72-hour AQI forecasts with hourly resolution
- **City Comparison** - Multi-city AQI comparison with live ranking
- **Detailed Analytics** - Historical trends, seasonal patterns, and pollutant analysis
- **Alert System** - Real-time warnings for unhealthy air quality levels
- **Responsive Design** - Mobile-first design with adaptive layouts

### ⚡ Backend (FastAPI)
- **RESTful API** - Modular endpoints for forecasts, historical data, and health monitoring
- **ML Model Integration** - Pre-trained models for multi-horizon predictions
- **Automated Data Pipeline** - Real-time data ingestion and processing
- **Security Features** - Rate limiting, input validation, CORS protection
- **Performance Optimized** - <200MB memory usage with local model storage

### 🔄 CI/CD Pipeline
- **Daily Data Collection** - Automated data ingestion at 3 AM UTC
- **Feature Processing** - Daily feature engineering pipeline
- **Model Training** - Daily model retraining with new data
- **Hourly Updates** - Real-time data updates every hour
- **Automated Deployment** - GitHub Actions integration

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Git

## 📖 API Documentation

### Core Endpoints

#### 🔍 Health Check
```http
GET /
```
Returns service status and loaded models count.

#### 🎯 Get Forecast
```http
GET /forecast?horizon=24
```
**Parameters:**
- `horizon` (optional): 24, 48, or 72 hours

**Response:**
```json
{
  "horizon_hours": 24,
  "predicted_aqi": 85.3,
  "risk_level": "Moderate",
  "timestamp": "2025-08-14T10:00:00Z"
}
```

#### 📊 Hourly Forecast
```http
GET /forecast/hourly
```
Returns hour-by-hour predictions for the next 72 hours.

#### 📈 Historical Data
```http
GET /historical/Karachi?days=7
```
**Parameters:**
- `days`: 1-21 days of historical data

#### 🌍 Available Locations
```http
GET /locations
```
Returns list of supported cities with current AQI readings.

#### 📋 Dashboard Overview
```http
GET /dashboard/overview
```
Returns comprehensive analytics including trends and weekly averages.

### Rate Limits
- General endpoints: 10-15 requests/minute
- Health check: 20 requests/minute
- Lightweight operations: 15 requests/minute

## 📊 Data Sources

### Primary Pipeline (Training Data)
- **OpenMeteo API**: 92 days of historical data
- **Coverage**: Comprehensive meteorological and air quality data
- **Update Frequency**: Daily
- **Storage**: Hopsworks Feature Store

### Secondary Pipeline (Real-time Data)
1. **IQAir API** (Priority 1): Real-time AQI calculations
2. **AQICN API** (Priority 2): Comprehensive pollutant data
3. **OpenWeather API** (Priority 3): Weather conditions

### Collected Parameters
- **Air Quality**: PM10, PM2.5, CO, CO₂, NO₂, SO₂, O₃
- **Weather**: Temperature, humidity, precipitation, wind patterns
- **Calculated**: AQI values using EPA breakpoint formula

## 📈 Model Performance

### Current Results (Tree-Based Models)

| Horizon | Best Model | Test R² | Test RMSE | Test MAE | Test MAPE |
|---------|------------|---------|-----------|----------|-----------|
| 24h     | ExtraTrees | 0.59    | 4.88      | 3.85     | 4.71%     |
| 48h     | ExtraTrees | 0.61    | 4.55      | 3.67     | 4.23%     |
| 72h     | ExtraTrees | 0.21    | 6.52      | 5.39     | 6.75%     |

### Model Architecture
- **7 algorithms**: CatBoost, XGBoost, LightGBM, RandomForest, ExtraTrees, GradientBoosting, DecisionTree
- **Feature Selection**: Hybrid MI-RF methodology
- **Validation**: Sliding window cross-validation
- **Optimization**: Optuna hyperparameter tuning

### AQI Risk Classification
- **Good**: 0-50
- **Moderate**: 51-100
- **Unhealthy for Sensitive**: 101-150
- **Unhealthy**: 151-200
- **Very Unhealthy**: 201-300
- **Hazardous**: 300+

## 🐳 Deployment

### Docker Deployment

#### Backend Container
```dockerfile
FROM python:3.9-slim
COPY backend/ /app/
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Frontend Container
```dockerfile
FROM python:3.9-slim
COPY frontend/ /app/
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Production Considerations
- **Memory Requirements**: <200MB per service
- **Scalability**: Horizontal scaling supported
- **Monitoring**: Built-in health checks and logging
- **Security**: Rate limiting, input validation, CORS protection

## 🤝 Acknowledgments

- OpenMeteo for comprehensive historical weather data
- IQAir for real-time air quality monitoring
- Hopsworks for feature store and MLOps platform
- EPA for AQI calculation standards and breakpoints

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact: [zainabfatimaa234@gmail.com]

**Built with ❤️ for cleaner air and healthier communities in Pakistan** 🇵🇰