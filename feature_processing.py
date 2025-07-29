import pandas as pd
import numpy as np
import os
import re
import warnings
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for GitHub Actions
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import hopsworks

warnings.filterwarnings('ignore')

def extract_max_window_from_features(feature_names: List[str]) -> int:
    """
    Extract the maximum lag or rolling window from feature names.

    Parameters:
    -----------
    feature_names : List[str]
        List of feature column names

    Returns:
    --------
    int : Maximum window size in hours
    """
    max_window = 0
    window_pattern = r'(\d+)h'  # Pattern to match numbers followed by 'h'

    print("Analyzing feature names for lag/rolling windows...")

    found_windows = []

    for feature in feature_names:
        # Find all window specifications in the feature name
        matches = re.findall(window_pattern, feature)

        for match in matches:
            window_size = int(match)
            found_windows.append((feature, window_size))
            max_window = max(max_window, window_size)

    # Sort and display findings
    found_windows.sort(key=lambda x: x[1], reverse=True)

    print(f"\nFound {len(found_windows)} features with time windows:")
    print("\nTop 10 largest windows:")
    for feature, window in found_windows[:10]:
        print(f"  {window:3d}h: {feature}")

    if found_windows:
        print(f"\nMaximum window detected: {max_window} hours")
    else:
        print("\nNo time windows found in feature names.")

    return max_window

def smart_drop_initial_rows(df: pd.DataFrame,
                           feature_columns: Optional[List[str]] = None,
                           min_drop_hours: int = 0) -> Tuple[pd.DataFrame, int]:
    if df.empty:
        print("Warning: Empty dataframe provided.")
        return df, 0

    # Use all columns if feature_columns not specified
    if feature_columns is None:
        feature_columns = df.columns.tolist()

    # Extract maximum window
    max_window = extract_max_window_from_features(feature_columns)

    # Use the larger of max_window or min_drop_hours
    rows_to_drop = max(max_window, min_drop_hours)

    if rows_to_drop == 0:
        print("No rows need to be dropped.")
        return df, 0

    initial_shape = df.shape

    # Check if we have enough rows
    if rows_to_drop >= len(df):
        print(f"Warning: Trying to drop {rows_to_drop} rows but dataframe only has {len(df)} rows.")
        print("Dropping all but the last row to preserve structure.")
        rows_to_drop = len(df) - 1

    # Drop the initial rows
    df_processed = df.iloc[rows_to_drop:].copy()

    print(f"\nRow dropping summary:")
    print(f"  Initial shape: {initial_shape}")
    print(f"  Rows dropped: {rows_to_drop}")
    print(f"  Final shape: {df_processed.shape}")
    print(f"  Data retained: {len(df_processed)/len(df)*100:.1f}%")

    return df_processed, rows_to_drop

class AQIDataPreprocessor:
    """
    Comprehensive data preprocessing for AQI forecasting
    Handles datasets with robust feature engineering and multicollinearity reduction.
    Can filter final features to match a target schema.
    """

    def __init__(self, file_path=None, dataframe=None):
        self.file_path = file_path
        self.df = dataframe
        self.processed_df = None
        self.feature_columns = []
        self.target_column = 'aqi'
        self.pollutant_columns = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3']
        self.weather_columns = ['temperature', 'humidity', 'wind_speed', 'wind_direction']
        self.removed_features = {
            'high_correlation': [],
            'low_variance': [],
            'high_vif': []
        }
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.original_numeric_features = []
        self.explicitly_kept_time_features = ['hour', 'day_of_week', 'month', 'quarter', 'dayofyear']
        self.target_schema_columns = None
        self.datetime_column = None

        # Define the specific columns you want to keep
        self.required_columns = [
            'date', 'wind_speed', 'wind_direction', 'hour', 'day', 'weekday', 'co', 'no2', 'aqi',
            'quarter', 'is_weekend', 'is_morning_rush', 'is_evening_rush', 'season', 'day_sin',
            'day_cos', 'dayofyear_sin', 'aqi_lag_48h', 'aqi_lag_72h', 'pm10_lag_24h', 'pm2_5_lag_24h',
            'co_lag_1h', 'co_lag_12h', 'co_lag_24h', 'no2_lag_1h', 'no2_lag_6h', 'no2_lag_12h',
            'no2_lag_24h', 'so2_lag_1h', 'so2_lag_6h', 'so2_lag_12h', 'so2_lag_24h', 'o3_lag_1h',
            'o3_lag_6h', 'o3_lag_12h', 'o3_lag_24h', 'humidity_lag_1h', 'humidity_lag_6h',
            'humidity_lag_12h', 'wind_speed_lag_1h', 'wind_speed_lag_6h', 'wind_speed_lag_12h',
            'wind_direction_lag_1h', 'wind_direction_lag_6h', 'wind_direction_lag_12h', 'aqi_rolling_std_3h',
            'aqi_rolling_std_6h', 'aqi_rolling_std_12h', 'aqi_rolling_std_24h', 'aqi_rolling_std_48h',
            'aqi_rolling_max_48h', 'aqi_rolling_std_72h', 'aqi_rolling_max_72h', 'pm10_rolling_std_6h',
            'pm2_5_rolling_std_6h', 'co_rolling_std_6h', 'co_rolling_std_12h', 'co_rolling_std_24h',
            'no2_rolling_std_6h', 'no2_rolling_std_24h', 'so2_rolling_std_6h', 'so2_rolling_std_24h',
            'o3_rolling_std_6h', 'o3_rolling_std_24h', 'hour_cos_rolling_mean_12h', 'day_cos_rolling_std_6h',
            'day_cos_rolling_mean_12h', 'day_cos_rolling_std_12h', 'month_cos_rolling_std_6h',
            'month_cos_rolling_std_24h', 'dayofyear_cos_rolling_std_6h', 'dayofyear_cos_rolling_std_12h',
            'pm10_lag_1h_rolling_std_6h', 'pm10_lag_1h_rolling_mean_12h', 'pm10_lag_1h_rolling_std_12h',
            'pm10_lag_1h_rolling_std_24h', 'pm10_lag_6h_rolling_std_6h', 'pm10_lag_6h_rolling_std_12h',
            'pm10_lag_6h_rolling_std_24h', 'pm10_lag_12h_rolling_mean_6h', 'pm10_lag_12h_rolling_std_6h',
            'pm10_lag_12h_rolling_mean_12h', 'pm10_lag_12h_rolling_std_12h', 'pm10_lag_12h_rolling_std_24h',
            'pm10_lag_24h_rolling_mean_6h', 'pm10_lag_24h_rolling_std_6h', 'pm10_lag_24h_rolling_std_12h',
            'pm10_lag_24h_rolling_mean_24h', 'pm10_lag_24h_rolling_std_24h', 'pm2_5_lag_1h_rolling_std_6h',
            'pm2_5_lag_1h_rolling_mean_12h', 'pm2_5_lag_1h_rolling_std_12h', 'pm2_5_lag_1h_rolling_std_24h',
            'pm2_5_lag_6h_rolling_std_6h', 'pm2_5_lag_6h_rolling_std_12h', 'pm2_5_lag_6h_rolling_std_24h',
            'pm2_5_lag_12h_rolling_mean_6h', 'pm2_5_lag_12h_rolling_std_6h', 'pm2_5_lag_12h_rolling_std_12h',
            'pm2_5_lag_12h_rolling_std_24h', 'pm2_5_lag_24h_rolling_mean_6h', 'pm2_5_lag_24h_rolling_std_6h',
            'pm2_5_lag_24h_rolling_mean_12h', 'pm2_5_lag_24h_rolling_std_12h', 'pm2_5_lag_24h_rolling_std_24h',
            'co_lag_1h_rolling_mean_6h', 'co_lag_1h_rolling_std_6h', 'co_lag_6h_rolling_std_6h',
            'co_lag_6h_rolling_mean_12h', 'co_lag_6h_rolling_std_12h', 'co_lag_6h_rolling_std_24h',
            'co_lag_12h_rolling_mean_6h', 'co_lag_12h_rolling_std_6h', 'co_lag_12h_rolling_mean_12h',
            'co_lag_12h_rolling_std_12h', 'co_lag_12h_rolling_std_24h', 'co_lag_24h_rolling_mean_6h',
            'co_lag_24h_rolling_std_6h', 'co_lag_24h_rolling_std_12h', 'co_lag_24h_rolling_mean_24h',
            'co_lag_24h_rolling_std_24h', 'no2_lag_1h_rolling_std_6h', 'no2_lag_1h_rolling_std_12h',
            'no2_lag_6h_rolling_mean_6h', 'no2_lag_6h_rolling_std_6h', 'no2_lag_6h_rolling_mean_12h',
            'no2_lag_6h_rolling_std_12h', 'no2_lag_6h_rolling_std_24h', 'no2_lag_12h_rolling_std_6h',
            'no2_lag_12h_rolling_std_12h', 'no2_lag_12h_rolling_mean_24h', 'no2_lag_12h_rolling_std_24h',
            'no2_lag_24h_rolling_mean_6h', 'no2_lag_24h_rolling_std_6h', 'no2_lag_24h_rolling_std_12h',
            'no2_lag_24h_rolling_mean_24h', 'no2_lag_24h_rolling_std_24h', 'so2_lag_1h_rolling_mean_6h',
            'so2_lag_1h_rolling_std_6h', 'so2_lag_1h_rolling_std_12h', 'so2_lag_6h_rolling_mean_6h',
            'so2_lag_6h_rolling_std_6h', 'so2_lag_6h_rolling_std_12h', 'so2_lag_6h_rolling_std_24h',
            'so2_lag_12h_rolling_std_6h', 'so2_lag_12h_rolling_std_12h', 'so2_lag_12h_rolling_std_24h',
            'so2_lag_24h_rolling_mean_6h', 'so2_lag_24h_rolling_std_6h', 'so2_lag_24h_rolling_mean_12h',
            'so2_lag_24h_rolling_std_12h', 'so2_lag_24h_rolling_mean_24h', 'so2_lag_24h_rolling_std_24h',
            'o3_lag_1h_rolling_mean_6h', 'o3_lag_1h_rolling_std_6h', 'o3_lag_1h_rolling_std_12h',
            'o3_lag_6h_rolling_mean_6h', 'o3_lag_6h_rolling_std_6h', 'o3_lag_6h_rolling_std_12h',
            'o3_lag_6h_rolling_std_24h', 'o3_lag_12h_rolling_mean_6h', 'o3_lag_12h_rolling_std_6h',
            'o3_lag_12h_rolling_std_12h', 'o3_lag_12h_rolling_std_24h', 'o3_lag_24h_rolling_mean_6h',
            'o3_lag_24h_rolling_std_6h', 'o3_lag_24h_rolling_std_12h', 'o3_lag_24h_rolling_mean_24h',
            'o3_lag_24h_rolling_std_24h', 'hour_cos_lag_1h_rolling_std_6h', 'hour_cos_lag_1h_rolling_std_12h',
            'hour_cos_lag_6h_rolling_mean_6h', 'hour_cos_lag_6h_rolling_std_6h', 'day_cos_lag_1h_rolling_std_6h',
            'day_cos_lag_1h_rolling_std_12h', 'day_cos_lag_6h_rolling_std_6h', 'day_cos_lag_6h_rolling_std_12h',
            'day_cos_lag_12h_rolling_mean_6h', 'day_cos_lag_12h_rolling_std_6h', 'day_cos_lag_12h_rolling_std_12h',
            'day_cos_lag_12h_rolling_mean_24h', 'day_cos_lag_12h_rolling_std_24h', 'day_cos_lag_24h_rolling_std_6h',
            'day_cos_lag_24h_rolling_std_12h', 'day_cos_lag_24h_rolling_mean_24h', 'day_cos_lag_24h_rolling_std_24h',
            'month_cos_lag_1h_rolling_std_6h', 'month_cos_lag_1h_rolling_std_12h', 'month_cos_lag_6h_rolling_std_6h',
            'month_cos_lag_6h_rolling_std_12h', 'month_cos_lag_6h_rolling_std_24h', 'month_cos_lag_12h_rolling_std_6h',
            'month_cos_lag_12h_rolling_std_12h', 'month_cos_lag_12h_rolling_std_24h', 'month_cos_lag_24h_rolling_std_6h',
            'month_cos_lag_24h_rolling_std_12h', 'month_cos_lag_24h_rolling_mean_24h', 'month_cos_lag_24h_rolling_std_24h',
            'dayofyear_cos_lag_1h_rolling_std_6h', 'dayofyear_cos_lag_1h_rolling_std_12h', 'dayofyear_cos_lag_6h_rolling_std_6h',
            'dayofyear_cos_lag_6h_rolling_std_12h', 'dayofyear_cos_lag_12h_rolling_std_6h', 'dayofyear_cos_lag_12h_rolling_std_12h',
            'dayofyear_cos_lag_12h_rolling_std_24h', 'temp_humidity_interaction', 'wind_u', 'wind_v',
            'pm25_pm10_ratio', 'morning_rush_no2', 'evening_rush_no2', 'aqi_rate_change_1h',
            'aqi_rate_change_6h', 'aqi_rate_change_24h'
        ]

    def load_data(self, file_path=None, dataframe=None):
        """Load the dataset from a file or use a provided DataFrame."""
        print("\n" + "="*50)
        print("LOADING DATA")
        print("="*50)

        if dataframe is not None:
            self.df = dataframe.copy()
            print("Using provided DataFrame.")
        elif file_path:
            self.file_path = file_path
            try:
                self.df = pd.read_csv(self.file_path)
                print(f"Data loaded successfully from {self.file_path}! Shape: {self.df.shape}")
            except FileNotFoundError:
                print(f"Error loading data: File not found at {self.file_path}")
                return False
            except Exception as e:
                print(f"Error loading data: {e}")
                return False
        else:
            print("No file path or DataFrame provided.")
            return False

        if self.df is not None:
            # Rename columns to be lowercase with underscores
            self.df.columns = self.df.columns.str.lower().str.replace('.', '_', regex=False)
            # Ensure specific pollutant/AQI columns match exactly after lowercasing
            self.df.rename(columns={
                'pm2_5': 'pm2_5',
                'pm10': 'pm10',
                'co': 'co',
                'so2': 'so2',
                'o3': 'o3',
                'no2': 'no2',
                'aqi': 'aqi'
            }, inplace=True)

            self.inspect_data()
            return True
        else:
            return False

    def inspect_data(self):
        """Inspect the loaded dataset"""
        print("\n" + "="*50)
        print("DATA INSPECTION")
        print("="*50)

        if self.df is None:
            print("No data available for inspection.")
            return

        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage().sum() / 1024**2:.2f} MB")

        print("\nColumn data types:")
        print(self.df.dtypes)

        print("\nMissing values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0])

        print("\nBasic statistics:")
        print(self.df.describe())

        # Identify datetime column
        self.datetime_column = None
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(self.df[col])
                    self.datetime_column = col
                    break
                except:
                    pass

        if self.datetime_column:
            print(f"\nDatetime column identified: {self.datetime_column}")
            print(f"Date range: {self.df[self.datetime_column].min()} to {self.df[self.datetime_column].max()}")
        else:
            print("\nNo suitable datetime column found.")

        # Identify target column
        self.target_column = None
        for col in self.df.columns:
            if 'aqi' in col.lower():
                self.target_column = col
                break

        if self.target_column:
            print(f"Target column: {self.target_column}")
        else:
            print("Warning: No target column ('aqi' or similar) found.")

        # Store original numeric features
        self.original_numeric_features = [col for col in self.df.select_dtypes(include=np.number).columns if col != self.datetime_column and col != self.target_column]
        print(f"\nIdentified original numeric features: {self.original_numeric_features}")

    def clean_data(self):
        """Clean and prepare the data"""
        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)

        if self.df is None:
            print("No data available to clean. Load data first.")
            return

        self.processed_df = self.df.copy()

        # 1. Handle datetime column and set index
        if self.datetime_column and self.datetime_column in self.processed_df.columns:
            print(f"Processing datetime column: {self.datetime_column}")
            self.processed_df[self.datetime_column] = pd.to_datetime(self.processed_df[self.datetime_column])
            self.processed_df = self.processed_df.sort_values(self.datetime_column)
            self.processed_df.set_index(self.datetime_column, inplace=True)
            if self.datetime_column in self.processed_df.columns:
                self.processed_df = self.processed_df.drop(columns=[self.datetime_column])
                print(f"Dropped original datetime column '{self.datetime_column}'.")
        else:
            print("No suitable datetime column found or specified. Skipping datetime index setting.")
            if not isinstance(self.processed_df.index, pd.DatetimeIndex):
                print("Creating a default DatetimeIndex for time series operations.")
                try:
                    self.processed_df.index = pd.date_range(start='2024-01-01', periods=len(self.processed_df), freq='H')
                    print("Default DatetimeIndex created.")
                except Exception as e:
                    print(f"Could not create a default DatetimeIndex: {e}. Some time-series features might fail.")

        # 2. Remove duplicates
        initial_shape = self.processed_df.shape
        self.processed_df = self.processed_df.drop_duplicates()
        print(f"Removed {initial_shape[0] - self.processed_df.shape[0]} duplicate rows")

        # 3. Handle missing values
        print("\nHandling missing values...")
        self.handle_missing_values()

        # 4. Remove outliers
        print("\nRemoving outliers...")
        self.remove_outliers()

        # 5. Data type optimization
        print("\nOptimizing data types...")
        self.optimize_data_types()

        print(f"\nCleaned data shape: {self.processed_df.shape}")

    def handle_missing_values(self):
        """Handle missing values using various strategies"""
        missing_before = self.processed_df.isnull().sum().sum()

        if isinstance(self.processed_df.index, pd.DatetimeIndex):
            # Strategy 1: Forward fill for short gaps
            for col in self.processed_df.columns:
                if self.processed_df[col].isnull().any():
                    self.processed_df[col] = self.processed_df[col].fillna(method='ffill', limit=2)

            # Strategy 2: Interpolation for longer gaps
            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.processed_df[col].isnull().any():
                    self.processed_df[col] = self.processed_df[col].interpolate(method='time', limit=6)
        else:
            print("Skipping time-based fillna and interpolate as index is not DatetimeIndex.")

        # Strategy 3: KNN imputation for remaining missing values
        if self.processed_df.isnull().any().any():
            print("Applying KNN imputation for remaining missing values...")
            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                self.processed_df[numeric_cols] = imputer.fit_transform(self.processed_df[numeric_cols])

        # Strategy 4: Mode imputation for categorical variables
        categorical_cols = self.processed_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.processed_df[col].isnull().any():
                mode_value = self.processed_df[col].mode()
                if len(mode_value) > 0:
                    self.processed_df[col] = self.processed_df[col].fillna(mode_value[0])

        missing_after = self.processed_df.isnull().sum().sum()
        print(f"Missing values: {missing_before} -> {missing_after}")

    def remove_outliers(self):
        """Replace outliers with NaN using IQR method and then backfill"""
        print("Replacing outliers with NaN using IQR and then backfilling...")
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        outliers_replaced = 0

        for col in numeric_cols:
            if col == self.target_column:
                Q1 = self.processed_df[col].quantile(0.05)
                Q3 = self.processed_df[col].quantile(0.95)
            else:
                Q1 = self.processed_df[col].quantile(0.25)
                Q3 = self.processed_df[col].quantile(0.75)

            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            outliers = (self.processed_df[col] < lower_bound) | (self.processed_df[col] > upper_bound)
            outliers_count = outliers.sum()

            if outliers_count > 0:
                print(f"  {col}: {outliers_count} outliers replaced with NaN")
                self.processed_df.loc[outliers, col] = np.nan
                outliers_replaced += outliers_count

        print(f"Total outliers replaced with NaN: {outliers_replaced}")

        if isinstance(self.processed_df.index, pd.DatetimeIndex):
            print("Backfilling NaN values after outlier replacement...")
            self.processed_df.fillna(method='bfill', inplace=True)
            self.processed_df.fillna(method='ffill', inplace=True)
            print("Backfilling and forward filling completed.")
        else:
            print("Skipping backfilling as index is not DatetimeIndex.")

    def optimize_data_types(self):
        """Optimize data types to reduce memory usage"""
        for col in self.processed_df.columns:
            if self.processed_df[col].dtype == 'float64':
                self.processed_df[col] = self.processed_df[col].astype('float32')
            elif self.processed_df[col].dtype == 'int64':
                self.processed_df[col] = self.processed_df[col].astype('int32')

    def create_time_features(self):
        """Create comprehensive time-based features"""
        print("\n" + "="*50)
        print("CREATING TIME FEATURES")
        print("="*50)

        if not isinstance(self.processed_df.index, pd.DatetimeIndex):
            print("Skipping time feature creation as index is not a DatetimeIndex.")
            return

        # Basic time features
        self.processed_df['hour'] = self.processed_df.index.hour
        self.processed_df['day_of_week'] = self.processed_df.index.dayofweek
        self.processed_df['day'] = self.processed_df.index.day
        self.processed_df['weekday'] = self.processed_df.index.dayofweek
        self.processed_df['month'] = self.processed_df.index.month
        self.processed_df['quarter'] = self.processed_df.index.quarter
        self.processed_df['year'] = self.processed_df.index.year
        self.processed_df['dayofyear'] = self.processed_df.index.dayofyear

        # Categorical time features
        self.processed_df['is_weekend'] = (self.processed_df.index.dayofweek >= 5).astype(int)
        self.processed_df['is_holiday'] = 0

        # Rush hour indicators for Karachi
        self.processed_df['is_morning_rush'] = ((self.processed_df['hour'] >= 7) &
                                               (self.processed_df['hour'] <= 10)).astype(int)
        self.processed_df['is_evening_rush'] = ((self.processed_df['hour'] >= 17) &
                                               (self.processed_df['hour'] <= 20)).astype(int)

        # Season indicators
        self.processed_df['season'] = self.processed_df['month'] % 12 // 3 + 1

        # Cyclical encoding for periodic features
        self.processed_df['hour_sin'] = np.sin(2 * np.pi * self.processed_df['hour'] / 24)
        self.processed_df['hour_cos'] = np.cos(2 * np.pi * self.processed_df['hour'] / 24)

        self.processed_df['day_sin'] = np.sin(2 * np.pi * self.processed_df['day_of_week'] / 7)
        self.processed_df['day_cos'] = np.cos(2 * np.pi * self.processed_df['day_of_week'] / 7)

        self.processed_df['month_sin'] = np.sin(2 * np.pi * self.processed_df['month'] / 12)
        self.processed_df['month_cos'] = np.cos(2 * np.pi * self.processed_df['month'] / 12)

        self.processed_df['dayofyear_sin'] = np.sin(2 * np.pi * self.processed_df['dayofyear'] / 365)
        self.processed_df['dayofyear_cos'] = np.cos(2 * np.pi * self.processed_df['dayofyear'] / 365)

        print(f"Created time features")

    def create_lag_features(self):
        """Create lag features for time series"""
        print("\n" + "="*50)
        print("CREATING LAG FEATURES")
        print("="*50)

        if not isinstance(self.processed_df.index, pd.DatetimeIndex):
            print("Skipping lag feature creation as index is not a DatetimeIndex.")
            return

        # Create lag features for target variable (aqi)
        if self.target_column and self.target_column in self.processed_df.columns:
            for lag in [48, 72]:
                self.processed_df[f'{self.target_column}_lag_{lag}h'] = self.processed_df[self.target_column].shift(lag)
        else:
            print(f"Warning: Target column '{self.target_column}' not found. Skipping target lag features.")

        # Create specific lag features for pollutants
        pollutant_lags = {
            'pm10': [24],
            'pm2_5': [24],
            'co': [1, 12, 24],
            'no2': [1, 6, 12, 24],
            'so2': [1, 6, 12, 24],
            'o3': [1, 6, 12, 24]
        }

        for pollutant, lags in pollutant_lags.items():
            if pollutant in self.processed_df.columns:
                for lag in lags:
                    col_name = f'{pollutant}_lag_{lag}h'
                    if col_name in self.required_columns:
                        self.processed_df[col_name] = self.processed_df[pollutant].shift(lag)

        # Create lag features for weather variables
        weather_lags = {
            'humidity': [1, 6, 12],
            'wind_speed': [1, 6, 12],
            'wind_direction': [1, 6, 12]
        }

        for weather_var, lags in weather_lags.items():
            if weather_var in self.processed_df.columns:
                for lag in lags:
                    col_name = f'{weather_var}_lag_{lag}h'
                    if col_name in self.required_columns:
                        self.processed_df[col_name] = self.processed_df[weather_var].shift(lag)

        print(f"Created lag features")

    def create_rolling_features(self):
        """Create rolling statistics features"""
        print("\n" + "="*50)
        print("CREATING ROLLING FEATURES")
        print("="*50)

        if not isinstance(self.processed_df.index, pd.DatetimeIndex):
            print("Skipping rolling feature creation as index is not a DatetimeIndex.")
            return

        # Rolling features for aqi
        if self.target_column and self.target_column in self.processed_df.columns:
            aqi_rolling_windows = [3, 6, 12, 24, 48, 72]
            for window in aqi_rolling_windows:
                if f'{self.target_column}_rolling_std_{window}h' in self.required_columns:
                    self.processed_df[f'{self.target_column}_rolling_std_{window}h'] = (
                        self.processed_df[self.target_column].rolling(window=window).std()
                    )
                if f'{self.target_column}_rolling_max_{window}h' in self.required_columns:
                    self.processed_df[f'{self.target_column}_rolling_max_{window}h'] = (
                        self.processed_df[self.target_column].rolling(window=window).max()
                    )

        # Rolling features for pollutants
        pollutant_rolling = {
            'pm10': [6],
            'pm2_5': [6],
            'co': [6, 12, 24],
            'no2': [6, 24],
            'so2': [6, 24],
            'o3': [6, 24]
        }

        for pollutant, windows in pollutant_rolling.items():
            if pollutant in self.processed_df.columns:
                for window in windows:
                    if f'{pollutant}_rolling_std_{window}h' in self.required_columns:
                        self.processed_df[f'{pollutant}_rolling_std_{window}h'] = (
                            self.processed_df[pollutant].rolling(window=window).std()
                        )

        # Create complex rolling features
        self.create_complex_rolling_features()

        print(f"Created rolling features")

    def create_complex_rolling_features(self):
        """Create complex rolling features for lag combinations"""

        # Define all the complex rolling features from required_columns
        complex_features = [
            ('hour_cos', 'mean', 12),
            ('day_cos', 'std', 6), ('day_cos', 'mean', 12), ('day_cos', 'std', 12),
            ('month_cos', 'std', 6), ('month_cos', 'std', 24),
            ('dayofyear_cos', 'std', 6), ('dayofyear_cos', 'std', 12)
        ]

        # Create required cos features if they don't exist
        if 'hour_cos' not in self.processed_df.columns:
            self.processed_df['hour_cos'] = np.cos(2 * np.pi * self.processed_df['hour'] / 24)
        if 'day_cos' not in self.processed_df.columns:
            self.processed_df['day_cos'] = np.cos(2 * np.pi * self.processed_df['day_of_week'] / 7)
        if 'month_cos' not in self.processed_df.columns:
            self.processed_df['month_cos'] = np.cos(2 * np.pi * self.processed_df['month'] / 12)
        if 'dayofyear_cos' not in self.processed_df.columns:
            self.processed_df['dayofyear_cos'] = np.cos(2 * np.pi * self.processed_df['dayofyear'] / 365)

        for feature, stat, window in complex_features:
            feature_name = f'{feature}_rolling_{stat}_{window}h'
            if feature_name in self.required_columns and feature in self.processed_df.columns:
                if stat == 'mean':
                    self.processed_df[feature_name] = self.processed_df[feature].rolling(window=window).mean()
                elif stat == 'std':
                    self.processed_df[feature_name] = self.processed_df[feature].rolling(window=window).std()

        # Create lag-based rolling features for pollutants
        pollutants = ['pm10', 'pm2_5', 'co', 'no2', 'so2', 'o3']
        time_features = ['hour_cos', 'day_cos', 'month_cos', 'dayofyear_cos']

        for pollutant in pollutants:
            if pollutant in self.processed_df.columns:
                # Ensure lag features exist before creating rolling features on them
                for lag in [1, 6, 12, 24]:
                    lag_col = f'{pollutant}_lag_{lag}h'
                    if lag_col not in self.processed_df.columns:
                        if any(f'{lag_col}_rolling_' in req_col for req_col in self.required_columns):
                            self.processed_df[lag_col] = self.processed_df[pollutant].shift(lag)

                # Create rolling features for these lag features
                for lag in [1, 6, 12, 24]:
                    lag_col = f'{pollutant}_lag_{lag}h'
                    if lag_col in self.processed_df.columns:
                        for window in [6, 12, 24]:
                            for stat in ['mean', 'std']:
                                feature_name = f'{lag_col}_rolling_{stat}_{window}h'
                                if feature_name in self.required_columns:
                                    if stat == 'mean':
                                        self.processed_df[feature_name] = self.processed_df[lag_col].rolling(window=window).mean()
                                    elif stat == 'std':
                                        self.processed_df[feature_name] = self.processed_df[lag_col].rolling(window=window).std()

        # Create lag-based rolling features for time features
        for time_feature in time_features:
            if time_feature in self.processed_df.columns:
                # Ensure lag features exist for time features
                for lag in [1, 6, 12, 24]:
                    lag_col = f'{time_feature}_lag_{lag}h'
                    if lag_col not in self.processed_df.columns:
                        if any(f'{lag_col}_rolling_' in req_col for req_col in self.required_columns):
                            self.processed_df[lag_col] = self.processed_df[time_feature].shift(lag)

                # Create rolling features for these lag features
                for lag in [1, 6, 12, 24]:
                    lag_col = f'{time_feature}_lag_{lag}h'
                    if lag_col in self.processed_df.columns:
                        for window in [6, 12, 24]:
                            for stat in ['mean', 'std']:
                                feature_name = f'{lag_col}_rolling_{stat}_{window}h'
                                if feature_name in self.required_columns:
                                    if stat == 'mean':
                                        self.processed_df[feature_name] = self.processed_df[lag_col].rolling(window=window).mean()
                                    elif stat == 'std':
                                        self.processed_df[feature_name] = self.processed_df[lag_col].rolling(window=window).std()

    def create_interaction_features(self):
        """Create interaction features between important variables"""
        print("\n" + "="*50)
        print("CREATING INTERACTION FEATURES")
        print("="*50)

        # Temperature and humidity interaction
        if 'temperature' in self.processed_df.columns and 'humidity' in self.processed_df.columns and 'temp_humidity_interaction' in self.required_columns:
            self.processed_df['temp_humidity_interaction'] = (
                self.processed_df['temperature'] * self.processed_df['humidity'] / 100
            )

        # Wind speed and direction features
        if 'wind_speed' in self.processed_df.columns and 'wind_direction' in self.processed_df.columns:
            if 'wind_u' in self.required_columns:
                self.processed_df['wind_u'] = (
                    self.processed_df['wind_speed'] * np.cos(np.radians(self.processed_df['wind_direction']))
                )
            if 'wind_v' in self.required_columns:
                self.processed_df['wind_v'] = (
                    self.processed_df['wind_speed'] * np.sin(np.radians(self.processed_df['wind_direction']))
                )

        # PM2.5 and PM10 ratio
        if 'pm2_5' in self.processed_df.columns and 'pm10' in self.processed_df.columns and 'pm25_pm10_ratio' in self.required_columns:
            self.processed_df['pm25_pm10_ratio'] = (
                self.processed_df['pm2_5'] / (self.processed_df['pm10'] + 1e-6)
            )

        # Rush hour and pollutant interactions
        if 'is_morning_rush' in self.processed_df.columns and 'no2' in self.processed_df.columns and 'morning_rush_no2' in self.required_columns:
            self.processed_df['morning_rush_no2'] = (
                self.processed_df['is_morning_rush'] * self.processed_df['no2']
            )

        if 'is_evening_rush' in self.processed_df.columns and 'no2' in self.processed_df.columns and 'evening_rush_no2' in self.required_columns:
            self.processed_df['evening_rush_no2'] = (
                self.processed_df['is_evening_rush'] * self.processed_df['no2']
            )

        print("Created interaction features")

    def create_statistical_features(self):
        """Create statistical features"""
        print("\n" + "="*50)
        print("CREATING STATISTICAL FEATURES")
        print("="*50)

        if self.target_column and self.target_column in self.processed_df.columns:
            # Rate of change features for aqi
            rate_changes = [1, 6, 24]
            for hours in rate_changes:
                feature_name = f'{self.target_column}_rate_change_{hours}h'
                if feature_name in self.required_columns:
                    self.processed_df[feature_name] = self.processed_df[self.target_column].diff(hours)

            print("Created statistical features for target column.")
        else:
            print(f"Warning: Target column '{self.target_column}' not found. Skipping statistical features for target column.")

    def remove_low_variance_features(self, threshold=1e-6):
        """Remove features with very low variance (effectively constant)"""
        print("\n" + "="*50)
        print("REMOVING LOW VARIANCE FEATURES")
        print("="*50)

        if self.processed_df is None or self.processed_df.empty:
            print("No data to remove low variance features from.")
            return

        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        features_to_drop = []

        # Only remove low variance features that are NOT in required_columns
        for col in numeric_cols:
            if col in self.required_columns:
                continue

            variance = self.processed_df[col].var(skipna=True)
            if pd.isna(variance) or variance < threshold:
                features_to_drop.append(col)
                self.removed_features['low_variance'].append(col)
                print(f"  Removing '{col}' due to low variance ({variance:.6f})")

        if features_to_drop:
            self.processed_df = self.processed_df.drop(columns=features_to_drop)
            print(f"Removed {len(features_to_drop)} low variance features.")
        else:
            print("No low variance features found to remove.")

    def reduce_multicollinearity(self, correlation_threshold=0.85):
        """Reduce multicollinearity but keep all required columns"""
        print("\n" + "="*50)
        print("REDUCING MULTICOLLINEARITY (Correlation)")
        print("="*50)

        if self.processed_df is None or self.processed_df.empty:
            print("No data to reduce multicollinearity from.")
            return []

        # Exclude non-numeric columns and the target column for correlation calculation
        numeric_df = self.processed_df.select_dtypes(include=[np.number]).copy()
        if self.target_column in numeric_df.columns:
            features_for_corr = numeric_df.drop(columns=[self.target_column])
        else:
            features_for_corr = numeric_df

        corr_matrix = features_for_corr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features to drop based on correlation threshold
        features_to_drop_corr = set()
        for column in upper.columns:
            if column in self.required_columns:
                continue

            highly_correlated_cols = upper.index[upper[column] > correlation_threshold].tolist()

            if highly_correlated_cols:
                for corr_col in highly_correlated_cols:
                    if corr_col not in self.required_columns and column not in self.required_columns:
                        if column not in features_to_drop_corr:
                            features_to_drop_corr.add(column)
                            self.removed_features['high_correlation'].append(column)

        # Drop the identified features
        if features_to_drop_corr:
            self.processed_df = self.processed_df.drop(columns=list(features_to_drop_corr), errors='ignore')
            print(f"Removed {len(features_to_drop_corr)} features due to high multicollinearity.")
        else:
            print("No features found with multicollinearity above the threshold to remove.")

        # Update feature_columns based on current columns, excluding target
        self.feature_columns = [col for col in self.processed_df.columns if col != self.target_column]

        return self.feature_columns

    def encode_categorical_features(self):
        """Encode categorical features"""
        print("\n" + "="*50)
        print("ENCODING CATEGORICAL FEATURES")
        print("="*50)

        categorical_cols = self.processed_df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if col != self.target_column and col in self.required_columns:
                le = LabelEncoder()
                self.processed_df[col] = le.fit_transform(self.processed_df[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {len(le.classes_)} unique categories")

    def scale_features(self):
        """Scale numerical features"""
        print("\n" + "="*50)
        print("SCALING FEATURES")
        print("="*50)
        print("Skipping scaling as requested.")

    def enhanced_finalize_dataset(self):
        """Enhanced finalize_dataset method with smart row dropping"""
        print("\n" + "="*50)
        print("SMART FINALIZING DATASET")
        print("="*50)

        # Step 1: Handle columns that were generated but are not in the required list
        generated_columns = set(self.processed_df.columns)
        initial_columns = set(self.df.columns) if self.df is not None else set()
        generated_only = [col for col in generated_columns if col not in initial_columns and col != self.target_column]

        columns_to_drop_not_required = [col for col in generated_only if col not in self.required_columns]

        if columns_to_drop_not_required:
            print(f"Dropping {len(columns_to_drop_not_required)} generated columns that are not in the required list.")
            self.processed_df = self.processed_df.drop(columns=list(columns_to_drop_not_required), errors='ignore')

        # Step 2: Add back any required columns that might be missing
        missing_required_cols = [col for col in self.required_columns if col not in self.processed_df.columns]
        if missing_required_cols:
            print(f"Adding back {len(missing_required_cols)} missing required columns filled with NaN.")
            for col in missing_required_cols:
                dtype = 'float32'
                if col in ['date', 'date_str']:
                    dtype = 'object'
                elif col in ['is_weekend', 'is_holiday', 'is_morning_rush', 'is_evening_rush', 'season', 'hour', 'day', 'weekday', 'month', 'quarter', 'year', 'dayofyear']:
                    dtype = 'int32'

                self.processed_df[col] = pd.NA

        # Step 3: Handle date column and index setup
        if self.datetime_column and self.datetime_column.lower() == 'date' and 'date' in self.required_columns:
            if self.processed_df.index.name != 'date':
                if self.processed_df.index.name is not None:
                    self.processed_df = self.processed_df.reset_index()
                
                actual_date_col = None
                for col in self.processed_df.columns:
                    try:
                        pd.to_datetime(self.processed_df[col])
                        actual_date_col = col
                        break
                    except:
                        pass

                if actual_date_col and actual_date_col != 'date':
                    self.processed_df = self.processed_df.rename(columns={actual_date_col: 'date'})

                if 'date' in self.processed_df.columns:
                    try:
                        self.processed_df['date'] = pd.to_datetime(self.processed_df['date'])
                        self.processed_df.set_index('date', inplace=True)
                        print("Successfully set 'date' as DatetimeIndex.")
                    except Exception as e:
                        print(f"Warning: Could not set 'date' as DatetimeIndex during finalization: {e}")
                else:
                    print("Warning: 'date' column not found after adding back missing required columns.")

        # Step 4: Filter to keep only required columns that exist
        existing_required_cols = [col for col in self.required_columns if col in self.processed_df.columns]
        missing_required_cols_after_filter = [col for col in self.required_columns if col not in self.processed_df.columns]

        if missing_required_cols_after_filter:
            print(f"Warning: The following required columns are still missing: {missing_required_cols_after_filter}")

        final_columns_to_keep = [col for col in existing_required_cols if col != self.processed_df.index.name]

        if final_columns_to_keep:
            self.processed_df = self.processed_df[final_columns_to_keep]
        else:
            print("Warning: No valid columns to keep after filtering!")

        # Step 5: SMART ROW DROPPING
        print("\nApplying smart row dropping based on maximum lag/rolling window...")

        all_columns = self.processed_df.columns.tolist()
        if self.processed_df.index.name:
            all_columns.append(self.processed_df.index.name)

        initial_rows = len(self.processed_df)
        self.processed_df, rows_dropped = smart_drop_initial_rows(
            self.processed_df,
            feature_columns=all_columns,
            min_drop_hours=0
        )

        print(f"Smart dropping removed {rows_dropped} initial rows")

        # Step 6: Handle any remaining NaN values
        remaining_nans_before = self.processed_df.isnull().sum().sum()
        if remaining_nans_before > 0:
            print(f"\nHandling {remaining_nans_before} remaining NaN values in the dataset...")

            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.processed_df[col].isnull().any():
                    self.processed_df[col] = self.processed_df[col].interpolate(
                        method='linear', limit_direction='both'
                    )

            self.processed_df = self.processed_df.fillna(method='ffill')
            self.processed_df = self.processed_df.fillna(method='bfill')

            remaining_nans_after = self.processed_df.isnull().sum().sum()
            print(f"NaN values after interpolation and filling: {remaining_nans_after}")

            if remaining_nans_after > 0:
                print("Dropping remaining rows with unfillable NaN values...")
                pre_dropna_shape = self.processed_df.shape
                self.processed_df = self.processed_df.dropna()
                post_dropna_shape = self.processed_df.shape
                print(f"Additional rows dropped due to unfillable NaNs: {pre_dropna_shape[0] - post_dropna_shape[0]}")

        # Step 7: Update feature_columns and final reporting
        self.feature_columns = [col for col in self.processed_df.columns if col != self.target_column]

        print("\nFinal check for low variance features among required columns...")
        self.remove_low_variance_features()
        self.feature_columns = [col for col in self.processed_df.columns if col != self.target_column]

        print(f"\nFinal dataset shape after smart processing: {self.processed_df.shape}")
        print(f"Final feature count: {len(self.feature_columns)}")
        print(f"Target column: {self.target_column}")
        print(f"Data retention: {len(self.processed_df)/initial_rows*100:.1f}% of original rows")

    def get_feature_importance_analysis(self):
        """Analyze feature importance using correlation with target"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)

        if not self.feature_columns:
            print("No features available for analysis")
            return None

        if self.target_column not in self.processed_df.columns:
            print(f"Target column '{self.target_column}' not found in processed data. Skipping feature importance analysis.")
            return None

        correlations = []
        analysis_df = self.processed_df[self.feature_columns + [self.target_column]]

        for feature in self.feature_columns:
            if pd.api.types.is_numeric_dtype(analysis_df[feature]):
                try:
                    if analysis_df[feature].var(skipna=True) > 1e-9:
                        corr = analysis_df[feature].corr(analysis_df[self.target_column])
                        correlations.append((feature, abs(corr)))
                    else:
                        correlations.append((feature, 0))
                except Exception as e:
                    print(f"Could not calculate correlation for feature '{feature}': {e}")
                    correlations.append((feature, 0))

        correlations.sort(key=lambda x: x[1], reverse=True)

        print("Top 20 features by correlation with target:")
        meaningful_correlations = [(f, c) for f, c in correlations if c > 0]
        for i, (feature, corr) in enumerate(meaningful_correlations[:20]):
            print(f"{i+1:2d}. {feature:<40}: {corr:.4f}")

        if not meaningful_correlations:
            print("No features with non-zero correlation found.")

        return correlations

    def generate_preprocessing_report(self):
        """Generate a comprehensive preprocessing report"""
        print("\n" + "="*80)
        print("PREPROCESSING REPORT")
        print("="*80)

        if self.df is not None:
            print(f"Original dataset shape: {self.df.shape}")
        else:
            print("Original dataset not loaded.")

        print(f"Processed dataset shape: {self.processed_df.shape}")
        print(f"Target column: {self.target_column}")
        print(f"Final feature count: {len(self.feature_columns)}")
        print(f"Required columns kept: {len([col for col in self.required_columns if col in self.processed_df.columns])}")
        
        missing_final_required = [col for col in self.required_columns if col not in self.processed_df.columns]
        if missing_final_required:
            print(f"Required columns still missing in final output: {missing_final_required}")

        print(f"\nFeatures removed during preprocessing steps:")
        print(f"  Removed due to low variance: {len(self.removed_features['low_variance'])}")
        print(f"  Removed due to high correlation: {len(self.removed_features['high_correlation'])}")

        print(f"\nData quality:")
        print(f"  Missing values: {self.processed_df.isnull().sum().sum()}")
        print(f"  Duplicate rows: {self.processed_df.duplicated().sum()}")

        print(f"\nMemory usage: {self.processed_df.memory_usage().sum() / 1024**2:.2f} MB")

    def run_full_preprocessing(self, correlation_threshold=0.85, variance_threshold=1e-6, vif_threshold=10, dataframe=None, target_schema_columns=None):
        """Run the complete preprocessing pipeline"""
        print("="*80)
        print("STARTING FULL PREPROCESSING PIPELINE")
        print("="*80)

        # Step 1: Load data
        if not self.load_data(dataframe=existing_df):
            return False

        # Step 2: Clean data
        self.clean_data()

        # Step 3: Feature engineering
        self.create_time_features()
        self.create_lag_features()
        self.create_rolling_features()
        self.create_interaction_features()
        self.create_statistical_features()

        # Step 4: Reduce multicollinearity
        self.feature_columns = self.reduce_multicollinearity(
            correlation_threshold=correlation_threshold
        )

        # Step 5: Encode and scale
        self.encode_categorical_features()
        self.scale_features()

        # Step 6: Finalize
        self.enhanced_finalize_dataset()

        # Step 7: Analysis and reporting
        if self.target_column and self.target_column in self.processed_df.columns:
            self.get_feature_importance_analysis()
        else:
            print("\nSkipping feature importance analysis as target column is not in final features.")

        self.generate_preprocessing_report()

        print("\n" + "="*80)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*80)

        return True

    def get_processed_data(self):
        """Return the processed dataset"""
        if self.processed_df is not None:
            y_data = None
            if self.target_column and self.target_column in self.processed_df.columns:
                y_data = self.processed_df[self.target_column]
                X_features = [col for col in self.feature_columns if col != self.target_column]
            else:
                X_features = self.feature_columns
                if self.target_column:
                    print(f"Warning: Target column '{self.target_column}' not found in processed data after filtering.")

            valid_X_features = [col for col in X_features if col in self.processed_df.columns]
            invalid_X_features = [col for col in X_features if col not in self.processed_df.columns]

            if invalid_X_features:
                print(f"Warning: The following features listed to keep were not found after preprocessing and filtering: {invalid_X_features}")

            return {
                'X': self.processed_df[valid_X_features],
                'y': y_data,
                'feature_names': valid_X_features,
                'target_name': self.target_column,
                'full_data': self.processed_df
            }
        else:
            print("No processed data available. Run preprocessing first.")
            return None

def get_hopsworks_api_key():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise EnvironmentError("HOPSWORKS_API_KEY environment variable not set.")
    return api_key

def get_hopsworks_project():
    project = os.getenv("HOPSWORKS_PROJECT")
    if not project:
        raise EnvironmentError("HOPSWORKS_PROJECT environment variable not set.")
    return project

# Initialize Hopsworks connection
api_key = get_hopsworks_api_key()
project_name = get_hopsworks_project()
project = hopsworks.login(api_key_value=api_key, project=project_name)
fs = project.get_feature_store()

# Read raw data from Hopsworks
fg = fs.get_feature_group(name="karachi_raw_data_store", version=1)
existing_df = fg.read()

# Assuming you have a Preprocessor class with the methods shown in your first code block
preprocessor = AQIDataPreprocessor()  # Initialize with your target column
preprocessor.run_full_preprocessing(dataframe=existing_df)

# Get processed data
processed_data = preprocessor.get_processed_data()

if processed_data is not None:
    final_features_df = processed_data['full_data']
    
    # Prepare the dataframe for Hopsworks feature store
    if final_features_df.index.name == 'date':
        final_features_df = final_features_df.reset_index()
        print("Moved 'date' from index to column.")

    # Create event_time column for timestamp, keep 'date' as string
    final_features_df["event_time"] = pd.to_datetime(final_features_df["date"])
    final_features_df["date"] = final_features_df["date"].astype(str)

    # Create or get feature group for processed features
    feature_group = fs.get_or_create_feature_group(
        name="aqi_processed_features_store_clean",
        version=1,
        description="Processed AQI features after cleaning and feature engineering",
        primary_key=["id"],
        event_time="event_time",
        online_enabled=True
    )

    # Insert data into feature group
    try:
        feature_group.insert(
            final_features_df,
            write_options={
                "wait_for_job": True,  # Wait for job to complete
                "start_offline_backfill": True  # Start offline backfill immediately
            }
        )
        print("Successfully stored processed features to Hopsworks feature store.")
    except Exception as e:
        print(f"Error storing data to Hopsworks: {e}")
else:
    print("No processed data available to store.")
