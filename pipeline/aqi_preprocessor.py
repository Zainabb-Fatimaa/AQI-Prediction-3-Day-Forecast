import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import re
from typing import List, Tuple, Optional

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

    for feature in feature_names:
        matches = re.findall(window_pattern, feature)
        for match in matches:
            window_size = int(match)
            max_window = max(max_window, window_size)

    return max_window

def smart_drop_initial_rows(df: pd.DataFrame,
                           feature_columns: Optional[List[str]] = None,
                           min_drop_hours: int = 0) -> Tuple[pd.DataFrame, int]:
    """
    Drop initial rows based on the maximum lag/rolling window found in feature names.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to process
    feature_columns : List[str], optional
        List of feature column names. If None, uses all columns.
    min_drop_hours : int, default=0
        Minimum number of hours to drop (useful if you want to ensure a minimum)

    Returns:
    --------
    Tuple[pd.DataFrame, int] : (processed_dataframe, rows_dropped)
    """
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
        self.df = dataframe # Initialize with provided DataFrame if any
        self.processed_df = None
        self.feature_columns = []
        self.target_column = 'aqi' # Updated to lowercase
        self.pollutant_columns = ['pm2_5', 'pm10', 'no2', 'so2', 'co', 'o3'] # Updated to lowercase
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
        self.target_schema_columns = None # Will be set in run_full_preprocessing

    def load_data(self, file_path=None, dataframe=None):
        """Load the dataset from a file or use a provided DataFrame."""
        print("\n" + "="*50)
        print("LOADING DATA")
        print("="*50)

        if dataframe is not None:
            self.df = dataframe.copy() # Work on a copy to avoid modifying the original
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
            print("No file path or DataFrame provided. Creating sample data structure...")
            self.create_sample_data()

        if self.df is not None:
             # Rename columns to be lowercase with underscores
             self.df.columns = self.df.columns.str.lower().str.replace('.', '_', regex=False) # Replace dots for consistency
             # Ensure specific pollutant/AQI columns match exactly after lowercasing
             self.df.rename(columns={
                 'pm2_5': 'pm2_5', # Ensure it's pm2_5
                 'pm10': 'pm10', # Ensure it's pm10
                 'co': 'co', # Ensure it's co
                 'so2': 'so2', # Ensure it's so2
                 'o3': 'o3', # Ensure it's o3
                 'no2': 'no2', # Ensure it's no2
                 'aqi': 'aqi' # Ensure it's aqi
             }, inplace=True)

             self.inspect_data()
             return True
        else:
             return False

    def create_sample_data(self):
        """Create sample data structure for demonstration"""
        # Create sample hourly data for 500 rows (about 21 days)
        dates = pd.date_range(start='2024-01-01', periods=500, freq='H')

        # Generate realistic AQI and pollutant data for Karachi
        np.random.seed(42)

        # Base patterns with seasonality and trends
        base_aqi = 80 + 30 * np.sin(np.arange(500) * 2 * np.pi / 24)  # Daily cycle
        base_aqi += 20 * np.sin(np.arange(500) * 2 * np.pi / (24 * 7))  # Weekly cycle
        base_aqi += np.random.normal(0, 15, 500)  # Random noise
        base_aqi = np.clip(base_aqi, 0, 300)  # Clip to realistic range

        # Generate correlated pollutants
        pm25 = base_aqi * 0.4 + np.random.normal(0, 5, 500)
        pm10 = base_aqi * 0.6 + np.random.normal(0, 8, 500)
        no2 = base_aqi * 0.3 + np.random.normal(0, 3, 500)
        so2 = base_aqi * 0.2 + np.random.normal(0, 2, 500)
        co = base_aqi * 0.1 + np.random.normal(0, 1, 500)
        o3 = base_aqi * 0.25 + np.random.normal(0, 4, 500)

        self.df = pd.DataFrame({
            'date': dates,
            'aqi': base_aqi, # Use lowercase 'aqi' for consistency
            'pm2_5': pm25, # Use lowercase pm2_5
            'pm10': pm10, # Use lowercase pm10
            'no2': no2, # Use lowercase no2
            'so2': so2, # Use lowercase so2
            'co': co, # Use lowercase co
            'o3': o3, # Use lowercase o3
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'location': 'Karachi'
        })

        # Add some missing values to simulate real data
        missing_indices = np.random.choice(500, 25, replace=False)
        self.df.loc[missing_indices, 'pm2_5'] = np.nan
        missing_indices = np.random.choice(500, 15, replace=False)
        self.df.loc[missing_indices, 'temperature'] = np.nan

        print("Sample data created successfully!")

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
        self.datetime_column = None # Reset datetime column
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                # Try converting to datetime to confirm
                try:
                    pd.to_datetime(self.df[col])
                    self.datetime_column = col
                    break
                except:
                    pass # Not a datetime column, continue

        if self.datetime_column:
            print(f"\nDatetime column identified: {self.datetime_column}")
            print(f"Date range: {self.df[self.datetime_column].min()} to {self.df[self.datetime_column].max()}")
        else:
            print("\nNo suitable datetime column found.")

        # Identify target column (ensure it matches the actual column name)
        self.target_column = None # Reset target column
        for col in self.df.columns:
            if 'aqi' in col.lower(): # Check for AQI (case insensitive)
                self.target_column = col
                break

        if self.target_column:
            print(f"Target column: {self.target_column}")
        else:
            print("Warning: No target column ('aqi' or similar) found.")

        # Store original numeric features (excluding datetime and target if they are numeric)
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
            # Drop the original datetime column if it's not needed after setting index
            if self.datetime_column in self.processed_df.columns:
                 self.processed_df = self.processed_df.drop(columns=[self.datetime_column])
                 print(f"Dropped original datetime column '{self.datetime_column}'.")
        else:
             print("No suitable datetime column found or specified. Skipping datetime index setting.")
             # If no datetime index, ensure there's a default index for time series operations
             if not isinstance(self.processed_df.index, pd.DatetimeIndex):
                  print("Creating a default DatetimeIndex for time series operations.")
                  # Attempt to create a default hourly index if possible
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

        # Ensure index is DatetimeIndex for time-based fillna and interpolate
        if isinstance(self.processed_df.index, pd.DatetimeIndex):
            # Strategy 1: Forward fill for short gaps (< 3 consecutive hours)
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
                # Be more conservative with target variable
                Q1 = self.processed_df[col].quantile(0.05)
                Q3 = self.processed_df[col].quantile(0.95)
            else:
                Q1 = self.processed_df[col].quantile(0.25)
                Q3 = self.processed_df[col].quantile(0.75)

            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # More conservative multiplier
            upper_bound = Q3 + 3 * IQR

            outliers = (self.processed_df[col] < lower_bound) | (self.processed_df[col] > upper_bound)
            outliers_count = outliers.sum()

            if outliers_count > 0:
                print(f"  {col}: {outliers_count} outliers replaced with NaN")
                self.processed_df.loc[outliers, col] = np.nan
                outliers_replaced += outliers_count

        print(f"Total outliers replaced with NaN: {outliers_replaced}")

        # After replacing outliers with NaN, backfill
        if isinstance(self.processed_df.index, pd.DatetimeIndex):
             print("Backfilling NaN values after outlier replacement...")
             self.processed_df.fillna(method='bfill', inplace=True)
             # Forward fill any remaining NaNs at the beginning
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
        self.processed_df['is_holiday'] = 0  # You can add Pakistan holidays here

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

    def create_future_targets(self, horizons=[24, 48, 72]):
        """Create future target variables for multi-step forecasting."""
        print("\n" + "="*50)
        print("CREATING FUTURE TARGETS")
        print("="*50)
        if self.target_column and self.target_column in self.processed_df.columns:
            for h in horizons:
                self.processed_df[f'aqi_{h}h'] = self.processed_df[self.target_column].shift(-h)
            print(f"Created future targets for horizons: {horizons}")
        else:
            print("Target column not found. Skipping future target creation.")

    def create_lag_features(self, min_lag=72):
        """Create lag features for time series, ensuring no data leakage."""
        print("\n" + "="*50)
        print(f"CREATING LAG FEATURES (min_lag={min_lag}h)")
        print("="*50)

        if not isinstance(self.processed_df.index, pd.DatetimeIndex):
             print("Skipping lag feature creation as index is not a DatetimeIndex.")
             return

        # Define lag periods (in hours) starting from min_lag
        lag_periods = [min_lag, min_lag + 12, min_lag + 24, min_lag + 48, min_lag + 72, min_lag + 96]

        # Create lag features for the original target variable (aqi)
        if self.target_column and self.target_column in self.processed_df.columns:
             for lag in lag_periods:
                  self.processed_df[f'{self.target_column}_lag_{lag}h'] = self.processed_df[self.target_column].shift(lag)
        else:
             print(f"Warning: Target column '{self.target_column}' not found. Skipping target lag features.")

        # Create lag features for pollutants
        for pollutant in self.pollutant_columns:
            if pollutant in self.processed_df.columns:
                for lag in lag_periods:
                    self.processed_df[f'{pollutant}_lag_{lag}h'] = self.processed_df[pollutant].shift(lag)

        # Create lag features for weather variables
        for weather_var in self.weather_columns:
            if weather_var in self.processed_df.columns:
                for lag in lag_periods:
                    self.processed_df[f'{weather_var}_lag_{lag}h'] = self.processed_df[weather_var].shift(lag)


        print(f"Created lag features with periods: {lag_periods}")

    def create_rolling_features(self, min_lag=72):
        """Create rolling statistics features on lagged data to prevent leakage."""
        print("\n" + "="*50)
        print(f"CREATING ROLLING FEATURES (min_lag={min_lag}h)")
        print("="*50)

        if not isinstance(self.processed_df.index, pd.DatetimeIndex):
             print("Skipping rolling feature creation as index is not a DatetimeIndex.")
             return

        # Rolling features for the original aqi column, but on a lagged series
        if self.target_column and self.target_column in self.processed_df.columns:
            aqi_rolling_windows = [24, 48, 72]
            for window in aqi_rolling_windows:
                # Create rolling features on a series that is already lagged by min_lag
                lagged_aqi = self.processed_df[self.target_column].shift(min_lag)
                self.processed_df[f'{self.target_column}_rolling_std_{window}h_lag{min_lag}h'] = lagged_aqi.rolling(window=window).std()
                self.processed_df[f'{self.target_column}_rolling_max_{window}h_lag{min_lag}h'] = lagged_aqi.rolling(window=window).max()

        # Rolling features for pollutants on lagged series
        pollutant_rolling = {
            'pm10': [24],
            'pm2_5': [24],
            'co': [24, 48],
            'no2': [24, 48],
            'so2': [24, 48],
            'o3': [24, 48]
        }

        for pollutant, windows in pollutant_rolling.items():
            if pollutant in self.processed_df.columns:
                for window in windows:
                    lagged_pollutant = self.processed_df[pollutant].shift(min_lag)
                    self.processed_df[f'{pollutant}_rolling_std_{window}h_lag{min_lag}h'] = lagged_pollutant.rolling(window=window).std()

        print(f"Created rolling features on data lagged by {min_lag}h")

    def create_complex_rolling_features(self):
        """Create complex rolling features for lag combinations"""
        pass # This method is now handled by the new rolling features logic


    def create_interaction_features(self):
        """Create interaction features between important variables"""
        print("\n" + "="*50)
        print("CREATING INTERACTION FEATURES")
        print("="*50)

        # Temperature and humidity interaction (heat index proxy)
        if 'temperature' in self.processed_df.columns and 'humidity' in self.processed_df.columns:
            self.processed_df['temp_humidity_interaction'] = (
                self.processed_df['temperature'] * self.processed_df['humidity'] / 100
            )

        # Wind speed and direction features
        if 'wind_speed' in self.processed_df.columns and 'wind_direction' in self.processed_df.columns:
            self.processed_df['wind_u'] = (
                self.processed_df['wind_speed'] * np.cos(np.radians(self.processed_df['wind_direction']))
            )
            self.processed_df['wind_v'] = (
                self.processed_df['wind_speed'] * np.sin(np.radians(self.processed_df['wind_direction']))
            )

        # PM2.5 and PM10 ratio
        if 'pm2_5' in self.processed_df.columns and 'pm10' in self.processed_df.columns:
             self.processed_df['pm25_pm10_ratio'] = (
                self.processed_df['pm2_5'] / (self.processed_df['pm10'] + 1e-6)
             )

        # Rush hour and pollutant interactions
        if 'is_morning_rush' in self.processed_df.columns and 'no2' in self.processed_df.columns:
             self.processed_df['morning_rush_no2'] = (
                self.processed_df['is_morning_rush'] * self.processed_df['no2']
             )

        if 'is_evening_rush' in self.processed_df.columns and 'no2' in self.processed_df.columns:
             self.processed_df['evening_rush_no2'] = (
                self.processed_df['is_evening_rush'] * self.processed_df['no2']
             )

        print("Created interaction features")

    def create_statistical_features(self, min_lag=72):
        """Create statistical features on lagged data."""
        print("\n" + "="*50)
        print(f"CREATING STATISTICAL FEATURES (min_lag={min_lag}h)")
        print("="*50)

        if self.target_column and self.target_column in self.processed_df.columns:
            # Rate of change features for aqi, calculated on lagged data
            rate_changes = [24, 48, 72]
            for hours in rate_changes:
                feature_name = f'{self.target_column}_rate_change_{hours}h_lag{min_lag}h'
                self.processed_df[feature_name] = self.processed_df[self.target_column].shift(min_lag).diff(hours)

            print("Created statistical features for target column on lagged data.")
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

         for col in numeric_cols:
             # Calculate variance, handle potential NaNs after dropna
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
        """
        Reduce multicollinearity
        """
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
            # Find columns highly correlated with the current column
            highly_correlated_cols = upper.index[upper[column] > correlation_threshold].tolist()

            if highly_correlated_cols:
                for corr_col in highly_correlated_cols:
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
            if col != self.target_column:
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
      """
      Enhanced finalize_dataset method that replaces the original dropna() approach
      with smart row dropping based on maximum lag/rolling window.
      This method keeps all columns that are fully preprocessed and have no NaN values.
      """
      print("\n" + "="*50)
      print("SMART FINALIZING DATASET")
      print("="*50)

      # Drop rows with NaN in future targets, as these cannot be used for training
      future_target_cols = [f'aqi_{h}h' for h in [24, 48, 72]]
      self.processed_df.dropna(subset=future_target_cols, inplace=True)


      # Step 1: SMART ROW DROPPING - Replace the original dropna() approach
      print("\nApplying smart row dropping based on maximum lag/rolling window...")

      # Get all column names for analysis (including index if it's named)
      all_columns = self.processed_df.columns.tolist()
      if self.processed_df.index.name:
        all_columns.append(self.processed_df.index.name)

      # Apply smart dropping
      initial_rows = len(self.processed_df)
      self.processed_df, rows_dropped = smart_drop_initial_rows(
        self.processed_df,
        feature_columns=all_columns,
        min_drop_hours=0
      )

      print(f"Smart dropping removed {rows_dropped} initial rows (instead of all NaN rows)")

      # Step 2: Handle any remaining NaN values more conservatively
      remaining_nans_before = self.processed_df.isnull().sum().sum()
      if remaining_nans_before > 0:
        print(f"\nHandling {remaining_nans_before} remaining NaN values in the dataset...")

        # More conservative NaN handling - interpolate first, then fill
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.processed_df[col].isnull().any():
                # Try interpolation first
                self.processed_df[col] = self.processed_df[col].interpolate(
                    method='linear', limit_direction='both'
                )

        # Forward fill any remaining
        self.processed_df = self.processed_df.fillna(method='ffill')
        # Backward fill for any at the beginning
        self.processed_df = self.processed_df.fillna(method='bfill')

        remaining_nans_after = self.processed_df.isnull().sum().sum()
        print(f"NaN values after interpolation and filling: {remaining_nans_after}")

        # Drop columns with remaining NaNs
        if remaining_nans_after > 0:
            print("Dropping columns with remaining unfillable NaN values...")
            cols_before_drop = set(self.processed_df.columns)
            self.processed_df = self.processed_df.dropna(axis=1)
            cols_after_drop = set(self.processed_df.columns)
            dropped_cols = cols_before_drop - cols_after_drop
            print(f"Columns dropped due to unfillable NaNs: {list(dropped_cols)}")


      # Step 3: Update feature_columns and final reporting
      self.feature_columns = [col for col in self.processed_df.columns if col != self.target_column and 'aqi_' not in col]


      # Final check for low variance features
      print("\nFinal check for low variance features...")
      self.remove_low_variance_features()
      self.feature_columns = [col for col in self.processed_df.columns if col != self.target_column and 'aqi_' not in col]


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

        # Calculate correlations with target
        correlations = []
        # Include target column in correlation calculation dataframe
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

        # Sort by absolute correlation
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

        print(f"\nFeatures removed during preprocessing steps:")
        print(f"  Removed due to low variance: {len(self.removed_features['low_variance'])}")
        print(f"  Removed due to high correlation: {len(self.removed_features['high_correlation'])}")

        print(f"\nData quality:")
        print(f"  Missing values: {self.processed_df.isnull().sum().sum()}")
        print(f"  Duplicate rows: {self.processed_df.duplicated().sum()}")

        print(f"\nMemory usage: {self.processed_df.memory_usage().sum() / 1024**2:.2f} MB")


    def run_full_preprocessing(self, correlation_threshold=0.85, variance_threshold=1e-6, vif_threshold=10, dataframe=None, target_schema_columns=None, forecast_horizon=72):
        """Run the complete preprocessing pipeline"""
        print("="*80)
        print("STARTING FULL PREPROCESSING PIPELINE")
        print("="*80)

        # Step 1: Load data
        if not self.load_data(dataframe=dataframe):
            return False

        # Step 2: Clean data
        self.clean_data()

        # Step 3: Create Future Targets
        self.create_future_targets(horizons=[24, 48, 72])

        # Step 4: Feature engineering (create all features that might be needed)
        self.create_time_features()
        self.create_lag_features(min_lag=forecast_horizon)
        self.create_rolling_features(min_lag=forecast_horizon)
        self.create_interaction_features()
        self.create_statistical_features(min_lag=forecast_horizon)

        # Step 5: Reduce multicollinearity
        self.feature_columns = self.reduce_multicollinearity(
            correlation_threshold=correlation_threshold
        )

        # Step 6: Encode and scale
        self.encode_categorical_features()
        self.scale_features()

        # Step 7: Finalize (filter to required columns only and handle NaNs)
        self.enhanced_finalize_dataset()

        # Step 8: Analysis and reporting
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
            # Ensure target column is included if it exists in the processed data
            y_data = None
            if self.target_column and self.target_column in self.processed_df.columns:
                 y_data = self.processed_df[[f'aqi_{h}h' for h in [24, 48, 72]]]
                 # Remove target from feature columns for X
                 X_features = [col for col in self.feature_columns if col != self.target_column and 'aqi_' not in col]
            else:
                 X_features = self.feature_columns
                 if self.target_column:
                    print(f"Warning: Target column '{self.target_column}' not found in processed data after filtering.")


            # Ensure all X_features are actually in the processed_df columns before selecting
            valid_X_features = [col for col in X_features if col in self.processed_df.columns]
            invalid_X_features = [col for col in X_features if col not in self.processed_df.columns]

            if invalid_X_features:
                 # This warning should ideally not happen if filtering in finalize_dataset works correctly
                 print(f"Warning: The following features listed to keep were not found after preprocessing and filtering: {invalid_X_features}")


            return {
                'X': self.processed_df[valid_X_features],
                'y': y_data,
                'feature_names': valid_X_features,
                'target_name': self.target_column,
                'full_data': self.processed_df # Return the full filtered dataframe
            }
        else:
            print("No processed data available. Run preprocessing first.")
            return None