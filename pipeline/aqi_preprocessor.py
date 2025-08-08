import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import re
from typing import List, Tuple, Optional

warnings.filterwarnings('ignore')

def extract_max_window_from_features(feature_names: List[str]) -> int:
    max_window = 0
    window_pattern = r'(\d+)h'  

    for feature in feature_names:
        matches = re.findall(window_pattern, feature)
        for match in matches:
            window_size = int(match)
            max_window = max(max_window, window_size)

    return max_window

def smart_drop_initial_rows(df: pd.DataFrame,
                           feature_columns: Optional[List[str]] = None,
                           min_drop_hours: int = 0) -> Tuple[pd.DataFrame, int]:
   
    if df.empty:
        print("Warning: Empty dataframe provided.")
        return df, 0

    if feature_columns is None:
        feature_columns = df.columns.tolist()

    max_window = extract_max_window_from_features(feature_columns)

    rows_to_drop = max(max_window, min_drop_hours)

    if rows_to_drop == 0:
        print("No rows need to be dropped.")
        return df, 0

    initial_shape = df.shape

    if rows_to_drop >= len(df):
        print(f"Warning: Trying to drop {rows_to_drop} rows but dataframe only has {len(df)} rows.")
        print("Dropping all but the last row to preserve structure.")
        rows_to_drop = len(df) - 1

    df_processed = df.iloc[rows_to_drop:].copy()

    print(f"\nRow dropping summary:")
    print(f"  Initial shape: {initial_shape}")
    print(f"  Rows dropped: {rows_to_drop}")
    print(f"  Final shape: {df_processed.shape}")
    print(f"  Data retained: {len(df_processed)/len(df)*100:.1f}%")

    return df_processed, rows_to_drop

class AQIDataPreprocessor:

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
        self.original_numeric_features = []
        self.target_schema_columns = None 

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
            print("No file path or DataFrame provided. Creating sample data structure...")
            self.create_sample_data()

        if self.df is not None:
             self.df.columns = self.df.columns.str.lower().str.replace('.', '_', regex=False) 
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

        self.target_column = None 
        for col in self.df.columns:
            if 'aqi' in col.lower(): 
                self.target_column = col
                break

        if self.target_column:
            print(f"Target column: {self.target_column}")
        else:
            print("Warning: No target column ('aqi' or similar) found.")

        self.original_numeric_features = [col for col in self.df.select_dtypes(include=np.number).columns if col != self.datetime_column and col != self.target_column]
        print(f"\nIdentified original numeric features: {self.original_numeric_features}")


    def clean_data(self):
        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)

        if self.df is None:
            print("No data available to clean. Load data first.")
            return

        self.processed_df = self.df.copy()

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
                       self.processed_df.index = pd.date_range(start='2025-04-21', periods=len(self.processed_df), freq='H')
                       print("Default DatetimeIndex created.")
                  except Exception as e:
                       print(f"Could not create a default DatetimeIndex: {e}. Some time-series features might fail.")

        initial_shape = self.processed_df.shape
        self.processed_df = self.processed_df.drop_duplicates()
        print(f"Removed {initial_shape[0] - self.processed_df.shape[0]} duplicate rows")

        print("\nHandling missing values...")
        self.handle_missing_values()

        print("\nRemoving outliers...")
        self.remove_outliers()

        print("\nOptimizing data types...")
        self.optimize_data_types()

        print(f"\nCleaned data shape: {self.processed_df.shape}")

    def handle_missing_values(self):
        """Handle missing values using various strategies"""
        missing_before = self.processed_df.isnull().sum().sum()

        if isinstance(self.processed_df.index, pd.DatetimeIndex):
            for col in self.processed_df.columns:
                if self.processed_df[col].isnull().any():
                    self.processed_df[col] = self.processed_df[col].fillna(method='ffill', limit=2)

            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.processed_df[col].isnull().any():
                    self.processed_df[col] = self.processed_df[col].interpolate(method='time', limit=6)
        else:
            print("Skipping time-based fillna and interpolate as index is not DatetimeIndex.")

        if self.processed_df.isnull().any().any():
            print("Applying KNN imputation for remaining missing values...")
            numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                self.processed_df[numeric_cols] = imputer.fit_transform(self.processed_df[numeric_cols])

        categorical_cols = self.processed_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.processed_df[col].isnull().any():
                mode_value = self.processed_df[col].mode()
                if len(mode_value) > 0:
                    self.processed_df[col] = self.processed_df[col].fillna(mode_value[0])

        missing_after = self.processed_df.isnull().sum().sum()
        print(f"Missing values: {missing_before} -> {missing_after}")

    def remove_outliers(self):
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
        for col in self.processed_df.columns:
            if self.processed_df[col].dtype == 'float64':
                self.processed_df[col] = self.processed_df[col].astype('float32')
            elif self.processed_df[col].dtype == 'int64':
                self.processed_df[col] = self.processed_df[col].astype('int32')

    def create_time_features(self):
        print("\n" + "="*50)
        print("CREATING TIME FEATURES")
        print("="*50)

        if not isinstance(self.processed_df.index, pd.DatetimeIndex):
            print("Skipping time feature creation as index is not a DatetimeIndex.")
            return

        self.processed_df['hour'] = self.processed_df.index.hour
        self.processed_df['day_of_week'] = self.processed_df.index.dayofweek
        self.processed_df['day'] = self.processed_df.index.day
        self.processed_df['weekday'] = self.processed_df.index.dayofweek
        self.processed_df['month'] = self.processed_df.index.month
        self.processed_df['quarter'] = self.processed_df.index.quarter
        self.processed_df['is_weekend'] = (self.processed_df.index.dayofweek >= 5).astype(int)

        self.processed_df['season'] = self.processed_df['month'] % 12 // 3 + 1

        self.processed_df['hour_sin'] = np.sin(2 * np.pi * self.processed_df['hour'] / 24)
        self.processed_df['hour_cos'] = np.cos(2 * np.pi * self.processed_df['hour'] / 24)

        self.processed_df['day_sin'] = np.sin(2 * np.pi * self.processed_df['day_of_week'] / 7)
        self.processed_df['day_cos'] = np.cos(2 * np.pi * self.processed_df['day_of_week'] / 7)
        self.processed_df['month_cos'] = np.cos(2 * np.pi * self.processed_df['month'] / 12)

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
        print("\n" + "="*50)
        print(f"CREATING LAG FEATURES (min_lag={min_lag}h)")
        print("="*50)

        if not isinstance(self.processed_df.index, pd.DatetimeIndex):
             print("Skipping lag feature creation as index is not a DatetimeIndex.")
             return

        lag_periods = [min_lag, min_lag + 12, min_lag + 24, min_lag + 48, min_lag + 72, min_lag + 96]

        if self.target_column and self.target_column in self.processed_df.columns:
             for lag in lag_periods:
                  self.processed_df[f'{self.target_column}_lag_{lag}h'] = self.processed_df[self.target_column].shift(lag)
        else:
             print(f"Warning: Target column '{self.target_column}' not found. Skipping target lag features.")

        for pollutant in self.pollutant_columns:
            if pollutant in self.processed_df.columns:
                for lag in lag_periods:
                    self.processed_df[f'{pollutant}_lag_{lag}h'] = self.processed_df[pollutant].shift(lag)
                    
        for weather_var in self.weather_columns:
            if weather_var in self.processed_df.columns:
                for lag in lag_periods:
                    self.processed_df[f'{weather_var}_lag_{lag}h'] = self.processed_df[weather_var].shift(lag)

        print(f"Created lag features with periods: {lag_periods}")

    def create_rolling_features(self, min_lag=72):
        print("\n" + "="*50)
        print(f"CREATING ROLLING FEATURES (min_lag={min_lag}h)")
        print("="*50)

        if not isinstance(self.processed_df.index, pd.DatetimeIndex):
             print("Skipping rolling feature creation as index is not a DatetimeIndex.")
             return

        if self.target_column and self.target_column in self.processed_df.columns:
            aqi_rolling_windows = [24, 48, 72]
            for window in aqi_rolling_windows:
                # Create rolling features on a series that is already lagged by min_lag
                lagged_aqi = self.processed_df[self.target_column].shift(min_lag)
                self.processed_df[f'{self.target_column}_rolling_std_{window}h_lag{min_lag}h'] = lagged_aqi.rolling(window=window).std()
                self.processed_df[f'{self.target_column}_rolling_max_{window}h_lag{min_lag}h'] = lagged_aqi.rolling(window=window).max()

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

        if 'pm2_5' in self.processed_df.columns and 'pm10' in self.processed_df.columns:
             self.processed_df['pm25_pm10_ratio'] = (
                self.processed_df['pm2_5'] / (self.processed_df['pm10'] + 1e-6)
             )

        print("Created interaction features")

    def create_statistical_features(self, min_lag=72):
        print("\n" + "="*50)
        print(f"CREATING STATISTICAL FEATURES (min_lag={min_lag}h)")
        print("="*50)

        if self.target_column and self.target_column in self.processed_df.columns:
            rate_changes = [24, 48, 72]
            for hours in rate_changes:
                feature_name = f'{self.target_column}_rate_change_{hours}h_lag{min_lag}h'
                self.processed_df[feature_name] = self.processed_df[self.target_column].shift(min_lag).diff(hours)

            print("Created statistical features for target column on lagged data.")
        else:
            print(f"Warning: Target column '{self.target_column}' not found. Skipping statistical features for target column.")


    def remove_low_variance_features(self, threshold=1e-6):
         print("\n" + "="*50)
         print("REMOVING LOW VARIANCE FEATURES")
         print("="*50)

         if self.processed_df is None or self.processed_df.empty:
             print("No data to remove low variance features from.")
             return

         numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
         features_to_drop = []

         for col in numeric_cols:
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

        print("\n" + "="*50)
        print("REDUCING MULTICOLLINEARITY (Correlation)")
        print("="*50)

        if self.processed_df is None or self.processed_df.empty:
            print("No data to reduce multicollinearity from.")
            return []

        numeric_df = self.processed_df.select_dtypes(include=[np.number]).copy()
        if self.target_column in numeric_df.columns:
            features_for_corr = numeric_df.drop(columns=[self.target_column])
        else:
             features_for_corr = numeric_df

        corr_matrix = features_for_corr.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        features_to_drop_corr = set()
        for column in upper.columns:
            highly_correlated_cols = upper.index[upper[column] > correlation_threshold].tolist()

            if highly_correlated_cols:
                for corr_col in highly_correlated_cols:
                    if column not in features_to_drop_corr:
                        features_to_drop_corr.add(column)
                        self.removed_features['high_correlation'].append(column)

        if features_to_drop_corr:
            self.processed_df = self.processed_df.drop(columns=list(features_to_drop_corr), errors='ignore')
            print(f"Removed {len(features_to_drop_corr)} features due to high multicollinearity.")
        else:
            print("No features found with multicollinearity above the threshold to remove.")

        self.feature_columns = [col for col in self.processed_df.columns if col != self.target_column]

        return self.feature_columns

    def encode_categorical_features(self):

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

    def enhanced_finalize_dataset(self):

      print("\n" + "="*50)
      print("SMART FINALIZING DATASET")
      print("="*50)
      future_target_cols = [f'aqi_{h}h' for h in [24, 48, 72]]
      self.processed_df.dropna(subset=future_target_cols, inplace=True)

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

      print(f"Smart dropping removed {rows_dropped} initial rows (instead of all NaN rows)")

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
            print("Dropping columns with remaining unfillable NaN values...")
            cols_before_drop = set(self.processed_df.columns)
            self.processed_df = self.processed_df.dropna(axis=1)
            cols_after_drop = set(self.processed_df.columns)
            dropped_cols = cols_before_drop - cols_after_drop
            print(f"Columns dropped due to unfillable NaNs: {list(dropped_cols)}")

      self.feature_columns = [col for col in self.processed_df.columns if col != self.target_column and 'aqi_' not in col]

      print("\nFinal check for low variance features...")
      self.remove_low_variance_features()
      self.feature_columns = [col for col in self.processed_df.columns if col != self.target_column and 'aqi_' not in col]


      print(f"\nFinal dataset shape after smart processing: {self.processed_df.shape}")
      print(f"Final feature count: {len(self.feature_columns)}")
      print(f"Target column: {self.target_column}")
      print(f"Data retention: {len(self.processed_df)/initial_rows*100:.1f}% of original rows")


    def get_feature_importance_analysis(self):
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

        print(f"\nFeatures removed during preprocessing steps:")
        print(f"  Removed due to low variance: {len(self.removed_features['low_variance'])}")
        print(f"  Removed due to high correlation: {len(self.removed_features['high_correlation'])}")

        print(f"\nData quality:")
        print(f"  Missing values: {self.processed_df.isnull().sum().sum()}")
        print(f"  Duplicate rows: {self.processed_df.duplicated().sum()}")

        print(f"\nMemory usage: {self.processed_df.memory_usage().sum() / 1024**2:.2f} MB")


    def run_full_preprocessing(self, correlation_threshold=0.85, variance_threshold=1e-6, vif_threshold=10, dataframe=None, target_schema_columns=None, forecast_horizon=72):
        print("="*80)
        print("STARTING FULL PREPROCESSING PIPELINE")
        print("="*80)

        if not self.load_data(dataframe=dataframe):
            return False

        self.clean_data()

        self.create_future_targets(horizons=[24, 48, 72])

        self.create_time_features()
        self.create_lag_features(min_lag=forecast_horizon)
        self.create_rolling_features(min_lag=forecast_horizon)
        self.create_interaction_features()
        self.create_statistical_features(min_lag=forecast_horizon)

        self.feature_columns = self.reduce_multicollinearity(
            correlation_threshold=correlation_threshold
        )
        self.encode_categorical_features()

        self.enhanced_finalize_dataset()

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
                 y_data = self.processed_df[[f'aqi_{h}h' for h in [24, 48, 72]]]
                 X_features = [col for col in self.feature_columns if col != self.target_column and 'aqi_' not in col]
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
