import pandas as pd
import numpy as np
from typing import List, Optional, Dict

class DataQuality:
    @staticmethod
    def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
        """
        Detect outliers using the IQR method. Returns a boolean Series.
        """
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = (df[col] < lower) | (df[col] > upper)
        return outliers

    @staticmethod
    def detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using the Z-score method. Returns a boolean Series.
        """
        z = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        outliers = z.abs() > threshold
        return outliers

    @staticmethod
    def impute_missing(df: pd.DataFrame, cols: List[str], method: str = 'mean') -> pd.DataFrame:
        """
        Impute missing values in specified columns using mean, median, or interpolation.
        """
        df = df.copy()
        for col in cols:
            if method == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif method == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif method == 'ffill':
                df[col] = df[col].fillna(method='ffill')
            elif method == 'bfill':
                df[col] = df[col].fillna(method='bfill')
            else:
                pass  # Unknown method, do nothing
        return df

    @staticmethod
    def check_consistency_across_apis(df: pd.DataFrame, value_cols: List[str], group_cols: List[str]) -> pd.DataFrame:
        """
        Check consistency of values across APIs for the same city/timestamp.
        Returns a DataFrame with max-min difference for each group.
        """
        grouped = df.groupby(group_cols)[value_cols]
        max_df = grouped.max().reset_index()
        min_df = grouped.min().reset_index()
        diff_df = max_df.copy()
        for col in value_cols:
            diff_df[col] = max_df[col] - min_df[col]
        inconsistent_mask = diff_df[value_cols].gt(0).any(axis=1)
        inconsistent = diff_df[inconsistent_mask]
        if inconsistent.empty:
            return pd.DataFrame(columns=diff_df.columns)
        return pd.DataFrame(inconsistent).copy()

    @staticmethod
    def generate_quality_report(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Generate a data quality report: missing, outlier counts, summary stats.
        """
        report = {}
        for col in cols:
            missing = df[col].isna().sum()
            iqr_outliers = DataQuality.detect_outliers_iqr(df, col).sum()
            z_outliers = DataQuality.detect_outliers_zscore(df, col).sum()
            stats = df[col].describe().to_dict()
            report[col] = {
                'missing': missing,
                'iqr_outliers': iqr_outliers,
                'zscore_outliers': z_outliers,
                **stats
            }
        return report

    @staticmethod
    def alert_on_quality_issues(report: Dict[str, Dict[str, float]], missing_thresh: int = 10, outlier_thresh: int = 5):
        """
        Log alerts if missing or outlier counts exceed thresholds.
        """
        for col, stats in report.items():
            if stats['missing'] > missing_thresh:
                print(f"ALERT: {col} has {stats['missing']} missing values!")
            if stats['iqr_outliers'] > outlier_thresh:
                print(f"ALERT: {col} has {stats['iqr_outliers']} IQR outliers!")
            if stats['zscore_outliers'] > outlier_thresh:
                print(f"ALERT: {col} has {stats['zscore_outliers']} Z-score outliers!") 