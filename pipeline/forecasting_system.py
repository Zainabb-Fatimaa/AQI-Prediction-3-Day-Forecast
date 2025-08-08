import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import pickle
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AQIForecastingSystem:

    def __init__(self, horizon_df, target_column, horizon_hours, model_save_path="./aqi_models"):
        self.df = horizon_df.copy()
        self.target_column = target_column
        self.horizon_hours = horizon_hours
        self.model_save_path = model_save_path

        if 'event_time' in self.df.columns:
            self.df['event_time'] = pd.to_datetime(self.df['event_time'])
            self.df = self.df.sort_values('event_time').reset_index(drop=True)

        self.models = {}
        self.ensemble_models = {}
        self.selected_features = []  
        self.results = {}
        self.predictions = {}
        self.ensemble_predictions = {}

        os.makedirs(self.model_save_path, exist_ok=True)

        print(f" Initialized for {horizon_hours}h horizon")
        print(f" Dataset shape: {self.df.shape}")
        print(f" Target column: {target_column}")
        print(f" Model save path: {self.model_save_path}")

    def validate_data_quality(self):
        """Validate data quality for the specific horizon"""
        print(f" Validating data quality for {self.horizon_hours}h horizon...")

        issues = []

        if self.target_column not in self.df.columns:
            issues.append(f"Target column '{self.target_column}' not found in dataframe")
            return issues

        target_data = self.df[self.target_column].dropna()
        if len(target_data) == 0:
            issues.append(f"{self.target_column} has no valid data")
            return issues

        target_std = target_data.std()
        target_mean = target_data.mean()

        print(f"   {self.target_column}: mean={target_mean:.2f}, std={target_std:.2f}, count={len(target_data)}")

        if target_std < 1:
            issues.append(f"{self.target_column} has very low variance ({target_std:.3f})")

        if (target_data < 0).any():
            issues.append(f"{self.target_column} has {(target_data < 0).sum()} negative values")

        if target_data.max() > 500:  
            issues.append(f"{self.target_column} has suspiciously high values (max: {target_data.max():.1f})")

        feature_cols = [col for col in self.df.columns if col not in [self.target_column, 'event_time', 'date', 'unique_id', 'aqi']]
        print(f"   Available features: {len(feature_cols)}")

        if len(feature_cols) == 0:
            issues.append("No feature columns found")

        high_missing_features = []
        for col in feature_cols:
            missing_pct = self.df[col].isna().sum() / len(self.df)
            if missing_pct > 0.8:
                high_missing_features.append((col, missing_pct))

        if high_missing_features:
            issues.append(f"{len(high_missing_features)} features have >80% missing values")

        if issues:
            print(" Data quality issues found:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print("Data quality checks passed")

        return issues

    def handle_outliers(self, data, method='iqr', cap_percentile=0.01):
        print(" Handling outliers in features...")

        data_cleaned = data.copy()
        outlier_counts = {}

        for col in data_cleaned.select_dtypes(include=[np.number]).columns:
            if col == self.target_column:
                continue  # Don't modify target column

            original_values = data_cleaned[col].dropna()

            if len(original_values) == 0:
                continue

            if method == 'iqr':
                Q1 = original_values.quantile(0.25)
                Q3 = original_values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (data_cleaned[col] < lower_bound) | (data_cleaned[col] > upper_bound)
                outlier_count = outlier_mask.sum()

                data_cleaned[col] = data_cleaned[col].clip(lower=lower_bound, upper=upper_bound)

            elif method == 'winsorize':
                lower_percentile = cap_percentile
                upper_percentile = 1 - cap_percentile

                lower_bound = original_values.quantile(lower_percentile)
                upper_bound = original_values.quantile(upper_percentile)

                outlier_mask = (data_cleaned[col] < lower_bound) | (data_cleaned[col] > upper_bound)
                outlier_count = outlier_mask.sum()

                data_cleaned[col] = data_cleaned[col].clip(lower=lower_bound, upper=upper_bound)

            if outlier_count > 0:
                outlier_counts[col] = outlier_count

        if outlier_counts:
            print(f"   Handled outliers in {len(outlier_counts)} features")
            top_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top outlier features: {top_outliers}")

        return data_cleaned

    def mutual_info_feature_selection(self, X, y, k=15):
        """Select features using mutual information"""
        selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        scores = selector.scores_[selector.get_support()]

        return selected_features, scores

    def rf_importance_feature_selection(self, X, y, k=15):
        """Select features using Random Forest importance"""
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        importances = rf.feature_importances_
        feature_importance_pairs = list(zip(X.columns, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

        selected_features = [pair[0] for pair in feature_importance_pairs[:k]]
        selected_scores = [pair[1] for pair in feature_importance_pairs[:k]]

        return selected_features, selected_scores

    def cross_validated_feature_selection(self, X, y, n_splits=3, max_features=15):
        print(f"    Hybrid feature selection for {self.horizon_hours}h horizon...")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        mi_score_total = {}
        rf_score_total = {}
        fold_counts = {}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train = X.iloc[train_idx].fillna(X.median())
            y_train = y.iloc[train_idx]

            try:
                mi_features, mi_scores = self.mutual_info_feature_selection(X_train, y_train, k=max_features)
                for feat, score in zip(mi_features, mi_scores):
                    mi_score_total[feat] = mi_score_total.get(feat, 0) + score
                    fold_counts[feat] = fold_counts.get(feat, 0) + 1
            except Exception as e:
                print(f"      MI selection failed in fold {fold}: {e}")

            try:
                rf_features, rf_scores = self.rf_importance_feature_selection(X_train, y_train, k=max_features)
                for feat, score in zip(rf_features, rf_scores):
                    rf_score_total[feat] = rf_score_total.get(feat, 0) + score
                    fold_counts[feat] = fold_counts.get(feat, 0) + 1
            except Exception as e:
                print(f"      RF selection failed in fold {fold}: {e}")

        all_features = set(mi_score_total.keys()) | set(rf_score_total.keys())
        hybrid_scores = {}
        for feat in all_features:
            mi = mi_score_total.get(feat, 0)
            rf = rf_score_total.get(feat, 0)
            mi_norm = mi / n_splits
            rf_norm = rf / n_splits
            hybrid_scores[feat] = (mi_norm + rf_norm) / 2

        sorted_features = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        final_features = [f for f, _ in sorted_features[:max_features]]

        print(f"      Selected {len(final_features)} features")
        print(f"      Top 5 features: {final_features[:5]}")

        return final_features, {
            'hybrid_scores': hybrid_scores,
            'fold_counts': fold_counts
        }

    def feature_selection_pipeline(self, max_features=15):
        """Complete feature selection pipeline for the specific horizon"""
        print(f" Starting feature selection for {self.horizon_hours}h horizon...")

        exclude_cols = [self.target_column, "date", "event_time", "unique_id"]
        all_features = [col for col in self.df.columns if col not in exclude_cols]

        if not all_features:
            print(f" No features found for selection")
            self.selected_features = []
            return {}

        print(f"   Initial features: {len(all_features)}")

        filtered_features = []
        for feature in all_features:
            missing_pct = self.df[feature].isna().sum() / len(self.df)
            if missing_pct < 0.5:  
                filtered_features.append(feature)

        print(f"   After missing value filter: {len(filtered_features)} features")

        if not filtered_features:
            print(f" No features passed missing value filter")
            self.selected_features = []
            return {}

        low_var_threshold = 1e-6
        variance_filtered_features = []

        for feature in filtered_features:
            if pd.api.types.is_numeric_dtype(self.df[feature]):
                feature_var = self.df[feature].var()
                if feature_var > low_var_threshold:
                    variance_filtered_features.append(feature)
            else:
                variance_filtered_features.append(feature)

        print(f"   After variance filter: {len(variance_filtered_features)} features")

        if not variance_filtered_features:
            print(f" No features passed variance filter")
            self.selected_features = []
            return {}

        # Step 3: Handle outliers
        feature_data = self.df[variance_filtered_features + [self.target_column]].copy()
        feature_data_clean = self.handle_outliers(feature_data[variance_filtered_features])
        feature_data[variance_filtered_features] = feature_data_clean

        numeric_features = feature_data[variance_filtered_features].select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_features) > 1:
            corr_data = feature_data[numeric_features].fillna(method='ffill').fillna(method='bfill')
            corr_matrix = corr_data.corr().abs()

            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            high_corr_features = []
            for col in upper_triangle.columns:
                high_corr_cols = upper_triangle[col][upper_triangle[col] > 0.85].index.tolist()
                if high_corr_cols:
                    target_corrs = feature_data[numeric_features + [self.target_column]].corr()[self.target_column].abs()
                    for corr_col in high_corr_cols:
                        if target_corrs[col] >= target_corrs[corr_col]:
                            if corr_col not in high_corr_features:
                                high_corr_features.append(corr_col)
                        else:
                            if col not in high_corr_features:
                                high_corr_features.append(col)

            correlation_filtered_features = [f for f in variance_filtered_features if f not in high_corr_features]
            print(f"   After correlation filter: {len(correlation_filtered_features)} features (removed {len(high_corr_features)})")
        else:
            correlation_filtered_features = variance_filtered_features

        if not correlation_filtered_features:
            print(f" No features passed correlation filter")
            self.selected_features = []
            return {}

        # Step 5: Cross-validated feature selection
        X_features = feature_data[correlation_filtered_features]
        y_target = feature_data[self.target_column]

        valid_rows = ~y_target.isna()
        X_features = X_features[valid_rows]
        y_target = y_target[valid_rows]

        if len(X_features) < 100:
            print(f" Insufficient clean data: {len(X_features)} rows")
            self.selected_features = []
            return {}

        try:
            selected_features, selection_details = self.cross_validated_feature_selection(
                X_features, y_target, max_features=max_features
            )

            self.selected_features = selected_features

            feature_selection_details = {
                'initial_features': len(all_features),
                'after_missing_filter': len(filtered_features),
                'after_variance_filter': len(variance_filtered_features),
                'after_correlation_filter': len(correlation_filtered_features),
                'final_selected': len(selected_features),
                'selection_details': selection_details,
                'selected_features': selected_features
            }

            print(f"    Final selected features: {len(selected_features)}")
            if selected_features:
                print(f"    Top 5 features: {selected_features[:5]}")

            return feature_selection_details

        except Exception as e:
            print(f" Error in feature selection: {e}")
            self.selected_features = []
            return {}

    def check_data_leakage(self):
        """Check for data leakage in selected features"""
        print(" Checking for data leakage...")

        if not self.selected_features:
            print("   No features selected to check")
            return {}

        target_data = self.df[self.selected_features + [self.target_column]].dropna()
        if len(target_data) == 0:
            print(f"     No clean data for correlation analysis")
            return {}

        correlations = target_data[self.selected_features].corrwith(target_data[self.target_column]).abs()
        high_corr_features = correlations[correlations > 0.85].index.tolist()

        suspicious_features = []
        for feature in self.selected_features:
            feature_lower = feature.lower()
            if any(suspicious in feature_lower for suspicious in ['future', 'ahead', 'target']):
                suspicious_features.append(feature)

        leakage_results = {
            'high_correlation_features': high_corr_features,
            'suspicious_name_features': suspicious_features,
            'max_correlation': correlations.max() if not correlations.empty else 0,
            'mean_correlation': correlations.mean() if not correlations.empty else 0
        }

        if high_corr_features or suspicious_features:
            print(f"      Potential leakage detected:")
            if high_corr_features:
                print(f"       High correlation (>0.95): {high_corr_features}")
            if suspicious_features:
                print(f"       Suspicious names: {suspicious_features}")
        else:
            print(f"      No obvious leakage detected")
            print(f"       Max correlation: {correlations.max():.3f}")

        return leakage_results

    def initialize_models(self):
        """Initialize all tree-based models"""
        print(" Initializing tree-based models...")

        self.model_configs = {
            'CatBoost': CatBoostRegressor(
                iterations=672,
                learning_rate=0.018212208878911028,
                depth=7,
                l2_leaf_reg=1.8344212761367933,
                early_stopping_rounds=50,
                verbose=False,
                subsample=0.6076978090537759,
                random_seed=42,
            ),
            'XGBoost': XGBRegressor(
                n_estimators=738,
                learning_rate=0.050902082752918125,
                max_depth=6,
                reg_alpha=3.623661367173843,
                reg_lambda=2.3632955886312637,
                early_stopping_rounds=50,
                random_state=42,
                subsample=0.728994202987864,
                colsample_bytree=0.7490071788934142,
                verbosity=0
            ),
            'LightGBM': LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                early_stopping_rounds=50,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=-1
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=251,
                max_depth=9,
                min_samples_split=10,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
                max_features=0.6449383229836702
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=115,
                learning_rate=0.1436659003499937,
                max_depth=3,
                min_samples_split=7,
                min_samples_leaf=2,
                random_state=42
            ),
            'DecisionTree': DecisionTreeRegressor(
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        }

        self.ensemble_configs = {
            'Ensemble_CBX_ET': ['CatBoost', 'XGBoost', 'ExtraTrees']
        }

    def create_ensemble_predictions(self, fold_predictions):
        """Create ensemble predictions by averaging specified models"""
        ensemble_preds = {}

        for ensemble_name, model_list in self.ensemble_configs.items():
            model_preds = []
            for model_name in model_list:
                if model_name in fold_predictions:
                    model_preds.append(fold_predictions[model_name])

            if len(model_preds) > 0:
                ensemble_pred = np.mean(model_preds, axis=0)
                ensemble_preds[ensemble_name] = ensemble_pred

        return ensemble_preds

    def calculate_metrics(self, y_true, y_pred):
        try:
            if len(y_true) == 0 or len(y_pred) == 0:
                return {
                    'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf,
                    'mse': np.inf, 'mape': np.inf, 'std': 0, 'mean': 0
                }

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if not valid_mask.any():
                return {
                    'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf,
                    'mse': np.inf, 'mape': np.inf, 'std': 0, 'mean': 0
                }

            y_true_clean = y_true[valid_mask]
            y_pred_clean = y_pred[valid_mask]

            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            r2 = r2_score(y_true_clean, y_pred_clean)

            mape_mask = y_true_clean != 0
            if mape_mask.any():
                mape = np.mean(np.abs((y_true_clean[mape_mask] - y_pred_clean[mape_mask]) / y_true_clean[mape_mask])) * 100
            else:
                mape = np.inf

            return {
                'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2),
                'mse': float(mse), 'mape': float(mape),
                'std': float(np.std(y_pred_clean)), 'mean': float(np.mean(y_pred_clean))
            }

        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return {
                'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf,
                'mse': np.inf, 'mape': np.inf, 'std': 0, 'mean': 0
            }

    def sliding_window_validation(self, train_size=8*7*24, test_size=4*7*24, step_size=2*24):
        print(f" Starting sliding window validation for {self.horizon_hours}h horizon...")
        print(f" Parameters: train_size={train_size}, test_size={test_size}, step_size={step_size}")

        if not self.selected_features:
            print(" No features selected! Run feature selection first.")
            return

        valid_features = [f for f in self.selected_features if f in self.df.columns]
        if not valid_features:
            print(" No valid selected features found in dataframe")
            return

        total_samples = len(self.df)
        print(f" Total samples: {total_samples}, Using {len(valid_features)} features")

        windows = []
        train_start = 0

        while train_start + train_size + test_size <= total_samples:
            train_end = train_start + train_size
            test_start = train_end
            test_end = min(test_start + test_size, total_samples)

            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })

            train_start += step_size

        if not windows:
            print(" Not enough data for sliding window validation!")
            print(f" Required: {train_size + test_size} samples, Available: {total_samples}")
            return

        print(f" Created {len(windows)} sliding windows")

        self.results = {model_name: [] for model_name in self.model_configs.keys()}
        self.predictions = {model_name: [] for model_name in self.model_configs.keys()}

        ensemble_names = list(self.ensemble_configs.keys())
        self.ensemble_results = {ensemble_name: [] for ensemble_name in ensemble_names}
        self.ensemble_predictions = {ensemble_name: [] for ensemble_name in ensemble_names}

        self.final_models = {model_name: None for model_name in self.model_configs.keys()}

        for fold, window in enumerate(windows):
            train_start, train_end, test_start, test_end = window['train_start'], window['train_end'], window['test_start'], window['test_end']

            print(f"Window {fold + 1}/{len(windows)}: Train[{train_start}:{train_end}] ({train_end-train_start} samples), Test[{test_start}:{test_end}] ({test_end-test_start} samples)")

            try:
                df_clean = self.df[valid_features + [self.target_column]].copy()
                df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')

                X_train = df_clean.iloc[train_start:train_end][valid_features]
                X_test = df_clean.iloc[test_start:test_end][valid_features]
                y_train = df_clean.iloc[train_start:train_end][self.target_column]
                y_test = df_clean.iloc[test_start:test_end][self.target_column]

                fold_predictions = {}

                for model_name, model_config in self.model_configs.items():
                    try:
                        if model_name in ['CatBoost', 'XGBoost', 'LightGBM']:
                            model = model_config.__class__(**model_config.get_params())

                            if len(X_train) > 100:
                                val_size = max(int(0.15 * len(X_train)), 50)
                                X_val, y_val = X_train.iloc[-val_size:], y_train.iloc[-val_size:]
                                X_train_sub, y_train_sub = X_train.iloc[:-val_size], y_train.iloc[:-val_size]

                                if model_name == 'CatBoost':
                                    model.fit(X_train_sub, y_train_sub, eval_set=(X_val, y_val))
                                elif model_name == 'XGBoost':
                                    model.fit(X_train_sub, y_train_sub, eval_set=[(X_val, y_val)])
                                else:  
                                    model.fit(X_train_sub, y_train_sub, eval_set=[(X_val, y_val)])
                            else:
                                model.fit(X_train, y_train)
                        else:
                            model = model_config.__class__(**model_config.get_params())
                            model.fit(X_train, y_train)

                        train_pred = model.predict(X_train)
                        test_pred = model.predict(X_test)

                        fold_predictions[model_name] = test_pred

                        train_metrics = self.calculate_metrics(y_train, train_pred)
                        test_metrics = self.calculate_metrics(y_test, test_pred)

                        fold_result = {
                            'fold': fold,
                            'window': window,
                            'train_size': len(X_train),
                            'test_size': len(X_test),
                            'train_metrics': train_metrics,
                            'test_metrics': test_metrics,
                            'train_pred': train_pred,
                            'test_pred': test_pred,
                            'y_train': y_train.values,
                            'y_test': y_test.values,
                        }

                        self.results[model_name].append(fold_result)
                        self.predictions[model_name].extend(test_pred)

                        if fold == len(windows) - 1:
                            self.final_models[model_name] = model

                    except Exception as e:
                        print(f"    Error training {model_name} in fold {fold}: {str(e)}")
                        continue

                ensemble_preds = self.create_ensemble_predictions(fold_predictions)

                for ensemble_name, ensemble_pred in ensemble_preds.items():
                    try:
                        ensemble_metrics = self.calculate_metrics(y_test, ensemble_pred)

                        ensemble_fold_result = {
                            'fold': fold,
                            'window': window,
                            'train_size': len(X_train),
                            'test_size': len(X_test),
                            'test_metrics': ensemble_metrics,
                            'test_pred': ensemble_pred,
                            'y_test': y_test.values,
                            'component_models': self.ensemble_configs[ensemble_name],
                        }

                        self.ensemble_results[ensemble_name].append(ensemble_fold_result)
                        self.ensemble_predictions[ensemble_name].extend(ensemble_pred)

                    except Exception as e:
                        print(f"    Error creating ensemble {ensemble_name} in fold {fold}: {str(e)}")
                        continue

            except Exception as e:
                print(f"    Error processing fold {fold}: {str(e)}")
                continue

        print(f" Completed {len(windows)} windows of sliding validation")

    def train_final_models(self):
        print(f" Training final models for {self.horizon_hours}h horizon...")

        if not self.selected_features:
            print(" No features selected! Run feature selection first.")
            return

        valid_features = [f for f in self.selected_features if f in self.df.columns]
        if not valid_features:
            print(" No valid selected features found in dataframe")
            return

        self.deployment_models = {model_name: None for model_name in self.model_configs.keys()}

        try:
            df_clean = self.df[valid_features + [self.target_column]].copy()
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')

            X_full = df_clean[valid_features]
            y_full = df_clean[self.target_column]

            for model_name, model_config in self.model_configs.items():
                try:
                    model = model_config.__class__(**model_config.get_params())

                    if model_name in ['CatBoost', 'XGBoost', 'LightGBM']:
                        val_size = max(int(0.2 * len(X_full)), 100)
                        X_val, y_val = X_full.iloc[-val_size:], y_full.iloc[-val_size:]
                        X_train_sub, y_train_sub = X_full.iloc[:-val_size], y_full.iloc[:-val_size]

                        if model_name == 'CatBoost':
                            model.fit(X_train_sub, y_train_sub, eval_set=(X_val, y_val))
                        elif model_name == 'XGBoost':
                            model.fit(X_train_sub, y_train_sub, eval_set=[(X_val, y_val)])
                        else:  
                            model.fit(X_train_sub, y_train_sub, eval_set=[(X_val, y_val)])
                    else:
                        model.fit(X_full, y_full)

                    self.deployment_models[model_name] = model

                    print(f"    {model_name} trained ({len(valid_features)} features)")

                except Exception as e:
                    print(f"    Error training final {model_name}: {str(e)}")
                    continue

        except Exception as e:
            print(f"    Error preparing data: {str(e)}")

        print(" Final model training completed")

    def save_models(self, save_metadata=True):
        print("Saving models and metadata...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = os.path.join(self.model_save_path, f"aqi_models_{self.horizon_hours}h_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        models_dir = os.path.join(save_dir, "individual_models")
        os.makedirs(models_dir, exist_ok=True)

        for model_name in self.model_configs.keys():
            if model_name in self.deployment_models and self.deployment_models[model_name] is not None:
                model = self.deployment_models[model_name]
                model_dir = os.path.join(models_dir, model_name)
                os.makedirs(model_dir, exist_ok=True)

                try:
                    if model_name == 'CatBoost':
                        model.save_model(os.path.join(model_dir, f"{model_name}_{self.horizon_hours}h.cbm"))
                    elif model_name == 'XGBoost':
                        joblib.dump(model, os.path.join(model_dir, f"{model_name}_{self.horizon_hours}h.joblib"))
                    elif model_name == 'LightGBM':
                        joblib.dump(model, os.path.join(model_dir, f"{model_name}_{self.horizon_hours}h.joblib"))
                    else:
                        joblib.dump(model, os.path.join(model_dir, f"{model_name}_{self.horizon_hours}h.pkl"))

                    print(f"    Saved {model_name}")

                except Exception as e:
                    print(f"    Error saving {model_name}: {str(e)}")

        ensemble_dir = os.path.join(save_dir, "ensemble_models")
        os.makedirs(ensemble_dir, exist_ok=True)

        ensemble_metadata = {
            'ensemble_configs': self.ensemble_configs,
            'component_models': list(self.model_configs.keys()),
            'horizon_hours': self.horizon_hours,
            'ensemble_method': 'simple_average'
        }

        with open(os.path.join(ensemble_dir, "ensemble_config.json"), 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)

        if save_metadata:
            metadata = {
                'selected_features': self.selected_features,
                'target_column': self.target_column,
                'horizon_hours': self.horizon_hours,
                'model_configs': {name: str(config.get_params()) for name, config in self.model_configs.items()},
                'ensemble_configs': self.ensemble_configs,
                'dataset_shape': self.df.shape,
                'training_timestamp': timestamp,
                'validation_method': 'sliding_window'
            }

            with open(os.path.join(save_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)

        with open(os.path.join(save_dir, "selected_features.json"), 'w') as f:
            json.dump(self.selected_features, f, indent=2)

        print(f" All models and metadata saved to: {save_dir}")
        return save_dir

    def load_models(self, model_dir):
        print(f" Loading models from: {model_dir}")

        metadata_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.selected_features = metadata.get('selected_features', [])
            self.target_column = metadata.get('target_column', '')
            self.horizon_hours = metadata.get('horizon_hours', 0)
            print(f"    Loaded metadata for {self.horizon_hours}h horizon")

        features_path = os.path.join(model_dir, "selected_features.json")
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.selected_features = json.load(f)

        models_dir = os.path.join(model_dir, "individual_models")
        self.loaded_models = {}

        for model_name in ['CatBoost', 'XGBoost', 'LightGBM', 'RandomForest',
                          'ExtraTrees', 'GradientBoosting', 'DecisionTree']:
            model_path = os.path.join(models_dir, model_name)
            if os.path.exists(model_path):
                try:
                    if model_name == 'CatBoost':
                        model_file = os.path.join(model_path, f"{model_name}_{self.horizon_hours}h.cbm")
                        if os.path.exists(model_file):
                            model = CatBoostRegressor()
                            model.load_model(model_file)
                            self.loaded_models[model_name] = model
                    elif model_name == 'XGBoost':
                        joblib_file = os.path.join(model_path, f"{model_name}_{self.horizon_hours}h.joblib")
                        json_file = os.path.join(model_path, f"{model_name}_{self.horizon_hours}h.json")

                        if os.path.exists(joblib_file):
                            model = joblib.load(joblib_file)
                        elif os.path.exists(json_file):
                            model = XGBRegressor()
                            model.load_model(json_file)
                        else:
                            raise FileNotFoundError(f"XGBoost model not found at {joblib_file} or {json_file}")

                        self.loaded_models[model_name] = model

                    elif model_name == 'LightGBM':
                        joblib_file = os.path.join(model_path, f"{model_name}_{self.horizon_hours}h.joblib")
                        txt_file = os.path.join(model_path, f"{model_name}_{self.horizon_hours}h.txt")

                        import lightgbm as lgb

                        if os.path.exists(joblib_file):
                            model = joblib.load(joblib_file)
                        elif os.path.exists(txt_file):
                            model = lgb.Booster(model_file=txt_file)
                        else:
                            raise FileNotFoundError(f"LightGBM model not found at {joblib_file} or {txt_file}")

                        self.loaded_models[model_name] = model

                    else:
                        model_file = os.path.join(model_path, f"{model_name}_{self.horizon_hours}h.pkl")
                        if os.path.exists(model_file):
                            model = joblib.load(model_file)
                            self.loaded_models[model_name] = model

                    if model_name in self.loaded_models:
                        print(f"    Loaded {model_name}")

                except Exception as e:
                    print(f"    Error loading {model_name}: {str(e)}")

        ensemble_config_path = os.path.join(model_dir, "ensemble_models", "ensemble_config.json")
        if os.path.exists(ensemble_config_path):
            with open(ensemble_config_path, 'r') as f:
                ensemble_data = json.load(f)
            self.ensemble_configs = ensemble_data['ensemble_configs']
            print(f"   Loaded ensemble configurations")

        print("Model loading completed")

    def predict_with_ensemble(self, X):
        """Make predictions using individual models and ensembles"""
        if not hasattr(self, 'loaded_models'):
            print("No models loaded. Please load models first using load_models()")
            return None

        if not self.selected_features:
            print("No selected features loaded. Please load models first using load_models()")
            return None

        predictions = {}

        for model_name, model in self.loaded_models.items():
            try:
                if isinstance(X, pd.DataFrame):
                    available_features = [f for f in self.selected_features if f in X.columns]
                    if len(available_features) != len(self.selected_features):
                        missing_features = set(self.selected_features) - set(available_features)
                        print(f"Warning: Missing features for {model_name}: {missing_features}")
                        if not available_features:
                            continue
                    X_model = X[self.selected_features]
                else:
                    X_model = X

                if model_name == 'LightGBM' and hasattr(model, 'predict'):
                    pred = model.predict(X_model)
                else:
                    pred = model.predict(X_model)

                predictions[model_name] = pred

            except Exception as e:
                print(f"Error predicting with {model_name}: {str(e)}")
                continue

        ensemble_predictions = {}
        for ensemble_name, model_list in self.ensemble_configs.items():
            model_preds = []

            for model_name in model_list:
                if model_name in predictions:
                    model_preds.append(predictions[model_name])

            if len(model_preds) > 0:
                ensemble_pred = np.mean(model_preds, axis=0)
                ensemble_predictions[ensemble_name] = ensemble_pred

        all_predictions = {**predictions, **ensemble_predictions}

        return all_predictions

    def evaluate_generalization(self):
        print(f"Evaluating generalization for {self.horizon_hours}h horizon...")

        evaluation_results = {}

        for model_name in self.model_configs.keys():
            if not self.predictions.get(model_name):
                continue

            preds = np.array(self.predictions[model_name])

            std_pred = np.std(preds)

            fold_results = self.results.get(model_name, [])
            if fold_results:
                train_rmse_list = [fold['train_metrics']['rmse'] for fold in fold_results]
                test_rmse_list = [fold['test_metrics']['rmse'] for fold in fold_results]
                train_mae_list = [fold['train_metrics']['mae'] for fold in fold_results]
                test_mae_list = [fold['test_metrics']['mae'] for fold in fold_results]
                train_r2_list = [fold['train_metrics']['r2'] for fold in fold_results]
                test_r2_list = [fold['test_metrics']['r2'] for fold in fold_results]
                train_mse_list = [fold['train_metrics']['mse'] for fold in fold_results]
                test_mse_list = [fold['test_metrics']['mse'] for fold in fold_results]
                train_mape_list = [fold['train_metrics']['mape'] for fold in fold_results]
                test_mape_list = [fold['test_metrics']['mape'] for fold in fold_results]

                avg_train_rmse = np.mean(train_rmse_list)
                avg_test_rmse = np.mean(test_rmse_list)
                avg_train_mae = np.mean(train_mae_list)
                avg_test_mae = np.mean(test_mae_list)
                avg_train_r2 = np.mean(train_r2_list)
                avg_test_r2 = np.mean(test_r2_list)
                avg_train_mse = np.mean(train_mse_list)
                avg_test_mse = np.mean(test_mse_list)
                avg_train_mape = np.mean(train_mape_list)
                avg_test_mape = np.mean(test_mape_list)

                train_sizes = [fold['train_size'] for fold in fold_results]
                avg_train_size = np.mean(train_sizes)
                final_train_size = train_sizes[-1] if train_sizes else 0
            else:
                avg_train_rmse = avg_test_rmse = avg_train_r2 = avg_test_r2 = 0
                avg_train_mae = avg_test_mae = avg_train_mse = avg_test_mse = 0
                avg_train_mape = avg_test_mape = 0
                avg_train_size = final_train_size = 0

            std_check = std_pred > 3
            feature_count = len(self.selected_features)

            evaluation_results[model_name] = {
                'std_pred': std_pred,
                'avg_train_rmse': avg_train_rmse,
                'avg_test_rmse': avg_test_rmse,
                'avg_train_r2': avg_train_r2,
                'avg_test_r2': avg_test_r2,
                'avg_train_mse': avg_train_mse,
                'avg_test_mse': avg_test_mse,
                'avg_train_mae': avg_train_mae,
                'avg_test_mae': avg_test_mae,
                'avg_train_mape': avg_train_mape,
                'avg_test_mape': avg_test_mape,
                'avg_train_size': avg_train_size,
                'final_train_size': final_train_size,
                'feature_count': feature_count,
                'std_check_passed': std_check,
                'prediction_quality': 'Good' if std_check else 'Poor'
            }
        if hasattr(self, 'ensemble_predictions'):
            for ensemble_name in self.ensemble_predictions.keys():
                if not self.ensemble_predictions.get(ensemble_name):
                    continue

                preds = np.array(self.ensemble_predictions[ensemble_name])
                std_pred = np.std(preds)
                fold_results = self.ensemble_results.get(ensemble_name, [])
                if fold_results:
                    test_rmse_list = [fold['test_metrics']['rmse'] for fold in fold_results]
                    test_r2_list = [fold['test_metrics']['r2'] for fold in fold_results]
                    test_mae_list = [fold['test_metrics']['mae'] for fold in fold_results]
                    test_mse_list = [fold['test_metrics']['mse'] for fold in fold_results]
                    test_mape_list = [fold['test_metrics']['mape'] for fold in fold_results]

                    avg_test_rmse = np.mean(test_rmse_list)
                    avg_test_r2 = np.mean(test_r2_list)
                    avg_test_mae = np.mean(test_mae_list)
                    avg_test_mse = np.mean(test_mse_list)
                    avg_test_mape = np.mean(test_mape_list)

                    train_sizes = [fold['train_size'] for fold in fold_results]
                    avg_train_size = np.mean(train_sizes)
                    final_train_size = train_sizes[-1] if train_sizes else 0
                else:
                    avg_test_rmse = avg_test_r2 = avg_test_mae = avg_test_mse = avg_test_mape = 0
                    avg_train_size = final_train_size = 0

                std_check = std_pred > 3
                feature_count = len(self.selected_features)

                evaluation_results[ensemble_name] = {
                    'std_pred': std_pred,
                    'avg_test_rmse': avg_test_rmse,
                    'avg_test_r2': avg_test_r2,
                    'avg_test_mae': avg_test_mae,
                    'avg_test_mse': avg_test_mse,
                    'avg_test_mape': avg_test_mape,
                    'avg_train_size': avg_train_size,
                    'final_train_size': final_train_size,
                    'feature_count': feature_count,
                    'std_check_passed': std_check,
                    'prediction_quality': 'Good' if std_check else 'Poor',
                    'component_models': self.ensemble_configs[ensemble_name]
                }

        return evaluation_results

    def generate_report(self):
        """Generate comprehensive performance report"""
        print(" Generating performance report...")

        evaluation_results = self.evaluate_generalization()

        print("\n" + "="*80)
        print(f"AQI FORECASTING SYSTEM PERFORMANCE REPORT - {self.horizon_hours}H HORIZON")
        print("="*80)

        print(f"\n Configuration:")
        print(f"   Forecast horizon: {self.horizon_hours} hours")
        print(f"   Target column: {self.target_column}")
        print(f"   Selected features: {len(self.selected_features)}")
        print(f"   Individual models: {len(self.model_configs)}")
        print(f"   Ensemble models: {len(self.ensemble_configs)}")
        print(f"   Validation method: Sliding Window")
        print(f"   Dataset shape: {self.df.shape}")

        print(f"\nQuality Checks:")
        print(f"  Standard deviation > 3 (prevents flat predictions)")
        print(f"  Sliding window validation (no future leakage)")
        print(f"  Cross-validated feature selection")

        print(f"\nIndividual Model Performance:")
        print("-" * 80)

        for model_name in self.model_configs.keys():
            if model_name not in evaluation_results:
                continue

            results = evaluation_results[model_name]

            print(f"\n {model_name}:")
            print(f"    Train Metrics:")
            print(f"       MSE  = {results.get('avg_train_mse', 0):.3f}")
            print(f"       RMSE = {results.get('avg_train_rmse', 0):.3f}")
            print(f"       MAE  = {results.get('avg_train_mae', 0):.3f}")
            print(f"       MAPE = {results.get('avg_train_mape', 0):.2f}%")
            print(f"       R²   = {results.get('avg_train_r2', 0):.3f}")

            print(f"   Test Metrics:")
            print(f"       MSE  = {results.get('avg_test_mse', 0):.3f}")
            print(f"       RMSE = {results.get('avg_test_rmse', 0):.3f}")
            print(f"       MAE  = {results.get('avg_test_mae', 0):.3f}")
            print(f"       MAPE = {results.get('avg_test_mape', 0):.2f}%")
            print(f"       R²   = {results.get('avg_test_r2', 0):.3f}")
            print(f"       STD  = {results.get('std_pred', 0):.3f}")
            print(f"       Quality = {results.get('prediction_quality', 'N/A')}")

        if hasattr(self, 'ensemble_results'):
            print(f"\n🔗 Ensemble Model Performance:")
            print("-" * 80)

            for ensemble_name in self.ensemble_configs.keys():
                if ensemble_name not in evaluation_results:
                    continue

                results = evaluation_results[ensemble_name]

                print(f"\n {ensemble_name} (Components: {', '.join(self.ensemble_configs[ensemble_name])}):")
                print(f"   MSE={results.get('avg_test_mse', 0):.3f}, "
                      f"RMSE={results.get('avg_test_rmse', 0):.3f}, "
                      f"MAE={results.get('avg_test_mae', 0):.3f}, "
                      f"MAPE={results.get('avg_test_mape', 0):.2f}%, "
                      f"R²={results.get('avg_test_r2', 0):.3f}, "
                      f"STD={results.get('std_pred', 0):.3f}, "
                      f"Quality={results.get('prediction_quality', 'N/A')}")

        print(f"\nBest Performing Model:")
        print("-" * 80)

        best_r2 = -999
        best_model = None

        for model_name, model_results in evaluation_results.items():
            r2 = model_results.get('avg_test_r2', -999)
            if r2 > best_r2:
                best_r2 = r2
                best_model = model_name

        if best_model:
            model_type = " Ensemble" if best_model in self.ensemble_configs else " Individual"
            print(f"   {model_type} - {best_model} (R²={best_r2:.3f})")

        # Feature summary
        if self.selected_features:
            print(f"\n Selected Features ({len(self.selected_features)}):")
            print("-" * 80)
            print(f"   {', '.join(self.selected_features[:10])}{'...' if len(self.selected_features) > 10 else ''}")

        print("\n" + "="*80)
        print(f" SYSTEM VALIDATION COMPLETE - {self.horizon_hours}H HORIZON")
        print("="*80)

        return evaluation_results

    def train_baseline(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_squared_error

        print(f" Training baseline models for {self.horizon_hours}h horizon...")

        if not self.selected_features:
            print(" No features selected! Run feature selection first.")
            return {}

        available_features = [f for f in self.selected_features if f in self.df.columns]
        if not available_features:
            print(" No valid feature columns found")
            return {}

        print(f"   Using {len(available_features)} features")

        all_cols = available_features + [self.target_column]
        clean_data = self.df[all_cols].dropna()

        if len(clean_data) < 100:
            print(f" Insufficient clean data! Only {len(clean_data)} rows available")
            return {}

        print(f"   Clean data shape: {clean_data.shape}")

        split_idx = int(0.8 * len(clean_data))

        X_train = clean_data[available_features].iloc[:split_idx]
        X_test = clean_data[available_features].iloc[split_idx:]
        y_train = clean_data[self.target_column].iloc[:split_idx]
        y_test = clean_data[self.target_column].iloc[split_idx:]

        results = {}

        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)

            lr_pred = lr.predict(X_test)
            lr_r2 = r2_score(y_test, lr_pred)
            lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

            print(f"   Linear Regression: R²={lr_r2:.4f}, RMSE={lr_rmse:.3f}")
            results['linear_r2'] = lr_r2
            results['linear_rmse'] = lr_rmse
        except Exception as e:
            print(f"   Linear Regression failed: {e}")
            results['linear_r2'] = -999
            results['linear_rmse'] = 999

        try:
            rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            rf.fit(X_train, y_train)

            rf_pred = rf.predict(X_test)
            rf_r2 = r2_score(y_test, rf_pred)
            rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

            print(f"   Random Forest: R²={rf_r2:.4f}, RMSE={rf_rmse:.3f}")
            results['rf_r2'] = rf_r2
            results['rf_rmse'] = rf_rmse

            if hasattr(rf, 'feature_importances_'):
                feature_imp = list(zip(available_features, rf.feature_importances_))
                feature_imp.sort(key=lambda x: x[1], reverse=True)
                top_features = feature_imp[:3]
                print(f"   Top 3 important features: {[(f, round(imp, 3)) for f, imp in top_features]}")

        except Exception as e:
            print(f"   Random Forest failed: {e}")
            results['rf_r2'] = -999
            results['rf_rmse'] = 999

        naive_pred = np.full(len(y_test), y_train.mean())
        naive_r2 = r2_score(y_test, naive_pred)
        naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))

        print(f"   Naive (mean): R²={naive_r2:.4f}, RMSE={naive_rmse:.3f}")
        print(f"   Target stats: mean={y_test.mean():.2f}, std={y_test.std():.2f}")

        results.update({
            'naive_r2': naive_r2,
            'naive_rmse': naive_rmse,
            'target_std': y_test.std(),
            'target_mean': y_test.mean(),
            'test_size': len(y_test),
            'train_size': len(y_train)
        })

        print(f"   Baseline evaluation completed for {self.horizon_hours}h horizon")
        return results

    def run_complete_pipeline(self, max_features=15, train_baseline=True, save_models=True):
        
        print(f"\n Starting complete AQI forecasting pipeline for {self.horizon_hours}h horizon")
        print("="*80)
        
        pipeline_results = {
            'horizon_hours': self.horizon_hours,
            'target_column': self.target_column,
            'dataset_shape': self.df.shape,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            print("\nData Quality Validation")
            quality_issues = self.validate_data_quality()
            pipeline_results['data_quality_issues'] = quality_issues
            
            if quality_issues:
                print(" Data quality issues found - proceeding with caution")
            
            print("\n Feature Selection Pipeline")
            feature_selection_results = self.feature_selection_pipeline(max_features=max_features)
            pipeline_results['feature_selection'] = feature_selection_results
            
            if not self.selected_features:
                print("Feature selection failed - cannot proceed")
                return pipeline_results
            
            print("\n Data Leakage Detection")
            leakage_results = self.check_data_leakage()
            pipeline_results['leakage_check'] = leakage_results
            
            print("\nModel Initialization")
            self.initialize_models()
            pipeline_results['models_initialized'] = list(self.model_configs.keys())
            
            if train_baseline:
                print("\nBaseline Model Training")
                baseline_results = self.train_baseline()
                pipeline_results['baseline_results'] = baseline_results
            
            print("\n Sliding Window Cross-Validation")
            self.sliding_window_validation()
            
            print("\nFinal Model Training")
            self.train_final_models()

            print("\nPerformance Evaluation")
            evaluation_results = self.generate_report()
            pipeline_results['evaluation_results'] = evaluation_results
            
            if save_models:
                print("\n Saving Models")
                save_path = self.save_models()
                pipeline_results['models_saved_to'] = save_path
            
            pipeline_results['status'] = 'completed'
            pipeline_results['success'] = True
            
            print(f"\n Pipeline completed successfully for {self.horizon_hours}h horizon!")
            
        except Exception as e:
            print(f"\n Pipeline failed: {str(e)}")
            pipeline_results['status'] = 'failed'
            pipeline_results['success'] = False
            pipeline_results['error'] = str(e)
        
        return pipeline_results

    def predict_new_data(self, new_data):
        if not hasattr(self, 'deployment_models') or not self.deployment_models:
            print(" No trained models found. Please run the training pipeline first.")
            return None
        
        if not self.selected_features:
            print(" No selected features found. Please run feature selection first.")
            return None
        
        print(f" Making predictions for {self.horizon_hours}h horizon...")
        
        missing_features = [f for f in self.selected_features if f not in new_data.columns]
        if missing_features:
            print(f" Missing features in new data: {missing_features}")
            available_features = [f for f in self.selected_features if f in new_data.columns]
            if not available_features:
                print(" No required features found in new data")
                return None
            print(f"   Using {len(available_features)} available features")
            feature_subset = available_features
        else:
            feature_subset = self.selected_features
        
        X_new = new_data[feature_subset].copy()
        
        X_new = X_new.fillna(method='ffill').fillna(method='bfill')
        for col in X_new.columns:
            if X_new[col].isna().any():
                X_new[col].fillna(X_new[col].median(), inplace=True)
        
        predictions = {}
        
        # Individual model predictions
        for model_name, model in self.deployment_models.items():
            if model is None:
                continue
                
            try:
                pred = model.predict(X_new)
                predictions[model_name] = pred
                print(f"    {model_name}: {len(pred)} predictions")
                
            except Exception as e:
                print(f"   Error with {model_name}: {str(e)}")
                continue
        
        # Ensemble predictions
        if hasattr(self, 'ensemble_configs'):
            for ensemble_name, component_models in self.ensemble_configs.items():
                available_preds = []
                for model_name in component_models:
                    if model_name in predictions:
                        available_preds.append(predictions[model_name])
                
                if len(available_preds) >= 2:  
                    ensemble_pred = np.mean(available_preds, axis=0)
                    predictions[ensemble_name] = ensemble_pred
                    print(f"   {ensemble_name}: {len(ensemble_pred)} predictions")
                else:
                    print(f"   {ensemble_name}: Insufficient models ({len(available_preds)}/3)")
        
        return predictions

    def get_feature_importance(self, top_k=10):
        if not hasattr(self, 'deployment_models') or not self.deployment_models:
            print(" No trained models found.")
            return {}
        
        print(f" Extracting feature importance for {self.horizon_hours}h horizon...")
        
        importance_results = {}
        
        for model_name, model in self.deployment_models.items():
            if model is None:
                continue
            
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance = list(zip(self.selected_features, importances))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    importance_results[model_name] = feature_importance[:top_k]
                    
                elif hasattr(model, 'get_feature_importance'):
                    importances = model.get_feature_importance()
                    feature_importance = list(zip(self.selected_features, importances))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    importance_results[model_name] = feature_importance[:top_k]
                    
                elif model_name == 'XGBoost' and hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance = list(zip(self.selected_features, importances))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    importance_results[model_name] = feature_importance[:top_k]
                    
                elif model_name == 'LightGBM':
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    elif hasattr(model, 'feature_importance'):
                        importances = model.feature_importance()
                    else:
                        continue
                    
                    feature_importance = list(zip(self.selected_features, importances))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    importance_results[model_name] = feature_importance[:top_k]
                
                print(f"    {model_name}: Top feature - {importance_results[model_name][0][0]}")
                
            except Exception as e:
                print(f"    Error extracting importance from {model_name}: {str(e)}")
                continue
        
        return importance_results

    def create_prediction_summary(self, predictions):
        if not predictions:
            print(" No predictions provided")
            return pd.DataFrame()
        
        print(" Creating prediction summary...")
        
        summary_data = []
        
        for model_name, pred_values in predictions.items():
            if len(pred_values) == 0:
                continue
                
            summary_data.append({
                'model': model_name,
                'mean_prediction': np.mean(pred_values),
                'median_prediction': np.median(pred_values),
                'std_prediction': np.std(pred_values),
                'min_prediction': np.min(pred_values),
                'max_prediction': np.max(pred_values),
                'q25_prediction': np.percentile(pred_values, 25),
                'q75_prediction': np.percentile(pred_values, 75),
                'prediction_count': len(pred_values),
                'model_type': 'ensemble' if model_name in getattr(self, 'ensemble_configs', {}) else 'individual'
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        if not summary_df.empty:
            summary_df = summary_df.sort_values('mean_prediction', ascending=False)
            print(f"    Created summary for {len(summary_df)} models")
        
        return summary_df

    def export_predictions(self, predictions, output_path=None, include_metadata=True):
        if not predictions:
            print(" No predictions to export")
            return None
        
        pred_df = pd.DataFrame(predictions)
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"aqi_predictions_{self.horizon_hours}h_{timestamp}.csv"
        
        try:
            pred_df.to_csv(output_path, index=False)
            print(f" Predictions exported to: {output_path}")
            
            if include_metadata:
                metadata_path = output_path.replace('.csv', '_metadata.json')
                
                metadata = {
                    'horizon_hours': self.horizon_hours,
                    'target_column': self.target_column,
                    'selected_features': self.selected_features,
                    'model_names': list(predictions.keys()),
                    'prediction_count': len(pred_df),
                    'export_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'prediction_summary': {
                        'models_used': len(predictions),
                        'ensemble_models': len([m for m in predictions.keys() if m in getattr(self, 'ensemble_configs', {})]),
                        'individual_models': len([m for m in predictions.keys() if m not in getattr(self, 'ensemble_configs', {})])
                    }
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f" Metadata exported to: {metadata_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Error exporting predictions: {str(e)}")
            return None
