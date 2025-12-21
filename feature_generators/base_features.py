import pandas as pd

class BaseFeatures:
    """Base class with common functionality for all feature builders. Not an apt name tho, but time constrained"""

    def __init__(self, outcomes_df: pd.DataFrame):
        self.outcomes_df = outcomes_df.copy()
        self._ensure_datetime()

    def _ensure_datetime(self):
        """Ensure application_at is datetime format."""
        self.outcomes_df['application_at'] = pd.to_datetime(self.outcomes_df['application_at'])

    def _preprocess_timeseries(self, df: pd.DataFrame, timestamp_col: str, 
                                window_days: int = None) -> pd.DataFrame:
        """
        Generic preprocessing for any timeseries data (events, GPS, etc.).
        
        Args:
            df: DataFrame with timeseries data (must have 'user_id' and timestamp column)
            timestamp_col: Name of the timestamp column in df
            window_days: Optional window in days before application. If None, uses all pre-app data.
        
        Returns:
            Filtered DataFrame with only pre-application data, merged with outcomes
        """
        # Ensure timestamp is datetime
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Merge with outcomes
        merged = df.merge(
            self.outcomes_df[['user_id', 'application_at']],
            on='user_id',
            how='inner'
        )
        
        # Filter for pre-application data (leakage prevention)
        mask = merged[timestamp_col] < merged['application_at']
        
        # Apply window if specified
        if window_days is not None:
            mask &= merged[timestamp_col] >= (merged['application_at'] - pd.Timedelta(days=window_days))
        
        return merged[mask].copy()

    def _merge_feature(self, feature_df: pd.DataFrame, feature_col: str, fillna_value=0):
        """
        Helper to merge a computed feature back to outcomes_df and fill missing values.
        
        Args:
            feature_df: DataFrame with user_id, application_at, and the new feature
            feature_col: Name of the feature column to fill NaN values
            fillna_value: Value to use for missing entries (default 0)
        """
        self.outcomes_df = self.outcomes_df.merge(
            feature_df,
            on=['user_id', 'application_at'],
            how='left'
        )
        self.outcomes_df[feature_col] = self.outcomes_df[feature_col].fillna(fillna_value)

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the current state of outcomes_df."""
        return self.outcomes_df
