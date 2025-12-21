import pandas as pd
from feature_generators.base_features import BaseFeatures

class TemporalFeatures(BaseFeatures):
    """Extracts time-based features from application timestamps."""

    def __init__(self, outcomes_df: pd.DataFrame):
        super().__init__(outcomes_df)

    def add_app_hour_feature(self):
        """Adds hour of application (0-23)."""
        self.outcomes_df['app_hour'] = self.outcomes_df['application_at'].dt.hour
        return self.outcomes_df

    def add_app_day_of_week_feature(self):
        """Adds day of week (0=Monday, 6=Sunday)."""
        self.outcomes_df['app_day_of_week'] = self.outcomes_df['application_at'].dt.dayofweek
        return self.outcomes_df

    def add_app_day_of_month_feature(self):
        """Adds day of month (1-31)."""
        self.outcomes_df['app_day_of_month'] = self.outcomes_df['application_at'].dt.day
        return self.outcomes_df

    def add_app_year_feature(self):
        """Adds year of application."""
        self.outcomes_df['app_year'] = self.outcomes_df['application_at'].dt.year
        return self.outcomes_df

    def add_is_weekend_feature(self):
        """Adds weekend flag (1 if Saturday/Sunday, 0 otherwise)."""
        if 'app_day_of_week' not in self.outcomes_df.columns:
            self.add_app_day_of_week_feature()
        self.outcomes_df['is_weekend'] = (self.outcomes_df['app_day_of_week'] >= 5).astype(int)
        return self.outcomes_df

    def add_is_late_night_feature(self):
        """
        Adds 'Financial Urgency' flag for late night applications.
        Logic: Applications between 11 PM (23:00) and 5 AM are flagged.
        """
        if 'app_hour' not in self.outcomes_df.columns:
            self.add_app_hour_feature()
        self.outcomes_df['is_late_night'] = (
            (self.outcomes_df['app_hour'] >= 23) | (self.outcomes_df['app_hour'] <= 5)
        ).astype(int)
        return self.outcomes_df

    def add_temporal_features(self):
        """Build all temporal features at once."""
        self.add_app_hour_feature()
        self.add_app_day_of_week_feature()
        self.add_app_day_of_month_feature()
        self.add_app_year_feature()
        self.add_is_weekend_feature()
        self.add_is_late_night_feature()
        return self.outcomes_df
