import numpy as np
import pandas as pd
from feature_generators.base_features import BaseFeatures

class GPSFeatures(BaseFeatures):
    """Extrcts GPS-based features from the gps and loan_outcomes_train data"""

    def __init__(self, gps_df: pd.DataFrame, outcomes_df: pd.DataFrame):
        super().__init__(outcomes_df)
        self.gps_df = gps_df.copy()
        
        self._gps_pre_app = self._preprocess_timeseries(
            self.gps_df, 
            timestamp_col='time_of_fix', 
            window_days=None
        )

    def _get_rounded_locations(self, precision=4):
        """Get pre-application GPS data with rounded lat/lon for location-based features."""
        df = self._gps_pre_app.copy()
        df['lat_round'] = df['latitude'].round(precision)
        df['lon_round'] = df['longitude'].round(precision)
        return df

    def add_max_speed_feature(self):
        """Adds 'max_land_speed'. -1 indicates no GPS data (distinct from 0 = stationary)."""
        speed_stats = (
            self._gps_pre_app
            .groupby(['user_id', 'application_at'])['land_speed']
            .max()
            .reset_index()
            .rename(columns={'land_speed': 'max_land_speed'})
        )
        self._merge_feature(speed_stats, 'max_land_speed', fillna_value=-1)
        return self.outcomes_df

    def add_gps_count_feature(self):
        """Adds count of GPS points before application."""
        counts = (
            self._gps_pre_app
            .groupby(['user_id', 'application_at'])
            .size()
            .reset_index(name='gps_points_pre_app')
        )
        self._merge_feature(counts, 'gps_points_pre_app', fillna_value=0)
        return self.outcomes_df

    def add_unique_locations_feature(self, precision=4):
        """Adds count of unique locations (rounded lat/lon)."""
        df = self._get_rounded_locations(precision)
        uniq = (
            df.groupby(['user_id', 'application_at'])
            .apply(lambda x: x[['lat_round', 'lon_round']].drop_duplicates().shape[0])
            .reset_index(name='unique_locations')
        )
        self._merge_feature(uniq, 'unique_locations', fillna_value=0)
        return self.outcomes_df

    def add_dominant_location_ratio(self, precision=4):
        """Adds ratio of most frequent location to total locations."""
        df = self._get_rounded_locations(precision)
        loc_counts = (
            df.groupby(['user_id', 'application_at', 'lat_round', 'lon_round'])
            .size()
            .reset_index(name='cnt')
        )
        ratio = (
            loc_counts
            .groupby(['user_id', 'application_at'])
            .agg(dominant_location_ratio=('cnt', lambda x: x.max() / x.sum()))
            .reset_index()
        )
        self._merge_feature(ratio, 'dominant_location_ratio', fillna_value=0)
        return self.outcomes_df

    def add_location_entropy(self, precision=4):
        """Adds entropy of location distribution."""
        def entropy(counts):
            p = counts / counts.sum()
            return -(p * np.log(p + 1e-9)).sum()

        df = self._get_rounded_locations(precision)
        ent = (
            df.groupby(['user_id', 'application_at', 'lat_round', 'lon_round'])
            .size()
            .reset_index(name='cnt')
            .groupby(['user_id', 'application_at'])['cnt']
            .apply(entropy)
            .reset_index(name='location_entropy')
        )
        self._merge_feature(ent, 'location_entropy', fillna_value=0)
        return self.outcomes_df

    def add_radius_of_gyration(self):
        """Adds radius of gyration - measure of spatial spread."""
        def rog(x):
            lat_c = x['latitude'].mean()
            lon_c = x['longitude'].mean()
            return np.sqrt(((x['latitude'] - lat_c)**2 + (x['longitude'] - lon_c)**2).mean())

        rog_df = (
            self._gps_pre_app
            .groupby(['user_id', 'application_at'])
            .apply(rog)
            .reset_index(name='radius_of_gyration')
        )
        self._merge_feature(rog_df, 'radius_of_gyration', fillna_value=0)
        return self.outcomes_df

    def add_gps_features(self):
        
        self.add_max_speed_feature()
        self.add_gps_count_feature()
        self.add_unique_locations_feature()
        self.add_dominant_location_ratio()
        self.add_location_entropy()
        self.add_radius_of_gyration()

        # log transformations
        self.outcomes_df['gps_points_pre_app_log'] = np.log1p(self.outcomes_df['gps_points_pre_app'])
        self.outcomes_df['radius_of_gyration_log'] = np.log1p(self.outcomes_df['radius_of_gyration'])

        return self.outcomes_df
