import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from feature_generators.base_features import BaseFeatures

class BehavioralFeatures(BaseFeatures):
    """Extracts behavioral features from user event data."""

    def __init__(self, events_df: pd.DataFrame, outcomes_df: pd.DataFrame, window_days: int = 14):
        super().__init__(outcomes_df)
        self.events_df = events_df.copy()
        self.window_days = window_days
        
        # Use base class preprocessing
        self._events_merged = self._preprocess_timeseries(
            self.events_df, 
            timestamp_col='timestamp', 
            window_days=self.window_days
        )
        # Add event date for daily aggregations
        self._events_merged['event_date'] = self._events_merged['timestamp'].dt.date
        # Storage for fitted bigram objects (for inference)
        self._bigram_fitted_objects = None

    def add_event_count_feature(self):
        """Adds count of events in the window period."""
        counts = (
            self._events_merged
            .groupby(['user_id', 'application_at'])
            .size()
            .reset_index(name='num_events_14d')
        )
        self._merge_feature(counts, 'num_events_14d', fillna_value=0)
        return self.outcomes_df

    def add_phone_instance_count_feature(self):
        """Adds count of unique phone instances per user (uses all pre-app events)."""
        # Use all pre-application events (not just window)
        all_events = self._preprocess_timeseries(
            self.events_df,
            timestamp_col='timestamp',
            window_days=None  # All pre-app events
        )
        
        ph_inst = (
            all_events
            .groupby(['user_id', 'application_at'])['phone_instance_id']
            .nunique()
            .reset_index(name='count_ph_inst')
        )
        self._merge_feature(ph_inst, 'count_ph_inst', fillna_value=0)
        return self.outcomes_df

    def add_session_count_feature(self, include_rank: bool = False):
        """Adds count of unique sessions in the window period and optional rank."""
        sessions = (
            self._events_merged
            .groupby(['user_id', 'application_at'])['session_id']
            .nunique()
            .reset_index(name='num_sessions_14d')
        )
        self._merge_feature(sessions, 'num_sessions_14d', fillna_value=0)
        
        if include_rank:
            self.outcomes_df['num_sessions_rank'] = self.outcomes_df['num_sessions_14d'].rank(pct=True)
        
        return self.outcomes_df

    def add_active_days_feature(self, include_rank: bool = False):
        """Adds count of unique active days in the window period and optional rank."""
        days = (
            self._events_merged
            .groupby(['user_id', 'application_at'])['event_date']
            .nunique()
            .reset_index(name='unique_days_active_14d')
        )
        self._merge_feature(days, 'unique_days_active_14d', fillna_value=0)
        
        if include_rank:
            self.outcomes_df['active_days_rank'] = self.outcomes_df['unique_days_active_14d'].rank(pct=True)
        
        return self.outcomes_df

    def add_avg_session_duration_feature(self):
        """Adds average session duration in seconds."""
        session_durations = (
            self._events_merged
            .groupby(['user_id', 'application_at', 'session_id'])['timestamp']
            .agg(lambda x: (x.max() - x.min()).total_seconds())
            .reset_index(name='session_duration')
        )
        avg_duration = (
            session_durations
            .groupby(['user_id', 'application_at'])['session_duration']
            .mean()
            .reset_index(name='avg_session_duration')
        )
        self._merge_feature(avg_duration, 'avg_session_duration', fillna_value=0)
        return self.outcomes_df

    def add_single_event_session_ratio_feature(self):
        """Adds percentage of sessions with only one event."""
        session_counts = (
            self._events_merged
            .groupby(['user_id', 'application_at', 'session_id'])
            .size()
            .reset_index(name='event_count')
        )
        ratio = (
            session_counts
            .groupby(['user_id', 'application_at'])
            .apply(lambda x: (x['event_count'] == 1).mean())
            .reset_index(name='pct_single_event_sessions')
        )
        self._merge_feature(ratio, 'pct_single_event_sessions', fillna_value=0)
        return self.outcomes_df

    def add_events_last_24h_feature(self):
        """Adds count of events in the last 24 hours before application."""
        last_24h = self._events_merged[
            self._events_merged['timestamp'] >= self._events_merged['application_at'] - pd.Timedelta(hours=24)
        ]
        counts = (
            last_24h
            .groupby(['user_id', 'application_at'])
            .size()
            .reset_index(name='events_last_24h')
        )
        self._merge_feature(counts, 'events_last_24h', fillna_value=0)
        return self.outcomes_df

    def add_time_gap_feature(self):
        """Adds time gap (seconds) from last event to application."""
        last_event = (
            self._events_merged
            .groupby(['user_id', 'application_at'])['timestamp']
            .max()
            .reset_index(name='last_event_time')
        )
        last_event['time_from_last_event_to_application'] = (
            (last_event['application_at'] - last_event['last_event_time']).dt.total_seconds()
        )
        self._merge_feature(
            last_event[['user_id', 'application_at', 'time_from_last_event_to_application']],
            'time_from_last_event_to_application',
            fillna_value=0
        )
        return self.outcomes_df

    def add_unique_screens_feature(self):
        """Adds count of unique screens visited."""
        screens = (
            self._events_merged
            .groupby(['user_id', 'application_at'])['screen_name']
            .nunique()
            .reset_index(name='num_unique_screens')
        )
        self._merge_feature(screens, 'num_unique_screens', fillna_value=0)
        return self.outcomes_df

    def add_bigram_risk_features(
        self,
        top_k_screens: int = 10,
        n_splits: int = 5,
        smoothing: int = 50,
        random_state: int = 42,
        mode: str = "train",
        fitted_objects: dict = None
    ):
        """biogram risk from eda notebook changed for inference"""

        if mode not in {"train", "test"}:
            raise ValueError("mode must be 'train' or 'test'")

        events = self.events_df.copy()
        events['timestamp'] = pd.to_datetime(events['timestamp'])

        df = events.merge(
            self.outcomes_df[['user_id', 'application_at']],
            on='user_id',
            how='inner'
        )

        df = (
            df[df['timestamp'] < df['application_at']]
            .sort_values(['user_id', 'timestamp'])
            .copy()
        )

        if df.empty:
            # Nothing to compute; fill defaults
            default_rate = 0.0
            self.outcomes_df['mean_bigram_default_rate'] = default_rate
            self.outcomes_df['max_bigram_default_rate'] = default_rate
            return self.outcomes_df

        if mode == 'train':
            if 'is_repaid' not in self.outcomes_df.columns:
                raise ValueError("outcomes_df must contain 'is_repaid' for training bigram features")
            df = df.merge(
                self.outcomes_df[['user_id', 'application_at', 'is_repaid']].drop_duplicates(),
                on=['user_id', 'application_at'],
                how='left'
            )
            top_screens = df['screen_name'].value_counts().head(top_k_screens).index
        else:
            if fitted_objects is None and self._bigram_fitted_objects is None:
                raise ValueError("fitted_objects must be provided for test mode")
            fitted_objects = fitted_objects or self._bigram_fitted_objects
            top_screens = fitted_objects['top_screens']

        df['screen_r'] = df['screen_name'].where(df['screen_name'].isin(top_screens), 'OTHER')
        df['next_screen'] = df.groupby('user_id')['screen_r'].shift(-1)
        df = df.dropna(subset=['next_screen'])
        df['bigram'] = df['screen_r'] + '→' + df['next_screen']

        if mode == 'train':
            df['default'] = 1 - df['is_repaid'].astype(int)
            global_default_rate = df['default'].mean()

            users = self.outcomes_df[['user_id']].drop_duplicates().copy()
            users['mean_bigram_default_rate'] = global_default_rate
            users['max_bigram_default_rate'] = global_default_rate

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

            for tr_idx, va_idx in kf.split(users):
                train_users = users.iloc[tr_idx]['user_id']
                val_users = users.iloc[va_idx]['user_id']

                train_df = df[df['user_id'].isin(train_users)]
                val_df = df[df['user_id'].isin(val_users)]

                stats = (
                    train_df.groupby('bigram')['default']
                    .agg(['mean', 'count'])
                    .reset_index()
                )

                stats['risk'] = (
                    (stats['mean'] * stats['count'] + global_default_rate * smoothing) /
                    (stats['count'] + smoothing)
                )

                risk_map = stats.set_index('bigram')['risk']

                val_df = val_df.copy()
                val_df['bigram_risk'] = val_df['bigram'].map(risk_map).fillna(global_default_rate)

                agg = (
                    val_df.groupby('user_id')['bigram_risk']
                    .agg(mean_bigram_default_rate='mean', max_bigram_default_rate='max')
                    .reset_index()
                )

                users.loc[
                    users['user_id'].isin(agg['user_id']),
                    ['mean_bigram_default_rate', 'max_bigram_default_rate']
                ] = agg[['mean_bigram_default_rate', 'max_bigram_default_rate']].values

            final_stats = (
                df.groupby('bigram')['default']
                .agg(['mean', 'count'])
                .reset_index()
            )

            final_stats['risk'] = (
                (final_stats['mean'] * final_stats['count'] + global_default_rate * smoothing) /
                (final_stats['count'] + smoothing)
            )

            self._bigram_fitted_objects = {
                'bigram_risk_map': final_stats.set_index('bigram')['risk'],
                'global_default_rate': global_default_rate,
                'top_screens': set(top_screens)
            }

            bigram_features = users
            default_rate = global_default_rate

        else:
            risk_map = fitted_objects['bigram_risk_map']
            global_default_rate = fitted_objects['global_default_rate']

            df['bigram_risk'] = df['bigram'].map(risk_map).fillna(global_default_rate)

            bigram_features = (
                df.groupby('user_id')['bigram_risk']
                .agg(mean_bigram_default_rate='mean', max_bigram_default_rate='max')
                .reset_index()
            )

            default_rate = global_default_rate

        self.outcomes_df = self.outcomes_df.merge(
            bigram_features[['user_id', 'mean_bigram_default_rate', 'max_bigram_default_rate']],
            on='user_id',
            how='left'
        )

        self.outcomes_df['mean_bigram_default_rate'] = self.outcomes_df['mean_bigram_default_rate'].fillna(default_rate)
        self.outcomes_df['max_bigram_default_rate'] = self.outcomes_df['max_bigram_default_rate'].fillna(default_rate)

        return self.outcomes_df


    def add_derived_features(self, include_ranks: bool = False):
        """Adds log transformations, derived features, and optional rank features."""
        self.outcomes_df['avg_session_duration_log'] = np.log1p(self.outcomes_df['avg_session_duration'])
        self.outcomes_df['time_from_last_event_to_application_log'] = np.log1p(
            self.outcomes_df['time_from_last_event_to_application']
        )
        self.outcomes_df['events_last_24h_ratio'] = (
            self.outcomes_df['events_last_24h'] / (self.outcomes_df['num_events_14d'] + 1)
        )
        self.outcomes_df['has_events_14d'] = (self.outcomes_df['num_events_14d'] > 0).astype(int)
        
        if include_ranks:
            self.outcomes_df['session_duration_rank'] = self.outcomes_df['avg_session_duration_log'].rank(pct=True)
            self.outcomes_df['recency_rank'] = self.outcomes_df['time_from_last_event_to_application_log'].rank(pct=True)
        
        return self.outcomes_df

    def add_diligence_stability_feature(self):
        """
        Adds diligence_stability interaction feature.
        Requires 'session_duration_rank' and 'location_entropy' to exist.
        """
        if 'session_duration_rank' not in self.outcomes_df.columns:
            raise ValueError("session_duration_rank must exist. Call add_derived_features(include_ranks=True) first.")
        if 'location_entropy' not in self.outcomes_df.columns:
            raise ValueError("location_entropy must exist. Run GPSFeatures first.")
        
        self.outcomes_df['diligence_stability'] = (
            self.outcomes_df['session_duration_rank'] *
            (1 - self.outcomes_df['location_entropy'])
        )
        return self.outcomes_df

    def add_behavioral_features(self, include_bigrams: bool = True, include_ranks: bool = False, 
                                 include_diligence_stability: bool = False, drop_raw: bool = True):
        """
        Build all behavioral features at once.
        
        Args:
            include_bigrams: Whether to include bigram risk features (requires 'is_repaid' column)
            include_ranks: Whether to include rank-normalized features
            include_diligence_stability: Whether to add diligence_stability (requires ranks + GPS entropy)
            drop_raw: If include_ranks=True, whether to drop raw features after ranking
        """
        self.add_event_count_feature()
        self.add_phone_instance_count_feature()
        self.add_session_count_feature(include_rank=include_ranks)
        self.add_active_days_feature(include_rank=include_ranks)
        self.add_avg_session_duration_feature()
        self.add_single_event_session_ratio_feature()
        self.add_events_last_24h_feature()
        self.add_time_gap_feature()
        self.add_unique_screens_feature()
        self.add_derived_features(include_ranks=include_ranks)
        
        if include_diligence_stability and include_ranks and 'location_entropy' in self.outcomes_df.columns:
            self.add_diligence_stability_feature()
        
        if include_ranks and drop_raw:
            drop_cols = [
                'num_sessions_14d',
                'unique_days_active_14d',
                'avg_session_duration_log',
                'time_from_last_event_to_application_log'
            ]
            self.outcomes_df.drop(columns=drop_cols, inplace=True, errors='ignore')
        
        if include_bigrams and 'is_repaid' in self.outcomes_df.columns:
            self.add_bigram_risk_features()
        
        return self.outcomes_df
