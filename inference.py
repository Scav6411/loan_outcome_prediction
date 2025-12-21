import xgboost as xgb
from config import settings
from schema import ModelPredictionError, DatabaseError, DataNotFoundError
import pandas as pd
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from feature_generators.temporal_features import TemporalFeatures
from feature_generators.gps_features import GPSFeatures
from feature_generators.behavioral_features import BehavioralFeatures
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "weights" / "best_weights.json"
X_COLS = [
    'feature_1', 'feature_2', 'feature_4', 'feature_8', 'feature_9', 'feature_10',
    'feature_3_is_missing',
    'count_ph_inst', 'gps_points_pre_app', 'unique_locations', 'dominant_location_ratio',
    'location_entropy', 'max_land_speed', 'radius_of_gyration_log', 'gps_points_pre_app_log',
    'app_hour', 'app_day_of_week', 'app_day_of_month', 'app_year', 'is_weekend', 'is_late_night',
    'num_sessions_rank', 'active_days_rank', 'session_duration_rank', 'recency_rank',
    'diligence_stability', 'mean_bigram_default_rate', 'max_bigram_default_rate'
]

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            dbname=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            connect_timeout=5,
            cursor_factory=RealDictCursor,
        )
        return conn

    except Exception as e:
        raise DatabaseError(
            f"Database connection failed: {str(e)}"
        )


class ModelService:
    _model = None

    @classmethod
    def load_model(cls) -> xgb.XGBClassifier:
        if cls._model is None:
            try:
                model = xgb.XGBClassifier()
                model.load_model(settings.MODEL_PATH)
                cls._model = model

            except Exception as e:
                raise ModelPredictionError(
                    f"Failed to load XGBoost model: {str(e)}"
                )

        return cls._model

def predict_probability(features: pd.DataFrame) -> float:
    try:
        model = ModelService.load_model()
        prob = model.predict_proba(features)[:, 1]
        return float(prob[0]) if len(prob) == 1 else prob.tolist()

    except Exception as e:
        raise ModelPredictionError(
            f"Inference failed: {str(e)}"
        )
    

def fetch_user_data(user_id: int, application_at: datetime) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features_df = fetch_features_table(user_id)
    events_df = fetch_events_table(user_id, application_at)
    gps_df = fetch_gps_table(user_id, application_at)
    
    return features_df, events_df, gps_df

def fetch_features_table(user_id):
    query = """
        SELECT *
        FROM features
        WHERE user_id = %s
    """
    return _fetch_df(query, (user_id,), "features")


def fetch_gps_table(user_id, application_at):
    query = """
        SELECT *
        FROM gps
        WHERE user_id = %s
          AND time_of_fix <= %s
    """
    return _fetch_df(query, (user_id, application_at), "gps")


def fetch_events_table(user_id, application_at):
    query = """
        SELECT *
        FROM events
        WHERE user_id = %s
          AND timestamp <= %s
    """
    return _fetch_df(query, (user_id, application_at), "events")


def _fetch_df(query, params, table_name):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()

        if not rows:
            raise DataNotFoundError(
                f"No data found in {table_name} table"
            )

        return pd.DataFrame(rows)

    except DataNotFoundError:
        raise
    except Exception as e:
        raise DatabaseError(
            f"Failed to fetch {table_name} data: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


def prepare_features_for_prediction(
    user_id: int,
    application_at: datetime,
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    gps_df: pd.DataFrame,
    bigram_fitted_objects: dict = None
) -> pd.DataFrame:
    try:
        outcomes_df = pd.DataFrame({
            'user_id': [user_id],
            'application_at': [pd.to_datetime(application_at)]
        })
        
        temporal = TemporalFeatures(outcomes_df=outcomes_df)
        result_df = temporal.add_temporal_features()
        
        gps = GPSFeatures(gps_df=gps_df, outcomes_df=result_df)
        result_df = gps.add_gps_features()
        
        behavior = BehavioralFeatures(
            events_df=events_df, 
            outcomes_df=result_df, 
            window_days=14
        )
        result_df = behavior.add_behavioral_features(
            include_bigrams=False,
            include_ranks=True,
            include_diligence_stability=True,
            drop_raw=True
        )
        
        if bigram_fitted_objects is not None:
            result_df = behavior.add_bigram_risk_features(
                mode='test',
                fitted_objects=bigram_fitted_objects
            )
        else:
            # Default bigram features to 0 if no fitted objects
            result_df['mean_bigram_default_rate'] = 0.0
            result_df['max_bigram_default_rate'] = 0.0
        
        result_df = result_df.merge(features_df, on='user_id', how='left')
        
        result_df['feature_3_is_missing'] = result_df['feature_3'].isna().astype(int)
        
        for col in X_COLS:
            if col not in result_df.columns:
                result_df[col] = 0
        
        # Extract final features and convert to numeric types for XGBoost
        final_features = result_df[X_COLS].copy()
        for col in final_features.columns:
            final_features[col] = pd.to_numeric(final_features[col], errors='coerce')
        
        return final_features
    
    except Exception as e:
        raise ModelPredictionError(
            f"Feature preparation failed for user {user_id}: {str(e)}"
        )


def make_prediction(
    user_id: int,
    application_at: datetime,
    bigram_fitted_objects: dict = None
) -> dict:
    """
    End-to-end prediction pipeline for loan repayment probability.
    
    This is the main entry point for the API. It orchestrates:
    1. Fetching user data from the database
    2. Preparing features using the feature engineering pipeline
    3. Running model inference
    
    Args:
        user_id: The user's ID
        application_at: The application timestamp
        bigram_fitted_objects: Optional pre-fitted bigram risk objects from training.
                               If None, bigram features default to 0.
    
    Returns:
        dict: {
            'user_id': int,
            'application_at': str (ISO format),
            'repayment_probability': float (0.0 to 1.0),
            'prediction_timestamp': str (ISO format)
        }
    
    Raises:
        DatabaseError: If database connection or query fails
        DataNotFoundError: If no data found for the user
        ModelPredictionError: If feature preparation or model inference fails
    """
    # fetch all required data from database
    features_df, events_df, gps_df = fetch_user_data(user_id, application_at)
    
    # prepare features for prediction
    features = prepare_features_for_prediction(
        user_id=user_id,
        application_at=application_at,
        features_df=features_df,
        events_df=events_df,
        gps_df=gps_df,
        bigram_fitted_objects=bigram_fitted_objects
    )
    
    # run model inference
    probability = predict_probability(features)
    
    # return structured response matching PredictResponse schema
    return {
        'user_id': user_id,
        'application_at': application_at,
        'prediction': probability,
    }

# needed patch for pandas 2.x compatibility with name parameter in reset_index, AI gen
_original_series_reset_index = pd.Series.reset_index
_original_dataframe_reset_index = pd.DataFrame.reset_index

def _patched_series_reset_index(self, level=None, *, drop=False, name=None, inplace=False, allow_duplicates=False):
    """Patched reset_index that supports 'name' parameter for pandas 2.x compatibility."""
    if name is not None:
        # Convert Series to DataFrame with the given name, then reset index
        # Use the original DataFrame reset_index to avoid recursion issues
        df = self.to_frame(name=name)
        result = _original_dataframe_reset_index(df, level=level, drop=drop, allow_duplicates=allow_duplicates)
        if inplace:
            raise ValueError("inplace=True is not supported with name parameter in this patch")
        return result
    return _original_series_reset_index(self, level=level, drop=drop, inplace=inplace, allow_duplicates=allow_duplicates)

def _patched_dataframe_reset_index(self, level=None, *, drop=False, inplace=False, col_level=0, col_fill='', allow_duplicates=False, names=None, name=None):
    """Patched DataFrame.reset_index that handles 'name' parameter for pandas 2.x compatibility."""
    if name is not None and names is None:
        # In pandas 2.x, groupby().apply() may already have grouping columns as regular columns
        # Check which index levels would conflict with existing columns
        index_names = [n for n in self.index.names if n is not None]
        conflicting = [n for n in index_names if n in self.columns]
        
        if conflicting:
            # Drop conflicting index levels (they're already columns)
            result = _original_dataframe_reset_index(self, level=level, drop=True, inplace=False, col_level=col_level, col_fill=col_fill, allow_duplicates=allow_duplicates, names=names)
        else:
            result = _original_dataframe_reset_index(self, level=level, drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill, allow_duplicates=allow_duplicates, names=names)
        
        # Find and rename the value column to the specified name
        # It could be 0, or the last column if it's unnamed
        if not inplace:
            if 0 in result.columns:
                result = result.rename(columns={0: name})
            else:
                # Find columns that aren't index names - the value column
                non_index_cols = [c for c in result.columns if c not in index_names]
                if len(non_index_cols) == 1 and non_index_cols[0] != name:
                    result = result.rename(columns={non_index_cols[0]: name})
                elif name not in result.columns:
                    # Last resort: rename the last column
                    last_col = result.columns[-1]
                    if last_col not in index_names:
                        result = result.rename(columns={last_col: name})
        return result
    return _original_dataframe_reset_index(self, level=level, drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill, allow_duplicates=allow_duplicates, names=names)

pd.Series.reset_index = _patched_series_reset_index
pd.DataFrame.reset_index = _patched_dataframe_reset_index
