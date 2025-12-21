from pydantic import BaseModel
from datetime import datetime

class PredictRequest(BaseModel):
    user_id: int
    application_at: datetime

class PredictResponse(BaseModel):
    user_id: int
    application_at: datetime
    prediction: float

# error handling sorted by different failure points
class DatabaseError(Exception):
    pass

class DataNotFoundError(Exception):
    pass

class FeaturePipelineError(Exception):
    pass

class ModelPredictionError(Exception):
    pass
