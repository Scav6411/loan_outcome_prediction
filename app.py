from fastapi import FastAPI, APIRouter, HTTPException
from schema import PredictRequest, PredictResponse, DatabaseError, DataNotFoundError, FeaturePipelineError, ModelPredictionError
from inference import make_prediction

from config import settings

# Create FastAPI app
app = FastAPI(
    title="Loan Outcome Prediction API",
    description="API for predicting loan repayment probability",
    version="1.0.0"
)

router = APIRouter()

@router.get("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = make_prediction(
            user_id=request.user_id,
            application_at=request.application_at,
        )

        return PredictResponse(
            user_id=result['user_id'],
            application_at=result['application_at'],
            prediction=result['prediction'],
        )

    except DataNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except DatabaseError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except FeaturePipelineError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except ModelPredictionError as e:
        raise HTTPException(status_code=500, detail=str(e))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unexpected internal server error",
        )


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Include the router
app.include_router(router, prefix="/api/v1", tags=["predictions"])