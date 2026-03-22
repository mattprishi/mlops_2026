import logging
from typing import Any
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException

from ml_service import config
from ml_service.features import to_dataframe
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)


logger = logging.getLogger(__name__)

MODEL = Model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Loads the initial model from MLflow on startup.
    """
    configure_mlflow()
    run_id = config.default_run_id()
    MODEL.set(run_id=run_id)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)

    @app.get('/health')
    def health() -> dict[str, Any]:
        model_state = MODEL.get()
        run_id = model_state.run_id
        return {'status': 'ok', 'run_id': run_id}

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        state = MODEL.get()
        if state.model is None:
            raise HTTPException(status_code=503, detail='Model is not loaded yet')

        needed = list(state.model.feature_names_in_)
        try:
            df = to_dataframe(request, needed_columns=needed)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        probability = float(state.model.predict_proba(df)[0][1])
        if not np.isfinite(probability):
            raise HTTPException(
                status_code=500,
                detail='Model returned non-finite probability',
            )
        prediction = int(probability >= 0.5)

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id
        try:
            MODEL.set(run_id=run_id)
        except MlflowException as e:
            logger.warning('MLflow error while loading run_id=%s: %s', run_id, e)
            code = 404 if getattr(e, 'error_code', None) == 'RESOURCE_DOES_NOT_EXIST' else 503
            raise HTTPException(
                status_code=code,
                detail=f'Could not load model from MLflow: {e}',
            ) from e
        except OSError as e:
            logger.warning('I/O error while loading run_id=%s: %s', run_id, e)
            raise HTTPException(
                status_code=503,
                detail=f'Model artifact could not be read: {e}',
            ) from e

        return UpdateModelResponse(run_id=run_id)

    return app


app = create_app()
