import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from mlflow.exceptions import MlflowException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ml_service import config
from ml_service import drift
from ml_service import metrics as prom
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
    prom.refresh_model_info(MODEL)
    st0 = MODEL.get()
    if st0.model is not None:
        drift.set_reference(st0.model, run_id)
    ev_task: asyncio.Task | None = None
    if config.evidently_enabled():
        ev_task = asyncio.create_task(drift.evidently_cron())
    yield
    if ev_task is not None:
        ev_task.cancel()
        try:
            await ev_task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)

    @app.middleware('http')
    async def prometheus_http_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        path = request.url.path
        method = request.method
        status = str(response.status_code)
        prom.http_requests_total.labels(method, path, status).inc()
        prom.http_request_duration_seconds.labels(method, path).observe(elapsed)
        return response

    @app.get('/metrics')
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

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
            t0 = time.perf_counter()
            df = to_dataframe(request, needed_columns=needed)
            prom.preprocess_duration_seconds.observe(time.perf_counter() - t0)
            prom.observe_feature_values(df)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        t1 = time.perf_counter()
        try:
            probability = float(state.model.predict_proba(df)[0][1])
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        prom.inference_duration_seconds.observe(time.perf_counter() - t1)
        if not np.isfinite(probability):
            raise HTTPException(
                status_code=500,
                detail='Model returned non-finite probability',
            )
        prediction = int(probability >= 0.5)
        prom.prediction_probability.observe(probability)
        prom.predictions_total.labels(prediction_class=str(prediction)).inc()
        if config.evidently_enabled():
            drift.append_row(df, prediction, probability)

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id
        try:
            MODEL.set(run_id=run_id)
        except MlflowException as e:
            logger.warning('MLflow error while loading run_id=%s: %s', run_id, e)
            code = 404 if getattr(e, 'error_code', None) == 'RESOURCE_DOES_NOT_EXIST' else 503
            if code == 404:
                prom.model_updates_total.labels(result='not_found').inc()
            else:
                prom.model_updates_total.labels(result='mlflow_error').inc()
            raise HTTPException(
                status_code=code,
                detail=f'Could not load model from MLflow: {e}',
            ) from e
        except OSError as e:
            logger.warning('I/O error while loading run_id=%s: %s', run_id, e)
            prom.model_updates_total.labels(result='io_error').inc()
            raise HTTPException(
                status_code=503,
                detail=f'Model artifact could not be read: {e}',
            ) from e

        prom.model_updates_total.labels(result='success').inc()
        prom.refresh_model_info(MODEL)
        st = MODEL.get()
        if st.model is not None and st.run_id is not None:
            drift.set_reference(st.model, st.run_id)
            drift.clear_buffer()
        return UpdateModelResponse(run_id=run_id)

    return app


app = create_app()
