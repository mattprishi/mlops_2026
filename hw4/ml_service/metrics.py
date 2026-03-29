from typing import TYPE_CHECKING

import pandas as pd
from prometheus_client import Counter, Histogram, Info, ProcessCollector, Summary

if TYPE_CHECKING:
    from ml_service.model import Model

ProcessCollector()

TIME_QUANTILES = (
    (0.75, 0.05),
    (0.9, 0.05),
    (0.95, 0.05),
    (0.99, 0.05),
    (0.999, 0.05),
)

http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'path', 'status_code'],
)

http_request_duration_seconds = Summary(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'path'],
    quantiles=TIME_QUANTILES,
)

preprocess_duration_seconds = Summary(
    'ml_preprocess_duration_seconds',
    'Time to build a dataframe for prediction',
    quantiles=TIME_QUANTILES,
)

inference_duration_seconds = Summary(
    'ml_inference_duration_seconds',
    'Model predict_proba latency in seconds',
    quantiles=TIME_QUANTILES,
)

feature_numeric_value = Histogram(
    'ml_feature_numeric_value',
    'Observed numeric feature values at inference',
    ['feature'],
    buckets=(
        0,
        1,
        5,
        10,
        17,
        25,
        40,
        55,
        65,
        80,
        100,
        250,
        500,
        1000,
        5000,
        10000,
        100000,
        1_000_000,
        float('inf'),
    ),
)

prediction_probability = Histogram(
    'ml_prediction_probability',
    'Distribution of predicted positive-class probability',
    buckets=[i / 20 for i in range(21)],
)

predictions_total = Counter(
    'ml_predictions_total',
    'Total model predictions',
    ['prediction_class'],
)

model_updates_total = Counter(
    'ml_model_updates_total',
    'Model reload attempts',
    ['result'],
)

model_metadata = Info('ml_model_metadata', 'Active model run, type, and feature list')


def model_type_name(model: object) -> str:
    if hasattr(model, 'steps') and getattr(model, 'steps', None):
        return type(model[-1]).__name__
    return type(model).__name__


def set_model_info(run_id: str, model: object, features: list[str]) -> None:
    feat_str = ','.join(features)
    if len(feat_str) > 2000:
        feat_str = feat_str[:1997] + '...'
    model_metadata.info(
        {
            'run_id': run_id,
            'model_type': model_type_name(model),
            'features': feat_str,
        }
    )


def observe_feature_values(df: pd.DataFrame) -> None:
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        feature_numeric_value.labels(feature=col).observe(float(series.iloc[0]))


def refresh_model_info(model_holder: 'Model') -> None:
    state = model_holder.get()
    if state.model is None or state.run_id is None:
        return
    feats = list(state.model.feature_names_in_)
    set_model_info(state.run_id, state.model, feats)


