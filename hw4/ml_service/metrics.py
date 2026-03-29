from typing import TYPE_CHECKING

import pandas as pd
from prometheus_client import Counter, Histogram, Info

if TYPE_CHECKING:
    from ml_service.model import Model

# Секунды; перцентили в Grafana: histogram_quantile(phi, sum(rate(..._bucket[5m])) by (le))
LATENCY_BUCKETS = (
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'path', 'status_code'],
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'path'],
    buckets=LATENCY_BUCKETS,
)

preprocess_duration_seconds = Histogram(
    'ml_preprocess_duration_seconds',
    'Time to build a dataframe for prediction',
    buckets=LATENCY_BUCKETS,
)

inference_duration_seconds = Histogram(
    'ml_inference_duration_seconds',
    'Model predict_proba latency in seconds',
    buckets=LATENCY_BUCKETS,
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
    inner = getattr(model, '_sklearn', model)
    if hasattr(inner, 'steps') and getattr(inner, 'steps', None):
        return type(inner[-1]).__name__
    return type(inner).__name__


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
