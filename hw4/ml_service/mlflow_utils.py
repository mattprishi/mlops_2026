import json
import logging
import os

import mlflow
import numpy as np
import pandas as pd

from ml_service import config
from ml_service.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class MlflowModelWrapper:
    def __init__(self, model: mlflow.pyfunc.PyFuncModel, feature_names: list[str]) -> None:
        self._model = model
        self.feature_names_in_ = np.array(feature_names, dtype=object)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._model.predict(df[list(self.feature_names_in_)])
        if isinstance(raw, pd.DataFrame):
            if 'probability' in raw.columns:
                proba = raw['probability'].to_numpy(dtype=float)
            else:
                values = raw.to_numpy(dtype=float)
                proba = values[:, 1] if values.ndim == 2 and values.shape[1] > 1 else values.reshape(-1)
        elif isinstance(raw, pd.Series):
            proba = raw.to_numpy(dtype=float)
        else:
            values = np.asarray(raw, dtype=float)
            proba = values[:, 1] if values.ndim == 2 and values.shape[1] > 1 else values.reshape(-1)
        return np.column_stack([1.0 - proba, proba])


def configure_mlflow() -> None:
    uri = config.tracking_uri()
    if uri:
        mlflow.set_tracking_uri(uri)


def get_model_uri(run_id: str) -> str:
    return f'runs:/{run_id}/model'


def _get_feature_names(model: mlflow.pyfunc.PyFuncModel) -> list[str]:
    schema = model.metadata.get_input_schema()
    if schema is None or not schema.inputs:
        logger.warning(
            'Model input schema is missing in MLflow artifacts; fallback to full service feature list.'
        )
        return list(FEATURE_COLUMNS)
    return [str(col.name) for col in schema.inputs if col.name]


def load_model(model_uri: str = None, run_id: str = None) -> MlflowModelWrapper:
    """
    Downloads artifacts locally (if needed) and loads model as an MLflow PyFunc model.
    """
    if not model_uri:
        model_uri = get_model_uri(run_id)
    model = mlflow.pyfunc.load_model(model_uri)
    return MlflowModelWrapper(model=model, feature_names=_get_feature_names(model))


def load_input_example_dataframe(run_id: str, feature_names: list[str]) -> pd.DataFrame | None:
    """
    Loads logged input example for the run's model artifact (Model schema), if present.
    """
    root = mlflow.artifacts.download_artifacts(artifact_uri=get_model_uri(run_id))
    for fname in ('input_example.json', 'serving_input_example.json'):
        fp = os.path.join(root, fname)
        if not os.path.isfile(fp):
            continue
        with open(fp, encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            df = pd.DataFrame([raw])
        elif isinstance(raw, list) and raw:
            df = pd.DataFrame(raw)
        else:
            continue
        df.columns = [str(c) for c in df.columns]
        fn = [str(f) for f in feature_names]
        missing = [c for c in fn if c not in df.columns]
        if missing:
            continue
        return df[fn].copy()
    return None

