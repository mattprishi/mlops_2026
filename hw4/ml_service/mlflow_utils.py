import json
import logging
import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml_service import config
from ml_service.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def _sklearn_inner_from_pyfunc(pyfunc: mlflow.pyfunc.PyFuncModel) -> object | None:
    try:
        pm = pyfunc._model_impl.python_model
        return getattr(pm, 'sklearn_model', None)
    except Exception:
        return None


class MlflowModelWrapper:
    """
    Инференс через sklearn.predict_proba. Нужен полный Pipeline (сырые категории → числа).
    Если в runs:/.../model лежит только классификатор без препроцессинга, predict упадёт — см. сообщение в predict_proba.
    """

    def __init__(self, sklearn_model: object, feature_names: list[str]) -> None:
        self._sklearn = sklearn_model
        self.feature_names_in_ = np.array(feature_names, dtype=object)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        cols = list(self.feature_names_in_)
        X = df[cols]
        try:
            return self._sklearn.predict_proba(X)
        except ValueError as e:
            msg = str(e).lower()
            if 'could not convert' in msg or 'could not cast' in msg:
                raise RuntimeError(
                    'В MLflow у этого run залогирован не полный Pipeline, а модель без препроцессинга '
                    '(на входе ожидаются уже числа). Выберите run_id из эксперимента курса, '
                    'где в артефакте — sklearn Pipeline от сырых фич до предсказания.'
                ) from e
            raise


def configure_mlflow() -> None:
    uri = config.tracking_uri()
    if uri:
        mlflow.set_tracking_uri(uri)


def get_model_uri(run_id: str) -> str:
    return f'runs:/{run_id}/model'


def _get_feature_names_from_pyfunc(model: mlflow.pyfunc.PyFuncModel) -> list[str]:
    schema = model.metadata.get_input_schema()
    if schema is None or not schema.inputs:
        logger.warning(
            'Model input schema is missing in MLflow artifacts; fallback to full service feature list.'
        )
        return list(FEATURE_COLUMNS)
    return [str(col.name) for col in schema.inputs if col.name]


def _resolve_feature_names(sklearn_model: object, model_uri: str) -> list[str]:
    fn = getattr(sklearn_model, 'feature_names_in_', None)
    if fn is not None and len(fn) > 0:
        return list(fn)
    pyfunc = mlflow.pyfunc.load_model(model_uri)
    return _get_feature_names_from_pyfunc(pyfunc)


def load_model(model_uri: str = None, run_id: str = None) -> MlflowModelWrapper:
    """
    Сначала sklearn-артефакт; если это не Pipeline — пробуем вложенную sklearn-модель из pyfunc (часто там лежит Pipeline).
    """
    if not model_uri:
        model_uri = get_model_uri(run_id)
    sk = mlflow.sklearn.load_model(model_uri)
    if isinstance(sk, Pipeline):
        names = _resolve_feature_names(sk, model_uri)
        return MlflowModelWrapper(sklearn_model=sk, feature_names=names)

    pyfunc = mlflow.pyfunc.load_model(model_uri)
    inner = _sklearn_inner_from_pyfunc(pyfunc)
    if isinstance(inner, Pipeline):
        names = _resolve_feature_names(inner, model_uri)
        logger.info('Using sklearn Pipeline from pyfunc wrapper (sklearn flavor was a single estimator).')
        return MlflowModelWrapper(sklearn_model=inner, feature_names=names)

    names = _resolve_feature_names(sk, model_uri)
    logger.warning(
        'Артефакт не sklearn Pipeline — predict по сырым строкам, скорее всего, не сработает. '
        'Возьмите другой run_id с полным пайплайном.'
    )
    return MlflowModelWrapper(sklearn_model=sk, feature_names=names)


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
