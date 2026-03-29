import json
import os

import mlflow
import pandas as pd

from ml_service import config


def configure_mlflow() -> None:
    uri = config.tracking_uri()
    if uri:
        mlflow.set_tracking_uri(uri)


def get_model_uri(run_id: str) -> str:
    return f'runs:/{run_id}/model'


def load_model(model_uri: str = None, run_id: str = None) -> mlflow.pyfunc.PyFuncModel:
    """
    Downloads artifacts locally (if needed) and loads model as an MLflow PyFunc model.
    """
    if not model_uri:
        model_uri = get_model_uri(run_id)
    return mlflow.sklearn.load_model(model_uri)


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

