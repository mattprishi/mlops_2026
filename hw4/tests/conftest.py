import numpy as np
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


def make_fake_model(feature_names: list[str], proba_class_1: float = 0.65):
    class FakeModel:
        def __init__(self) -> None:
            self.feature_names_in_ = np.array(feature_names, dtype=object)
            self._p = proba_class_1

        def predict_proba(self, X):
            p = self._p
            return np.array([[1.0 - p, p]])

    return FakeModel()


@pytest.fixture
def monkeypatch_env(monkeypatch):
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000/')
    monkeypatch.setenv('DEFAULT_RUN_ID', 'run-startup')


@pytest.fixture
def all_feature_names():
    return [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education.num',
        'marital.status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital.gain',
        'capital.loss',
        'hours.per.week',
        'native.country',
    ]


def sample_payload_all_features():
    return {
        'age': 39,
        'workclass': 'State-gov',
        'fnlwgt': 77516,
        'education': 'Bachelors',
        'education.num': 13,
        'marital.status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital.gain': 2174,
        'capital.loss': 0,
        'hours.per.week': 40,
        'native.country': 'United-States',
    }


@pytest.fixture
def client_full_model(monkeypatch_env, all_feature_names):
    fake = make_fake_model(all_feature_names)
    with patch('ml_service.mlflow_utils.load_model', return_value=fake):
        from ml_service.app import create_app

        with TestClient(create_app()) as c:
            yield c


@pytest.fixture
def client_two_feature_model(monkeypatch_env):
    fake = make_fake_model(['age', 'sex'], proba_class_1=0.72)
    with patch('ml_service.mlflow_utils.load_model', return_value=fake):
        from ml_service.app import create_app

        with TestClient(create_app()) as c:
            yield c
