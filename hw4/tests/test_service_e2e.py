"""
Полный цикл: старт с мок-моделью, предсказание, смена модели, снова предсказание.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from tests.conftest import make_fake_model, sample_payload_all_features


def test_predict_then_update_then_predict():
    sklearn_names = [
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

    m1 = make_fake_model(sklearn_names, proba_class_1=0.2)
    m2 = make_fake_model(sklearn_names, proba_class_1=0.95)

    with patch(
        'ml_service.mlflow_utils.load_model',
        side_effect=[m1, m2],
    ):
        from ml_service.app import create_app

        with TestClient(create_app()) as client:
            p = sample_payload_all_features()
            r0 = client.post('/predict', json=p)
            assert r0.status_code == 200
            assert abs(r0.json()['probability'] - 0.2) < 1e-9

            up = client.post('/updateModel', json={'run_id': 'run-second'})
            assert up.status_code == 200

            r1 = client.post('/predict', json=p)
            assert r1.status_code == 200
            assert abs(r1.json()['probability'] - 0.95) < 1e-9
