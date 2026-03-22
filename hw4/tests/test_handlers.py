from unittest.mock import patch

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from tests.conftest import make_fake_model, sample_payload_all_features


def test_health_returns_ok_and_run_id(client_full_model):
    r = client_full_model.get('/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'
    assert body['run_id'] == 'run-startup'


def test_predict_ok_probability_in_unit_interval(client_full_model):
    r = client_full_model.post('/predict', json=sample_payload_all_features())
    assert r.status_code == 200
    data = r.json()
    assert 'prediction' in data and 'probability' in data
    assert 0.0 <= data['probability'] <= 1.0
    assert data['prediction'] in (0, 1)


def test_predict_422_on_negative_feature(client_full_model):
    p = sample_payload_all_features()
    p['age'] = -5
    r = client_full_model.post('/predict', json=p)
    assert r.status_code == 422


def test_predict_400_when_model_needs_more_features(client_two_feature_model):
    r = client_two_feature_model.post('/predict', json={'age': 40})
    assert r.status_code == 400
    assert 'Missing values' in r.json()['detail']


def test_update_model_mocked_swap():
    features = ['age', 'sex']
    m_a = make_fake_model(features, proba_class_1=0.11)
    m_b = make_fake_model(features, proba_class_1=0.88)

    with patch(
        'ml_service.mlflow_utils.load_model',
        side_effect=[m_a, m_b],
    ):
        from ml_service.app import create_app

        with TestClient(create_app()) as client:
            r1 = client.post(
                '/predict',
                json={'age': 35, 'sex': 'Female'},
            )
            assert r1.status_code == 200
            assert abs(r1.json()['probability'] - 0.11) < 1e-9

            u = client.post('/updateModel', json={'run_id': 'other-run'})
            assert u.status_code == 200

            r2 = client.post(
                '/predict',
                json={'age': 35, 'sex': 'Female'},
            )
            assert r2.status_code == 200
            assert abs(r2.json()['probability'] - 0.88) < 1e-9


def test_update_model_keeps_old_model_on_mlflow_error():
    features = ['age', 'sex']
    m_a = make_fake_model(features, proba_class_1=0.5)

    err = MlflowException('run not found', error_code=RESOURCE_DOES_NOT_EXIST)

    with patch(
        'ml_service.mlflow_utils.load_model',
        side_effect=[m_a, err],
    ):
        from ml_service.app import create_app

        with TestClient(create_app()) as client:
            u = client.post('/updateModel', json={'run_id': 'bad-id'})
            assert u.status_code == 404

            r = client.post('/predict', json={'age': 20, 'sex': 'Male'})
            assert r.status_code == 200
            assert abs(r.json()['probability'] - 0.5) < 1e-9
