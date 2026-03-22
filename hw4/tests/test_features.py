import pytest
from pydantic import ValidationError

from ml_service.features import to_dataframe
from ml_service.schemas import PredictRequest


def test_to_dataframe_keeps_only_needed_columns_and_order():
    req = PredictRequest(
        age=30,
        sex='Male',
        workclass='Private',
        fnlwgt=100,
        education='HS-grad',
        education_num=9,
        marital_status='Never-married',
        occupation='Craft-repair',
        relationship='Own-child',
        race='White',
        capital_gain=0,
        capital_loss=0,
        hours_per_week=40,
        native_country='United-States',
    )
    needed = ['sex', 'age']
    df = to_dataframe(req, needed_columns=needed)
    assert list(df.columns) == ['sex', 'age']
    assert df.shape == (1, 2)
    assert df.iloc[0]['age'] == 30
    assert df.iloc[0]['sex'] == 'Male'


def test_to_dataframe_raises_when_required_value_missing():
    req = PredictRequest(age=30)
    with pytest.raises(ValueError, match='Missing values'):
        to_dataframe(req, needed_columns=['age', 'sex'])


def test_to_dataframe_raises_on_unknown_feature():
    req = PredictRequest(age=30, sex='Male')
    with pytest.raises(ValueError, match='Unknown feature names'):
        to_dataframe(req, needed_columns=['age', 'unknown_col'])


def test_predict_request_rejects_negative_age():
    with pytest.raises(ValidationError):
        PredictRequest(age=-1, sex='Male')
