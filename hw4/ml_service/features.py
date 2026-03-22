import pandas as pd

from ml_service.schemas import PredictRequest


FEATURE_COLUMNS = [
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


def _attr_for_column(column: str) -> str:
    return column.replace('.', '_')


def to_dataframe(req: PredictRequest, needed_columns: list[str]) -> pd.DataFrame:
    unknown = [c for c in needed_columns if c not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f'Unknown feature names for this service: {sorted(unknown)}')

    columns = [c for c in needed_columns if c in FEATURE_COLUMNS]
    missing: list[str] = []
    row: list = []
    for column in columns:
        val = getattr(req, _attr_for_column(column))
        if val is None or (isinstance(val, float) and pd.isna(val)):
            missing.append(column)
        row.append(val)
    if missing:
        raise ValueError(
            f'Missing values for features required by the current model: {sorted(missing)}'
        )

    return pd.DataFrame([row], columns=columns)
