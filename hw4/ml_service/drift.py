"""
Накопление онлайн-данных и периодическая отправка отчётов Evidently (дрифт фичей и предсказаний).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from evidently.ui.workspace import RemoteWorkspace

from ml_service import config
from ml_service.mlflow_utils import load_input_example_dataframe

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_buffer: deque[dict] = deque(maxlen=config.evidently_buffer_max())
_reference: pd.DataFrame | None = None


def _expand_reference(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    if len(df) >= min_rows:
        return df.copy()
    rng = np.random.default_rng(42)
    reps = []
    i = 0
    feat_cols = [c for c in df.columns if c not in ('prediction', 'probability')]
    while len(reps) < min_rows:
        row = df.iloc[i % len(df)].copy()
        for col in feat_cols:
            if pd.api.types.is_numeric_dtype(df[col]) and pd.notna(row[col]):
                v = float(row[col])
                if np.isfinite(v):
                    row[col] = v + rng.normal(0, max(abs(v) * 0.02, 1e-6))
        reps.append(row)
        i += 1
    return pd.DataFrame(reps)


def set_reference(model, run_id: str) -> None:
    global _reference
    feats = list(model.feature_names_in_)
    base = load_input_example_dataframe(run_id, feats)
    if base is None or base.empty:
        logger.warning(
            'Evidently: нет input_example у runs:/%s/model — дрифт не настроен.',
            run_id,
        )
        _reference = None
        return
    df = base[feats].copy()
    proba = model.predict_proba(df)[:, 1]
    df = df.assign(probability=proba, prediction=(proba >= 0.5).astype(int))
    need = max(config.evidently_min_samples(), len(df))
    _reference = _expand_reference(df, need)


def append_row(features_df: pd.DataFrame, prediction: int, probability: float) -> None:
    row = features_df.iloc[0].to_dict()
    row['prediction'] = prediction
    row['probability'] = float(probability)
    with _lock:
        _buffer.append(row)


def clear_buffer() -> None:
    with _lock:
        _buffer.clear()


async def evidently_cron() -> None:
    while True:
        await asyncio.sleep(config.evidently_interval_sec())
        if not config.evidently_enabled():
            continue
        ref = _reference
        if ref is None:
            continue
        with _lock:
            if len(_buffer) < config.evidently_min_samples():
                continue
            current = pd.DataFrame(list(_buffer))
            _buffer.clear()
        current = current.reindex(columns=ref.columns)
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref, current_data=current)
            ws = RemoteWorkspace(config.evidently_url())
            ws.add_report(config.evidently_project_id(), report)
            logger.info('Evidently: отчёт отправлен в проект %s', config.evidently_project_id())
        except Exception:
            logger.exception('Evidently: не удалось построить или отправить отчёт')
