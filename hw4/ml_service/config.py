import os

MODEL_ARTIFACT_PATH = 'model'

EVIDENTLY_URL_DEFAULT = 'http://158.160.2.37:8000/'
EVIDENTLY_PROJECT_DEFAULT = '019d061f-cc08-7b5e-b932-d792a1f258e2'


def tracking_uri() -> str:
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not tracking_uri:
        raise RuntimeError('Please set MLFLOW_TRACKING_URI')
    return tracking_uri


def default_run_id() -> str:
    """
    Returns model URI for startup.
    """

    default_run_id = os.getenv('DEFAULT_RUN_ID')
    if not default_run_id:
        raise RuntimeError('Set DEFAULT_RUN_ID to load model on startup')
    return default_run_id


def evidently_enabled() -> bool:
    return os.getenv('EVIDENTLY_ENABLED', 'true').lower() in ('1', 'true', 'yes')


def evidently_url() -> str:
    return os.getenv('EVIDENTLY_URL', EVIDENTLY_URL_DEFAULT).rstrip('/') + '/'


def evidently_project_id() -> str:
    return os.getenv('EVIDENTLY_PROJECT_ID', EVIDENTLY_PROJECT_DEFAULT)


def evidently_interval_sec() -> int:
    return int(os.getenv('EVIDENTLY_INTERVAL_SEC', '600'))


def evidently_min_samples() -> int:
    return int(os.getenv('EVIDENTLY_MIN_SAMPLES', '50'))


def evidently_buffer_max() -> int:
    return int(os.getenv('EVIDENTLY_BUFFER_MAX', '5000'))
