MLflow и FastAPI: при старте подтягивается пайплайн из указанного run, ручка predict отдаёт класс и вероятность, updateModel меняет модель по новому run_id.

Нужны переменные MLFLOW_TRACKING_URI и DEFAULT_RUN_ID. Запуск через docker compose из этого каталога, снаружи порт смотри в compose-файле.

Тесты: устанавливаем зависимости из requirements.txt и выполняем pytest в корне проекта.
