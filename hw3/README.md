
Сравниваю три варианта инференса эмбеддингов на CPU: обычный PyTorch, ONNX Runtime и ONNX Runtime с динамическим батчированием.

Если папка onnx_model уже есть, экспорт можно не запускать. Если её нет, команда такая:
optimum-cli export onnx --model sergeyzh/rubert-mini-frida --task feature-extraction onnx_model/

Сервер запускается так:
uvicorn app:app --host 0.0.0.0 --port 8000

Есть два способа замера.

Первый способ — через Locust. Для этого нужно запустить:
locust -f locustfile.py --host=http://localhost:8000

Потом открыть http://localhost:8089 и выставить, например, Users = 50, Spawn rate = 1, Run time = 60-120s. В результатах смотреть Failures, 95%ile, 99%ile и Current RPS.

Второй способ — через встроенный скрипт:
python benchmark.py

Этот скрипт отправляет по 100 запросов на каждый эндпоинт с concurrency 10 и печатает RPS, p50, p95 и p99.

Текущие результаты benchmark.py:
Base PyTorch: RPS 150.4, p50 39.2 ms, p95 283.9 ms, p99 303.6 ms.
ONNX: RPS 268.9, p50 25.8 ms, p95 99.1 ms, p99 118.7 ms.
Dynamic batching: RPS 369.0, p50 26.4 ms, p95 30.9 ms, p99 30.9 ms.

Результаты Locust при 20 пользователях:
Base PyTorch: current RPS 72, median 11 ms, p95 24 ms, p99 53 ms.
ONNX: current RPS 68.5, median 10 ms, p95 23 ms, p99 51 ms.
Dynamic batching: current RPS 203.7, median 32 ms, p95 67 ms, p99 120 ms.

Результаты Locust при 50 пользователях:
Base PyTorch: current RPS 74.9, median 18 ms, p95 37 ms, p99 63 ms.
ONNX: current RPS 74.8, median 17 ms, p95 35 ms, p99 56 ms.
Dynamic batching: current RPS 222, median 130 ms, p95 210 ms, p99 270 ms.

Эти результаты в целом выглядят разумно. Базовый PyTorch оказался самым медленным по хвостовым задержкам. ONNX заметно ускорил инференс и снизил p95 и p99. Динамическое батчирование в коротком синтетическом тесте дало лучший throughput и лучшие перцентили.

По Locust картина тоже логичная. При 20 и 50 пользователях ONNX немного лучше или примерно на уровне обычного PyTorch по latency, а dynamic batching даёт сильно больший RPS. При этом latency у dynamic batching выше, особенно на 50 пользователях, потому что запросы ждут, пока соберётся батч. Это нормальное поведение для batching: он выигрывает по пропускной способности, но может проигрывать по задержке отдельного запроса.

Ещё один важный нюанс: в locustfile.py у dynamic batching вес 3, а у base и onnx по 1. Это значит, что на dynamic идёт в три раза больше трафика. Поэтому сравнивать в Locust нужно в первую очередь latency и failures, а не только абсолютный RPS между строками.

