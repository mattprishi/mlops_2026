"""
Скрипт бенчмарка с замером latency, throughput и потребления ресурсов (CPU, RAM).
Запускаем при уже работающем сервере (uvicorn app:app --port 8000).
"""
import asyncio
import sys
import time

import psutil
import aiohttp

BASE_URL = "http://localhost:8000"
PAYLOAD = {"text": "Пример текста для генерации эмбеддингов."}
NUM_REQUESTS = 100
CONCURRENCY = 10


async def fetch(session, url, json):
    start = time.perf_counter()
    async with session.post(url, json=json) as resp:
        await resp.json()
    return (time.perf_counter() - start) * 1000


async def run_benchmark(endpoint: str):
    latencies = []
    start_wall = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def bounded_fetch():
            async with sem:
                return await fetch(session, f"{BASE_URL}{endpoint}", PAYLOAD)

        tasks = [bounded_fetch() for _ in range(NUM_REQUESTS)]
        latencies = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start_wall
    latencies.sort()

    p50 = latencies[int(len(latencies) * 0.5)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    rps = NUM_REQUESTS / elapsed

    return {
        "rps": rps,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "elapsed_s": elapsed,
    }


def main():
    endpoints = [
        ("/predict/base", "Часть 1: Base PyTorch"),
        ("/predict/onnx", "Часть 2: ONNX"),
        ("/predict/dynamic", "Часть 3: Dynamic Batching"),
    ]

    print("Бенчмарк (CPU/RAM — смотрите в отдельном терминале: top или htop)\n")

    for path, name in endpoints:
        print(f"--- {name} ---")
        try:
            result = asyncio.run(run_benchmark(path))
            print(f"  RPS: {result['rps']:.1f}")
            print(f"  Latency p50: {result['p50_ms']:.1f} ms")
            print(f"  Latency p95: {result['p95_ms']:.1f} ms")
            print(f"  Latency p99: {result['p99_ms']:.1f} ms")
        except Exception as e:
            print(f"  Ошибка: {e}")
        print()


if __name__ == "__main__":
    main()
