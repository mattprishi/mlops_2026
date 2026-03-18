from locust import HttpUser, between, task


class EmbeddingUser(HttpUser):
    wait_time = between(0.01, 0.05)
    payload = {"text": "Пример текста для генерации эмбеддингов в процессе тестирования."}

    @task(1)
    def test_base(self):
        self.client.post("/predict/base", json=self.payload, name="1_Base")

    @task(1)
    def test_onnx(self):
        self.client.post("/predict/onnx", json=self.payload, name="2_ONNX")

    @task(3)
    def test_dynamic(self):
        self.client.post("/predict/dynamic", json=self.payload, name="3_Dynamic")
