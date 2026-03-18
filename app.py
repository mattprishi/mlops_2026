import asyncio
import time
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForFeatureExtraction
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sergeyzh/rubert-mini-frida"
ONNX_PATH = "onnx_model/"

tokenizer = None
model_base = None
model_onnx = None
batch_queue = None

MAX_BATCH_SIZE = 16
MAX_WAIT_TIME = 0.01  # 10 мс


class TextRequest(BaseModel):
    text: str


def _to_tensor(x):
    return torch.tensor(x) if not isinstance(x, torch.Tensor) else x


def mean_pooling(model_output, attention_mask):
    token_embeddings = _to_tensor(model_output[0])
    mask = _to_tensor(attention_mask)
    input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return F.normalize(sum_embeddings / sum_mask, p=2, dim=1)


async def batch_worker():
    while True:
        batch = []
        try:
            item = await batch_queue.get()
            batch.append(item)

            end_time = time.monotonic() + MAX_WAIT_TIME
            while len(batch) < MAX_BATCH_SIZE:
                timeout = end_time - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(batch_queue.get(), timeout=timeout)
                    batch.append(next_item)
                except asyncio.TimeoutError:
                    break

            texts = [req["text"] for req in batch]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            outputs = model_onnx(**inputs)
            embeddings = mean_pooling(outputs, inputs["attention_mask"])

            for req, emb in zip(batch, embeddings.tolist()):
                req["future"].set_result(emb)

        except Exception as e:
            for req in batch:
                if not req["future"].done():
                    req["future"].set_exception(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model_base, model_onnx, batch_queue

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_base = AutoModel.from_pretrained(MODEL_NAME)
    model_base.eval()

    model_onnx = ORTModelForFeatureExtraction.from_pretrained(ONNX_PATH)

    batch_queue = asyncio.Queue()
    worker_task = asyncio.create_task(batch_worker())

    yield
    worker_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.post("/predict/base")
async def predict_base(req: TextRequest):
    inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_base(**inputs)
    emb = mean_pooling(outputs, inputs["attention_mask"])
    return {"embedding": emb[0].tolist()}


@app.post("/predict/onnx")
async def predict_onnx(req: TextRequest):
    inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True)
    outputs = model_onnx(**inputs)
    emb = mean_pooling(outputs, inputs["attention_mask"])
    return {"embedding": emb[0].tolist()}


@app.post("/predict/dynamic")
async def predict_dynamic(req: TextRequest):
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    await batch_queue.put({"text": req.text, "future": future})

    embedding = await future
    return {"embedding": embedding}
