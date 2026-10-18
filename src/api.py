"""FastAPI service.  Run:  uvicorn api:app --app-dir src --reload"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from predictor import Predictor

app = FastAPI(title="MoMo Scam SMS Detector")
predictor = Predictor()


class Message(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@app.post("/predict")
def predict(msg: Message):
    return predictor.analyse(msg.text)


@app.get("/health")
def health():
    return {"ok": True}
