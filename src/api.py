"""Local website + API.

    python run_site.py            (or: uvicorn api:app --app-dir src --port 8765)
    open http://localhost:8765
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from predictor import Predictor

WEB = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="MoMo Scam Check")
predictor = Predictor()


class Message(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@app.post("/api/predict")
def predict(msg: Message):
    return predictor.analyse(msg.text)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html", media_type="text/html")
