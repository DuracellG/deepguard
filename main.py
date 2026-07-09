import io
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model import load_model
from predict import predict_image

logger = logging.getLogger("deepguard")

DEVICE     = "cpu"
MODEL      = None
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pth")
MAX_BYTES  = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    if os.path.exists(MODEL_PATH):
        MODEL = load_model(MODEL_PATH, DEVICE)
        print("[OK] Modele pret")
    yield


app = FastAPI(title="DeepGuard", version="3.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


# Endpoint volontairement synchrone : FastAPI l'exécute dans un threadpool,
# l'inférence PyTorch ne bloque donc pas la boucle d'événements (/health reste réactif).
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(503, "Modèle non disponible")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Image requise (JPG, PNG, WEBP)")
    if file.size is not None and file.size > MAX_BYTES:
        raise HTTPException(400, "Fichier trop volumineux (max 10 MB)")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(400, "Fichier trop volumineux (max 10 MB)")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(400, "Fichier image illisible ou corrompu")
    try:
        result = predict_image(MODEL, img, DEVICE)
        result["filename"] = file.filename
        return JSONResponse(result)
    except Exception:
        logger.exception("Échec de l'analyse de %s", file.filename)
        raise HTTPException(500, "Erreur interne pendant l'analyse")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
@app.head("/")
def root():
    return FileResponse("static/index.html")
