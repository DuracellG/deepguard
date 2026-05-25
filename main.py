# main.py — API FastAPI DeepGuard

import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from model import load_model
from predict import predict_single, predict_video

# ── Configuration ─────────────────────────────────────────────
app        = FastAPI(title="DeepGuard API", version="1.0")
DEVICE     = "cpu"
MODEL      = None
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pth")


@app.on_event("startup")
def startup():
    """Charge le modèle depuis le repo au démarrage."""
    global MODEL
    if os.path.exists(MODEL_PATH):
        MODEL = load_model(MODEL_PATH, DEVICE)
        print(f"✅ Modèle prêt")
    else:
        print(f"❌ Modèle introuvable : {MODEL_PATH}")


# ── Routes API ────────────────────────────────────────────────

@app.get("/health")
def health():
    """Vérification état du serveur et du modèle."""
    return {
        "status"      : "ok",
        "model_loaded": MODEL is not None,
        "device"      : DEVICE
    }


@app.get("/debug")
def debug():
    """Test rapide du modèle sur une image noire synthétique."""
    import torch
    import numpy as np
    from PIL import Image as PILImage
    if MODEL is None:
        return {"error": "Modèle non chargé"}
    img    = PILImage.fromarray(
        __import__('numpy').zeros((224, 224, 3), dtype=__import__('numpy').uint8))
    result = predict_single(MODEL, img, DEVICE)
    return result


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Analyse une image — retourne verdict + scores."""
    if MODEL is None:
        raise HTTPException(503, "Modèle non disponible")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Fichier image requis (JPG, PNG, WEBP)")

    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, "Fichier trop volumineux (max 10 MB)")

    try:
        img    = Image.open(io.BytesIO(data))
        result = predict_single(MODEL, img, DEVICE)
        result["filename"] = file.filename
        result["type"]     = "image"
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"Erreur analyse : {str(e)}")


@app.post("/predict/video")
async def predict_video_route(file: UploadFile = File(...)):
    """Analyse une vidéo frame par frame — retourne verdict + timeline."""
    if MODEL is None:
        raise HTTPException(503, "Modèle non disponible")

    allowed = ["video/mp4", "video/avi", "video/quicktime",
               "video/x-matroska", "video/webm", "video/x-msvideo"]
    if file.content_type not in allowed:
        raise HTTPException(400, "Format vidéo non supporté (MP4, AVI, MOV)")

    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(400, "Fichier trop volumineux (max 50 MB)")

    try:
        result = predict_video(MODEL, data, DEVICE)
        result["filename"] = file.filename
        result["type"]     = "video"
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"Erreur analyse vidéo : {str(e)}")


# ── Interface web ─────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
@app.head("/")
def root():
    return FileResponse("static/index.html")