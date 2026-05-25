# main.py — API FastAPI DeepGuard

import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from model import load_model
from predict import predict_single, predict_video

# ── Initialisation ────────────────────────────────────────────
app    = FastAPI(title="DeepGuard API", version="1.0")
DEVICE = "cpu"   # Render gratuit = CPU uniquement
MODEL  = None    # Chargé au démarrage

MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pth")


@app.on_event("startup")
def startup():
    global MODEL
    if os.path.exists(MODEL_PATH):
        MODEL = load_model(MODEL_PATH, DEVICE)
    else:
        print(f"⚠️  Modèle introuvable : {MODEL_PATH}")


# ── Routes API ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Analyse une image — retourne verdict + scores."""
    if MODEL is None:
        raise HTTPException(503, "Modèle non disponible")

    # Vérification type fichier
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Fichier image requis (JPG, PNG, WEBP)")

    try:
        data  = await file.read()
        img   = Image.open(io.BytesIO(data))
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

    # Vérification type fichier
    allowed = ["video/mp4", "video/avi", "video/quicktime",
               "video/x-matroska", "video/webm"]
    if file.content_type not in allowed:
        raise HTTPException(400, "Format vidéo non supporté (MP4, AVI, MOV)")

    try:
        data   = await file.read()
        result = predict_video(MODEL, data, DEVICE)
        result["filename"] = file.filename
        result["type"]     = "video"
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"Erreur analyse vidéo : {str(e)}")


# ── Interface web ─────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
