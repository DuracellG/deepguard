# main.py — API FastAPI DeepGuard

import os
import io
import gdown
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
MODEL_URL  = os.getenv(
    "MODEL_URL",
    "https://drive.google.com/uc?id=1XUly96BwbC8xUyZQamQQl91DZbkho8f9"
)


@app.on_event("startup")
def startup():
    """Télécharge le modèle depuis Google Drive si absent, puis le charge."""
    global MODEL

    if not os.path.exists(MODEL_PATH):
        print(f"📥 Modèle absent — téléchargement depuis Google Drive...")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
            print("✅ Téléchargement terminé")
        except Exception as e:
            print(f"❌ Échec téléchargement : {e}")
            return

    if os.path.exists(MODEL_PATH):
        MODEL = load_model(MODEL_PATH, DEVICE)
    else:
        print("❌ Modèle introuvable après téléchargement")


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
    from PIL import Image
    from predict import predict_single

    # Image noire 224x224
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = predict_single(MODEL, img, DEVICE)
    return result

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Analyse une image — retourne verdict + scores."""
    if MODEL is None:
        raise HTTPException(503, "Modèle non disponible — réessayez dans quelques secondes")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Fichier image requis (JPG, PNG, WEBP)")

    # Limite taille : 10 MB
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
        raise HTTPException(503, "Modèle non disponible — réessayez dans quelques secondes")

    allowed = ["video/mp4", "video/avi", "video/quicktime",
               "video/x-matroska", "video/webm", "video/x-msvideo"]
    if file.content_type not in allowed:
        raise HTTPException(400, "Format vidéo non supporté (MP4, AVI, MOV, MKV)")

    # Limite taille : 100 MB
    data = await file.read()
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(400, "Fichier trop volumineux (max 100 MB)")

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
def root():
    return FileResponse("static/index.html")