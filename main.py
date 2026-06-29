import os, io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from model import load_model
from predict import predict_image

app        = FastAPI(title="DeepGuard", version="2.0")
DEVICE     = "cpu"
MODEL      = None
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pth")

@app.on_event("startup")
def startup():
    global MODEL
    if os.path.exists(MODEL_PATH):
        MODEL = load_model(MODEL_PATH, DEVICE)
        print("✅ Modèle prêt")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(503, "Modèle non disponible")
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Image requise (JPG, PNG, WEBP)")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, "Fichier trop volumineux (max 10 MB)")
    try:
        img    = Image.open(io.BytesIO(data))
        result = predict_image(MODEL, img, DEVICE)
        result["filename"] = file.filename
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, f"Erreur : {str(e)}")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
@app.head("/")
def root():
    return FileResponse("static/index.html")
