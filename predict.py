# predict.py — Logique de prédiction image et vidéo

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Dict, Any

IMG_SIZE = 224

# Transformations image standard
TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


def extract_dct(img_array: np.ndarray) -> np.ndarray:
    """Image numpy RGB → carte DCT normalisée (1, H, W)."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dct  = cv2.dct(gray)
    dct  = np.log(np.abs(dct) + 1e-8)
    dct  = (dct - dct.min()) / (dct.max() - dct.min() + 1e-8)
    return dct[np.newaxis, :, :]  # (1, H, W)


def predict_single(model, img: Image.Image, device: str) -> Dict[str, Any]:
    """
    Prédit sur une image PIL.
    Retourne : verdict, score_real, score_fake, confidence
    """
    img    = img.convert("RGB")
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))

    img_t  = TRANSFORM(img).unsqueeze(0).to(device)
    dct    = extract_dct(img_np)
    freq_t = torch.tensor(dct).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(img_t, freq_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    score_real = float(probs[0])
    score_fake = float(probs[1])
    is_fake    = score_fake > 0.5

    return {
        "verdict"    : "deepfake" if is_fake else "authentic",
        "score_real" : round(score_real * 100, 2),
        "score_fake" : round(score_fake * 100, 2),
        "confidence" : round(max(score_real, score_fake) * 100, 2),
        "label"      : "Deepfake détecté" if is_fake else "Authentique",
    }


def predict_video(model, video_bytes: bytes, device: str,
                  sample_every: int = 10) -> Dict[str, Any]:
    """
    Prédit sur une vidéo (bytes).
    Analyse 1 frame sur sample_every — retourne verdict + timeline des scores.
    """
    # Écriture temporaire pour OpenCV
    tmp_path = "/tmp/deepguard_video.mp4"
    with open(tmp_path, "wb") as f:
        f.write(video_bytes)

    cap       = cv2.VideoCapture(tmp_path)
    fps       = cap.get(cv2.CAP_PROP_FPS) or 25
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scores    = []
    timeline  = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img   = Image.fromarray(rgb)
            res   = predict_single(model, img, device)
            scores.append(res["score_fake"])
            timeline.append({
                "time" : round(frame_idx / fps, 2),
                "score": res["score_fake"]
            })
        frame_idx += 1

    cap.release()

    if not scores:
        return {"error": "Aucune frame analysée"}

    mean_score = float(np.mean(scores))
    max_score  = float(np.max(scores))
    is_fake    = mean_score > 50.0

    return {
        "verdict"      : "deepfake" if is_fake else "authentic",
        "label"        : "Deepfake détecté" if is_fake else "Authentique",
        "score_fake"   : round(mean_score, 2),
        "score_real"   : round(100 - mean_score, 2),
        "confidence"   : round(max(mean_score, 100 - mean_score), 2),
        "max_score"    : round(max_score, 2),
        "frames_total" : total,
        "frames_analyzed": len(scores),
        "timeline"     : timeline,
        "fps"          : round(fps, 1),
    }
