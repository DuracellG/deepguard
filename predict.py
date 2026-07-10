import cv2, numpy as np, torch, base64, io, os, threading
import torchvision.transforms as T
from PIL import Image, ImageDraw

IMG_SIZE        = 224
THRESHOLD       = 0.75   # au-delà : deepfake
MAX_ANNOT_SIDE  = 800    # taille max de l'image annotée renvoyée en base64
MAX_DETECT_SIDE = 1280   # taille max soumise au détecteur (les boîtes sont remises à l'échelle)
MAX_WORK_SIDE   = 1600   # résolution de travail max : borne la mémoire du pipeline
                         # (une photo 4000×3000 = ~36 MB par copie, OOM sur 512 MB)

TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Détecteur de visages YuNet (DNN) — robuste aux profils et têtes inclinées,
# contrairement au Haar Cascade frontal utilisé auparavant.
YUNET_PATH = os.path.join(os.path.dirname(__file__), "face_detection_yunet.onnx")
FACE_DETECTOR = cv2.FaceDetectorYN.create(
    YUNET_PATH, "", (320, 320), score_threshold=0.6)
# FaceDetectorYN n'est pas thread-safe ; l'endpoint tourne dans un threadpool.
_DETECTOR_LOCK = threading.Lock()

COLORS = {
    "deepfake" : (192, 57, 43),   # rouge
    "authentic": (26, 122, 74),   # vert
}
# ASCII uniquement : la police bitmap par défaut de PIL ne rend pas ⚠ / ✓
BADGES = {
    "deepfake" : "! DEEPFAKE",
    "authentic": "AUTHENTIQUE",
}


def extract_dct(arr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dct  = cv2.dct(gray)
    dct  = np.log(np.abs(dct) + 1e-8)
    dct  = (dct - dct.min()) / (dct.max() - dct.min() + 1e-8)
    return dct[np.newaxis, :, :]


def dct_high_freq_ratio(arr: np.ndarray) -> float:
    """Part de l'énergie du spectre DCT située hors du bloc basses fréquences
    (quart supérieur gauche). Métrique réellement mesurée, affichée à l'utilisateur."""
    gray  = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    d     = np.abs(cv2.dct(gray))
    k     = d.shape[0] // 4
    total = d.sum() + 1e-8
    return float((total - d[:k, :k].sum()) / total)


def detect_faces(img_rgb: np.ndarray):
    """Détecte les visages avec YuNet — retourne uniquement le visage principal
    (le plus grand), sous forme de liste [(x, y, w, h)]."""
    h, w = img_rgb.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_DETECT_SIDE:
        scale   = MAX_DETECT_SIDE / max(h, w)
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))
        h, w    = img_rgb.shape[:2]
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    with _DETECTOR_LOCK:
        FACE_DETECTOR.setInputSize((w, h))
        _, faces = FACE_DETECTOR.detect(bgr)

    if faces is None or len(faces) == 0:
        return []
    # Chaque ligne : x, y, w, h, 10 landmarks, score. Garder le plus grand visage.
    best = max(faces, key=lambda f: f[2] * f[3])
    x, y, bw, bh = (best[:4] / scale).astype(int)
    return [(max(0, x), max(0, y), bw, bh)]


def draw_face_box(img_pil: Image.Image, faces, verdict: str) -> str:
    """
    Dessine un encadrement visible autour du visage principal.
    Vert = Authentique | Rouge = Deepfake
    Retourne l'image encodée en base64 PNG (redimensionnée à 800 px max).
    """
    color = COLORS[verdict]
    img   = img_pil.copy().convert("RGB")
    draw  = ImageDraw.Draw(img)

    if len(faces) > 0:
        x, y, w, h = [int(v) for v in faces[0]]
        # Épaisseur proportionnelle à la taille de l'image
        lw = max(4, w // 30)
        cs = w // 5  # longueur des coins

        # Rectangle principal semi-transparent via numpy
        arr = np.array(img)
        # Bordure épaisse du rectangle
        cv2.rectangle(arr, (x, y), (x+w, y+h), color, lw)
        img  = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)

        # Coins épais en L (plus visibles)
        lw2 = lw + 3
        for px, py, dx, dy in [
            (x, y, 1, 1), (x+w, y, -1, 1),
            (x, y+h, 1, -1), (x+w, y+h, -1, -1)
        ]:
            draw.line([(px, py), (px+dx*cs, py)], fill=color, width=lw2)
            draw.line([(px, py), (px, py+dy*cs)], fill=color, width=lw2)

        # Badge verdict au-dessus du visage
        label = BADGES[verdict]
        pad   = 8
        tw    = len(label) * 9
        th    = 22
        ty    = max(0, y - th - pad)
        draw.rectangle([x, ty, x + tw + pad*2, ty + th + pad*2],
                       fill=color)
        draw.text((x + pad, ty + pad), label, fill="white")

    # Limiter la taille avant encodage : l'UI affiche l'image en ~200 px,
    # inutile de renvoyer un PNG pleine résolution dans le JSON.
    if max(img.size) > MAX_ANNOT_SIDE:
        img.thumbnail((MAX_ANNOT_SIDE, MAX_ANNOT_SIDE))

    # Encode en base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def dct_message(verdict: str, hf_ratio: float, fake_pct: float) -> str:
    """Message d'explication basé sur des valeurs réellement mesurées."""
    pct = round(hf_ratio * 100, 1)
    base = (f"Énergie mesurée hors basses fréquences : {pct} % du spectre DCT. "
            f"Le classifieur hybride (spatial + fréquentiel) estime la probabilité "
            f"de manipulation à {round(fake_pct, 1)} %.")
    if verdict == "deepfake":
        return base + " Ce score dépasse le seuil de décision (75 %)."
    return base + " Ce score est sous le seuil de décision (75 %)."


def predict_image(model, img: Image.Image, device: str) -> dict:
    img = img.convert("RGB")
    # Borner la résolution de travail : l'analyse se fait sur un crop 224×224,
    # aucune précision utile n'est perdue et la mémoire reste bornée.
    if max(img.size) > MAX_WORK_SIDE:
        img.thumbnail((MAX_WORK_SIDE, MAX_WORK_SIDE))
    arr_orig = np.array(img)

    # --- Détection du visage principal (affichage du cadre uniquement) ---
    faces = detect_faces(arr_orig)

    # --- Analyse sur l'image ENTIÈRE ---
    # Le modèle est entraîné sur le dataset AI Face (portraits cadrés avec
    # arrière-plan, spécialisation modèles de diffusion) : l'inférence doit
    # recevoir l'image entière. Vérifié empiriquement sur Macron_2.png :
    # 87,11 % de score deepfake en image entière, 0 % sur crop serré.
    arr_224 = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    img_t   = TRANSFORM(img).unsqueeze(0).to(device)
    freq_t  = torch.tensor(extract_dct(arr_224)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        probs = torch.softmax(model(img_t, freq_t), 1).cpu().numpy()[0]

    real, fake = float(probs[0]), float(probs[1])
    is_fake    = fake > THRESHOLD
    verdict    = "deepfake" if is_fake else "authentic"
    label      = "Deepfake détecté" if is_fake else "Authentique"

    if fake < 0.50:   risk = "Faible"
    elif fake < 0.75: risk = "Modéré"
    else:             risk = "Élevé"

    hf_ratio      = dct_high_freq_ratio(arr_224)
    annotated_img = draw_face_box(img, faces, verdict)

    return {
        "verdict"        : verdict,
        "label"          : label,
        "score_real"     : round(real * 100, 2),
        "score_fake"     : round(fake * 100, 2),
        "confidence"     : round(max(real, fake) * 100, 2),
        "risk"           : risk,
        "faces_count"    : len(faces),
        "dct_analysis"   : dct_message(verdict, hf_ratio, fake * 100),
        "annotated_img"  : annotated_img,   # base64 PNG avec boîtes
    }
