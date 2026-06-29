import cv2, numpy as np, torch, base64, io
import torchvision.transforms as T
from PIL import Image, ImageDraw

IMG_SIZE  = 224
THRESHOLD = 0.75

TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Chargement du détecteur de visages Haar Cascade (inclus dans OpenCV)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def extract_dct(arr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dct  = cv2.dct(gray)
    dct  = np.log(np.abs(dct) + 1e-8)
    dct  = (dct - dct.min()) / (dct.max() - dct.min() + 1e-8)
    return dct[np.newaxis, :, :]


def detect_faces(img_rgb: np.ndarray):
    """Détecte les visages — retourne uniquement le visage principal (le plus grand)."""
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray  = cv2.equalizeHist(gray)  # améliore la détection sur peaux foncées
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=8,      # plus strict → moins de faux positifs
        minSize=(80, 80),    # ignorer les très petits détections parasites
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return []
    # Garder uniquement le visage le plus grand (surface w*h)
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    return [faces[0]]


def draw_face_box(img_pil: Image.Image, faces, verdict: str) -> str:
    """
    Dessine un encadrement visible autour du visage principal.
    Vert = Authentique | Rouge = Deepfake
    Retourne l'image encodée en base64 PNG.
    """
    color = (192, 57, 43) if verdict == "deepfake" else (26, 122, 74)
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
        label = "⚠ DEEPFAKE" if verdict == "deepfake" else "✓ AUTHENTIQUE"
        pad   = 8
        tw    = len(label) * 9
        th    = 22
        ty    = max(0, y - th - pad)
        draw.rectangle([x, ty, x + tw + pad*2, ty + th + pad*2],
                       fill=color)
        draw.text((x + pad, ty + pad), label, fill="white")

    # Encode en base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def predict_image(model, img: Image.Image, device: str) -> dict:
    img    = img.convert("RGB")
    arr_orig = np.array(img)

    # --- Analyse sur image redimensionnée ---
    arr_224 = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    img_t   = TRANSFORM(img).unsqueeze(0).to(device)
    freq_t  = torch.tensor(extract_dct(arr_224)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        probs = torch.softmax(model(img_t, freq_t), 1).cpu().numpy()[0]

    real, fake = float(probs[0]), float(probs[1])
    is_fake    = fake > THRESHOLD
    verdict    = "deepfake" if is_fake else "authentic"

    if fake < 0.50:   risk = "Faible"
    elif fake < 0.75: risk = "Modéré"
    else:             risk = "Élevé"

    # --- Détection et délimitation des visages ---
    faces         = detect_faces(arr_orig)
    annotated_img = draw_face_box(img, faces, verdict)
    faces_count   = len(faces) if len(faces) > 0 else 0

    return {
        "verdict"      : verdict,
        "label"        : "Deepfake détecté" if is_fake else "Authentique",
        "score_real"   : round(real * 100, 2),
        "score_fake"   : round(fake * 100, 2),
        "confidence"   : round(max(real, fake) * 100, 2),
        "risk"         : risk,
        "faces_count"  : faces_count,
        "annotated_img": annotated_img,   # base64 PNG avec boîtes
    }
