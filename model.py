# model.py — Définition et chargement du HybridDeepfakeDetector

import torch
import torch.nn as nn
import timm


class HybridDeepfakeDetector(nn.Module):

    def __init__(self):
        super().__init__()
        # Branche spatiale : EfficientNet-B0 sans tête de classification
        self.spatial = timm.create_model(
            'efficientnet_b0', pretrained=False, num_classes=0)

        # Branche fréquentielle : CNN léger sur carte DCT (1 canal)
        self.freq = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )

        # Tête de classification fusionnée (1280 + 64 features)
        self.head = nn.Sequential(
            nn.Linear(1280 + 64, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, img, freq):
        f1 = self.spatial(img)
        f2 = self.freq(freq)
        return self.head(torch.cat([f1, f2], 1))


def load_model(model_path: str, device: str = "cpu") -> HybridDeepfakeDetector:
    """Charge le modèle depuis le fichier .pth."""
    model = HybridDeepfakeDetector().to(device)
    # weights_only=False requis pour charger le dict complet
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Modèle chargé — {sum(p.numel() for p in model.parameters()):,} paramètres")
    return model