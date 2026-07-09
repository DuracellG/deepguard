import torch
import torch.nn as nn
import timm

class HybridDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = timm.create_model(
            'efficientnet_b0', pretrained=False, num_classes=0)
        self.freq = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(1344,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,2))

    def forward(self, img, freq):
        return self.head(torch.cat([self.spatial(img), self.freq(freq)], 1))

def load_model(path, device="cpu"):
    model = HybridDeepfakeDetector().to(device)
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # Checkpoint contenant des objets non-tenseurs (ex. état d'optimiseur picklé).
        # Acceptable uniquement parce que le fichier est produit par nous.
        ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    # Libérer la mémoire des gradients — non nécessaires en inférence
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[OK] {sum(p.numel() for p in model.parameters()):,} parametres")
    return model
