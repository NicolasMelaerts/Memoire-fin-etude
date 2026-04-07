"""
model.py — SimpleCNN partagé + utilitaire GradCAM différentiable.

GradCAMExtractor :
  Permet de calculer la carte GradCAM *dans le graphe de calcul* PyTorch,
  ce qui est indispensable pour le GradCAM-guided training où la loss de
  localisation doit être rétropropagée jusqu'aux poids du réseau.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """CNN simple (3 conv + 2 fc) pour classification inside/outside."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 16 * 16, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(128, 2)

    # -----------------------------------------------------------------------
    # Forward classique
    # -----------------------------------------------------------------------
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32 x 64 x 64
        x = self.pool(F.relu(self.conv2(x)))   # 64 x 32 x 32
        x = self.pool(F.relu(self.conv3(x)))   # 64 x 16 x 16
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    # -----------------------------------------------------------------------
    # Forward partiel pour obtenir les feature maps de conv3 avant pooling
    # -----------------------------------------------------------------------
    def forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        feat = F.relu(self.conv3(x))           # [B, 64, 32, 32] avant pool
        return feat

    def forward_from_features(self, feat):
        x = self.pool(feat)                    # [B, 64, 16, 16]
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_fresh_model(device):
    """Retourne un nouveau modèle vierge sur le device indiqué."""
    return SimpleCNN().to(device)


# ---------------------------------------------------------------------------
# GradCAM pour la visualisation post-entraînement (images superposées dans les HTML).
# Contrairement à compute_gradcam_differentiable dans strategies.py, cette version
# détache le graphe de calcul et retourne un tableau NumPy — elle ne peut pas
# être utilisée dans une loss.
# ---------------------------------------------------------------------------

def compute_gradcam_numpy(model, image_tensor, target_class=None, device='cpu'):
    """
    Calcule la carte GradCAM (numpy array, [H, W]) pour une image.
    Utilisé uniquement pour la visualisation, pas pour l'entraînement.

    Args:
        model        : SimpleCNN en mode eval
        image_tensor : tensor [1, H, W] (sans batch dim)
        target_class : int ou None (auto)
        device       : device PyTorch

    Returns:
        cam          : np.ndarray [128, 128], valeurs dans [0, 1]
        pred_class   : int
        logits       : tensor [1, 2]
    """
    import numpy as np
    import cv2

    model.eval()
    gradients  = []
    activations = []

    def _save_grad(m, gi, go): gradients.append(go[0])
    def _save_act(m, inp, out): activations.append(out)

    h1 = model.conv3.register_full_backward_hook(_save_grad)
    h2 = model.conv3.register_forward_hook(_save_act)

    x       = image_tensor.unsqueeze(0).to(device)
    logits  = model(x)

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    model.zero_grad()
    logits[0, target_class].backward()

    grads = gradients[0].detach().cpu().numpy()[0]    # [C, H, W]
    acts  = activations[0].detach().cpu().numpy()[0]  # [C, H, W]

    h1.remove()
    h2.remove()

    weights = grads.mean(axis=(1, 2))                 # [C]
    cam     = (weights[:, None, None] * acts).sum(0)  # [H, W]
    cam     = np.maximum(cam, 0)
    cam     = cv2.resize(cam, (128, 128))
    mn, mx  = cam.min(), cam.max()
    cam     = (cam - mn) / (mx - mn + 1e-8)

    return cam, target_class, logits.detach()
