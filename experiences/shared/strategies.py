"""
strategies.py — Stratégies d'entraînement pour la comparaison.

Chaque stratégie définit comment calculer la loss totale pendant l'entraînement.
Toutes héritent de BaseStrategy et implémentent compute_loss().

Usage:
    strategy = NormalStrategy()
    total_loss, metrics = strategy.compute_loss(model, images, labels, heatmaps)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Base Strategy
# ===========================================================================

class BaseStrategy:
    """Classe de base pour toutes les stratégies d'entraînement."""

    def __init__(self, name):
        self.name = name
        self.ce_criterion = nn.CrossEntropyLoss()

    def compute_loss(self, model, images, labels, heatmaps):
        """
        Calcule la loss totale pour un batch.

        Args:
            model: SimpleCNN model
            images: tensor [B, 1, H, W]
            labels: tensor [B] (class labels)
            heatmaps: tensor [B, 1, H, W] (supervision heatmaps)

        Returns:
            total_loss: scalar tensor (pour backward)
            metrics: dict avec les métriques à logger {'ce_loss': float, ...}
        """
        raise NotImplementedError("Chaque stratégie doit implémenter compute_loss()")

    def get_tracking_metrics(self):
        """Retourne la liste des métriques supplémentaires à tracker."""
        return []


# ===========================================================================
# Strategy 1: Normal (Baseline)
# ===========================================================================

class NormalStrategy(BaseStrategy):
    """Stratégie baseline : CrossEntropy pure."""

    def __init__(self):
        super().__init__("Normal")

    def compute_loss(self, model, images, labels, heatmaps):
        logits = model(images)
        ce_loss = self.ce_criterion(logits, labels)

        metrics = {
            'ce_loss': ce_loss.item(),
            'logits': logits.detach()
        }

        return ce_loss, metrics


# ===========================================================================
# Strategy 2: Double Backpropagation
# ===========================================================================

class DoubleBackpropStrategy(BaseStrategy):
    """
    Stratégie Double Backpropagation : pénalise la norme des gradients.

    Loss = CE + λ * ||∇_x CE||²

    Encourage le modèle à avoir des gradients d'entrée plus lisses.
    """

    def __init__(self, lambda_bp=100.0):
        super().__init__(f"Double BP (λ={lambda_bp})")
        self.lambda_bp = lambda_bp

    def compute_loss(self, model, images, labels, heatmaps):
        # Activer le gradient sur l'entrée
        images_with_grad = images.clone().requires_grad_(True)

        # Forward
        logits = model(images_with_grad)
        ce_loss = self.ce_criterion(logits, labels)

        # Calculer les gradients par rapport à l'entrée
        grads = torch.autograd.grad(
            ce_loss, images_with_grad,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Pénalité sur la norme L2 des gradients
        grad_penalty = (grads ** 2).sum() / images.size(0)

        # Loss totale
        total_loss = ce_loss + self.lambda_bp * grad_penalty

        metrics = {
            'ce_loss': ce_loss.item(),
            'grad_penalty': grad_penalty.item(),
            'logits': logits.detach()
        }

        return total_loss, metrics

    def get_tracking_metrics(self):
        return ['grad_penalty']


# ===========================================================================
# Strategy 3: GradCAM-Guided
# ===========================================================================

def compute_gradcam_differentiable(model, image_batch, target_classes):
    """
    Calcule une carte GradCAM différentiable pour un batch d'images.

    Utilise la décomposition features / classifier du modèle pour que le
    graphe de calcul reste intact et permette la rétropropagation de la
    loss de localisation.

    Args:
        model         : SimpleCNN (mode train)
        image_batch   : tensor [B, 1, H, W]
        target_classes: tensor [B] (labels vrais ou prédits)

    Returns:
        cam_batch : tensor [B, 1, 128, 128], valeurs dans [0, 1]
                    (différentiable par rapport aux paramètres du modèle)
    """
    B = image_batch.size(0)

    # 1. Forward jusqu'à conv3 (avant pool)  — avec create_graph pour double dérivée
    feat  = model.forward_features(image_batch)          # [B, 64, 32, 32]
    feat.retain_grad()                                    # on veut ∂loss/∂feat

    # 2. Suite du forward à partir des feature maps
    logits = model.forward_from_features(feat)            # [B, 2]

    # 3. Score de la classe cible (scalaire par batch item)
    scores = logits[torch.arange(B), target_classes]     # [B]
    score_sum = scores.sum()

    # 4. Gradient ∂score/∂feat  avec create_graph pour pouvoir dériver à nouveau
    grads = torch.autograd.grad(
        score_sum, feat,
        create_graph=True,
        retain_graph=True,
    )[0]  # [B, 64, 32, 32]

    # 5. Global Average Pooling des gradients → poids par canal
    weights = grads.mean(dim=(2, 3), keepdim=True)        # [B, 64, 1, 1]

    # 6. Combinaison pondérée des activations
    cam = (weights * feat).sum(dim=1, keepdim=True)       # [B, 1, 32, 32]
    cam = F.relu(cam)

    # 7. Upsample vers 128x128
    cam = F.interpolate(cam, size=(128, 128), mode='bilinear', align_corners=False)  # [B, 1, 128, 128]

    # 8. Normalisation par item du batch dans [0, 1]
    B_, C_, H_, W_ = cam.shape
    cam_flat   = cam.view(B_, -1)
    cam_min    = cam_flat.min(dim=1, keepdim=True).values
    cam_max    = cam_flat.max(dim=1, keepdim=True).values
    cam_norm   = (cam_flat - cam_min) / (cam_max - cam_min + 1e-8)
    cam_norm   = cam_norm.view(B_, C_, H_, W_)

    return cam_norm, logits


class GradCAMStrategy(BaseStrategy):
    """
    Stratégie GradCAM-Guided : pénalise le désalignement entre GradCAM et heatmap.

    Loss = CE + λ * L_localisation
    où L_localisation = 1 - cosine_similarity(GradCAM, heatmap)

    Force le modèle à activer ses features sur la bonne zone spatiale.
    """

    def __init__(self, lambda_gc=0.5):
        super().__init__(f"GradCAM (λ={lambda_gc})")
        self.lambda_gc = lambda_gc

    def compute_loss(self, model, images, labels, heatmaps):
        # GradCAM différentiable basé sur les labels vrais
        cam_batch, logits = compute_gradcam_differentiable(model, images, labels)

        # Cross-entropy
        ce_loss = self.ce_criterion(logits, labels)

        # Loss de localisation (1 - cosine similarity)
        loc_loss = self._localisation_loss(cam_batch, heatmaps)

        # Loss totale
        total_loss = ce_loss + self.lambda_gc * loc_loss

        metrics = {
            'ce_loss': ce_loss.item(),
            'loc_loss': loc_loss.item(),
            'logits': logits.detach()
        }

        return total_loss, metrics

    def _localisation_loss(self, cam_batch, supervision_batch):
        """
        Loss de localisation : 1 - similarité cosinus entre les cartes aplaties.

        Args:
            cam_batch: [B, 1, 128, 128] — cartes GradCAM différentiables
            supervision_batch: [B, 1, 128, 128] — heatmaps de supervision

        Returns:
            scalar tensor (différentiable)
        """
        B = cam_batch.size(0)
        cam_flat = cam_batch.view(B, -1)
        sup_flat = supervision_batch.view(B, -1).to(cam_flat.device)

        # Cosine similarity par image, puis moyenne sur le batch
        cos_sim = F.cosine_similarity(cam_flat, sup_flat, dim=1)  # [B]
        loc_loss = 1.0 - cos_sim.mean()
        return loc_loss

    def get_tracking_metrics(self):
        return ['loc_loss']


# ===========================================================================
# Strategy 4: GAIN (Gradient-Adjusted Input Network)
# ===========================================================================

class GAINStrategy(BaseStrategy):
    """
    Stratégie GAIN : masquage d'attention.

    Loss = CE(original) + λ * CE(masked)

    où masked = image * (1 - attention_mask)
    et attention_mask est dérivé des gradients d'entrée.

    Force le modèle à ne pas pouvoir classer correctement si on cache la zone importante.
    """

    def __init__(self, lambda_gain=1.0):
        super().__init__(f"GAIN (λ={lambda_gain})")
        self.lambda_gain = lambda_gain

    def compute_loss(self, model, images, labels, heatmaps):
        # (1) Forward normal avec gradients sur l'entrée
        images_with_grad = images.clone().requires_grad_(True)
        logits = model(images_with_grad)
        ce_loss = self.ce_criterion(logits, labels)

        # (2) Calculer les gradients par rapport à l'entrée
        grads = torch.autograd.grad(
            ce_loss, images_with_grad,
            create_graph=True,
            retain_graph=True,
        )[0]

        # (3) Construire un masque d'attention depuis les gradients
        attention_map = grads.abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Normaliser dans [0, 1]
        B = attention_map.size(0)
        att_flat = attention_map.view(B, -1)
        att_min = att_flat.min(dim=1, keepdim=True).values
        att_max = att_flat.max(dim=1, keepdim=True).values
        att_norm = (att_flat - att_min) / (att_max - att_min + 1e-8)
        attention_map = att_norm.view_as(attention_map)

        # (4) Masquer l'image (cacher les zones importantes)
        masked_images = images * (1.0 - attention_map)

        # (5) Forward sur l'image masquée
        logits_masked = model(masked_images)

        # (6) Loss sur l'image masquée (on veut que le modèle échoue sur l'image masquée)
        # On maximise l'entropie → le modèle devrait être incertain
        # Ici on utilise la CE avec le label opposé (flip 0↔1)
        labels_flipped = 1 - labels
        masked_loss = self.ce_criterion(logits_masked, labels_flipped)

        # Loss totale
        total_loss = ce_loss + self.lambda_gain * masked_loss

        metrics = {
            'ce_loss': ce_loss.item(),
            'masked_loss': masked_loss.item(),
            'logits': logits.detach()
        }

        return total_loss, metrics

    def get_tracking_metrics(self):
        return ['masked_loss']


# ===========================================================================
# Strategy 5: Right for the Right Reasons (RRR)
# ===========================================================================

class RRRStrategy(BaseStrategy):
    """
    Stratégie Right for the Right Reasons (Ross et al. 2017).

    Loss = CE + λ * ||A ⊙ ∇_x log p(y|x)||²

    où :
    - A = masque des régions interdites (1 = interdit, 0 = autorisé)
    - A = 1 - heatmap (on interdit tout sauf la zone pertinente)
    - ∇_x log p(y|x) = gradients de la log-probabilité par rapport à l'entrée

    Pénalise les gradients dans les zones interdites pour forcer le modèle
    à se concentrer sur les bonnes régions.

    Référence : Ross et al., "Right for the Right Reasons: Training Differentiable
                Models by Constraining their Explanations", IJCAI 2017
    """

    def __init__(self, lambda_rrr=10.0):
        super().__init__(f"RRR (λ={lambda_rrr})")
        self.lambda_rrr = lambda_rrr

    def compute_loss(self, model, images, labels, heatmaps):
        # (1) Forward avec gradients activés sur l'entrée
        images_with_grad = images.clone().requires_grad_(True)
        logits = model(images_with_grad)
        ce_loss = self.ce_criterion(logits, labels)

        # (2) Calculer log p(y|x) pour la classe vraie
        log_probs = F.log_softmax(logits, dim=1)  # [B, num_classes]

        # Récupérer les log-probabilités des classes vraies
        B = images.size(0)
        target_log_probs = log_probs[torch.arange(B), labels]  # [B]

        # (3) Calculer ∇_x log p(y|x)
        grads = torch.autograd.grad(
            target_log_probs.sum(),
            images_with_grad,
            create_graph=True,
            retain_graph=True,
        )[0]  # [B, 1, H, W]

        # (4) Construire le masque des régions interdites
        # A = 1 - heatmap (1 = interdit, 0 = autorisé)
        forbidden_mask = 1.0 - heatmaps  # [B, 1, H, W]

        # (5) Pénalité RRR : ||A ⊙ ∇_x||² (L2 au carré des gradients dans zones interdites)
        masked_grads = forbidden_mask * grads
        rrr_penalty = (masked_grads ** 2).sum() / B

        # (6) Loss totale
        total_loss = ce_loss + self.lambda_rrr * rrr_penalty

        metrics = {
            'ce_loss': ce_loss.item(),
            'rrr_penalty': rrr_penalty.item(),
            'logits': logits.detach()
        }

        return total_loss, metrics

    def get_tracking_metrics(self):
        return ['rrr_penalty']
