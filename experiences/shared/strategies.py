"""
strategies.py - Stratégies d'entraînement pour la comparaison.

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

    def __init__(self, lambda_bp=50.0):
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

    # 1. Forward jusqu'à conv3 (avant pool)  - avec create_graph pour double dérivée
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
            cam_batch: [B, 1, 128, 128] - cartes GradCAM différentiables
            supervision_batch: [B, 1, 128, 128] - heatmaps de supervision

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
# Strategy 4: GAIN (Guided Attention Inference Network)
# Li et al., "Tell Me Where to Look: Guided Attention Inference Network", CVPR 2018
# ===========================================================================

class GAINStrategy(BaseStrategy):
    """
    Stratégie GAIN (Guided Attention Inference Network) - Li et al., CVPR 2018.

    Implémentation fidèle à la variante self-guidée (Section 3.1 de l'article).

    Pipeline :
      1. Carte d'attention A^c via Grad-CAM sur la dernière couche conv (éq. 1–2).
      2. Masque soft différentiable T(A^c) via sigmoid paramétrique (éq. 3–4).
      3. Image masquée I^{*c} = I - T(A^c) ⊙ I  (éq. 3).
      4. Loss totale : L_self = L_cl + α * L_am  (éq. 6)
         avec L_am = (1/n) Σ_c  s^c(I^{*c})     (éq. 5)
         où s^c(·) est le score softmax de la classe c.

    Paramètres :
        lambda_gain : poids de L_am (noté α dans l'article, = 1 dans leurs expériences)
        omega   : échelle du sigmoid T (noté ω dans l'article)
        sigma   : seuil du sigmoid T (noté σ dans l'article, matrice scalaire ici)
    """

    def __init__(self, lambda_gain=1.0, omega=10.0, sigma=0.5):
        super().__init__(f"GAIN (λ={lambda_gain})")
        self.lambda_gain = lambda_gain  # α dans l'article (éq. 6)
        self.omega = omega              # échelle du sigmoid (éq. 4)
        self.sigma = sigma              # seuil du sigmoid  (éq. 4)

    def compute_loss(self, model, images, labels, heatmaps):
        B = images.size(0)

        # ------------------------------------------------------------------
        # Étape 1 & 2 - Carte d'attention A^c par Grad-CAM (éq. 1–2)
        # compute_gradcam_differentiable retourne A^c normalisée dans [0,1]
        # et les logits du forward classique (stream S_cl).
        # ------------------------------------------------------------------
        A_c, logits = compute_gradcam_differentiable(model, images, labels)
        # A_c   : [B, 1, 128, 128], dans [0, 1]
        # logits: [B, 2]

        # ------------------------------------------------------------------
        # Étape 3 - Perte de classification L_cl  (éq. 6)
        # ------------------------------------------------------------------
        L_cl = self.ce_criterion(logits, labels)

        # ------------------------------------------------------------------
        # Étape 4 - Masque soft T(A^c) différentiable (éq. 4)
        #
        #   T(A^c) = sigmoid(-ω * (A^c - σ))
        #          = 1 / (1 + exp(-ω * (A^c - σ)))
        #
        # Quand A^c_{i,j} >> σ  →  T ≈ 1  (zone très activée → masquée)
        # Quand A^c_{i,j} << σ  →  T ≈ 0  (zone peu activée → conservée)
        # ------------------------------------------------------------------
        T_Ac = torch.sigmoid(self.omega * (A_c - self.sigma))  # [B, 1, 128, 128]

        # ------------------------------------------------------------------
        # Étape 5 - Image masquée I^{*c} = I - T(A^c) ⊙ I  (éq. 3)
        # Les régions fortement activées sont soustraites (mises à ~0).
        # ------------------------------------------------------------------
        I_masked = images - T_Ac * images  # [B, 1, 128, 128]

        # ------------------------------------------------------------------
        # Étape 6 - Score de la classe cible sur l'image masquée (éq. 5)
        #
        #   L_am = (1/n) Σ_c  s^c(I^{*c})
        #
        # s^c(·) est le score softmax pour la classe c.
        # On minimise ce score → le modèle ne doit plus reconnaître la classe
        # sur l'image masquée.
        # ------------------------------------------------------------------
        logits_masked = model(I_masked)                              # [B, 2]
        probs_masked  = F.softmax(logits_masked, dim=1)              # [B, 2]
        scores_c      = probs_masked[torch.arange(B), labels]        # [B]
        L_am          = scores_c.mean()                              # scalaire

        # ------------------------------------------------------------------
        # Loss totale L_self = L_cl + α * L_am  (éq. 6)
        # ------------------------------------------------------------------
        total_loss = L_cl + self.lambda_gain * L_am

        metrics = {
            'ce_loss': L_cl.item(),
            'am_loss': L_am.item(),
            'logits':  logits.detach()
        }

        return total_loss, metrics

    def get_tracking_metrics(self):
        return ['am_loss']


# ===========================================================================
# Strategy 5: Right for the Right Reasons (RRR)
# Andrew Slavin Ross et al., "Right for the Right Reasons: Training Differentiable Models by Constraining
# their Explanations", IJCAI 2017
# ===========================================================================

class RRRStrategy(BaseStrategy):
    """
    Stratégie Right for the Right Reasons (Ross et al., IJCAI 2017).

    Loss = CE + λ * ||A ⊙ ∇_x Σ_k log ŷ_k||²

    où :
    - A = masque des régions interdites = 1 - heatmap
      (1 = pixel interdit, 0 = pixel autorisé)
    - ∇_x Σ_k log ŷ_k = gradient de la somme des log-probabilités
      sur toutes les classes, par rapport aux pixels d'entrée

    Pénalise les gradients d'entrée dans les zones non pertinentes,
    forçant le modèle à construire ses décisions uniquement à partir
    des régions indiquées par les heatmaps de supervision.
    """

    def __init__(self, lambda_rrr=1.0):
        super().__init__(f"RRR (λ={lambda_rrr})")
        self.lambda_rrr = lambda_rrr

    def compute_loss(self, model, images, labels, heatmaps):
        # (1) Cloner les images et activer le suivi des gradients par rapport
        #     aux pixels d'entrée. Le clone évite de modifier le tenseur original.
        images_with_grad = images.clone().requires_grad_(True)

        # (2) Propagation avant : obtenir les logits et la cross-entropy.
        logits = model(images_with_grad)
        ce_loss = self.ce_criterion(logits, labels)

        # (3) Calculer les log-probabilités sur toutes les classes via log-softmax.
        #     log_probs[i, k] = log p_θ(classe k | x_i)
        #     shape : [B, K]
        log_probs = F.log_softmax(logits, dim=1)

        B = images.size(0)

        # (4) Calculer le gradient de la somme des log-probabilités sur le batch
        #     ET sur toutes les classes, par rapport aux pixels d'entrée.
        #     C'est la formulation exacte de Ross et al. : Σ_n Σ_k log ŷ_nk.
        #     create_graph=True est indispensable : la pénalité dépend de ces
        #     gradients, qui dépendent eux-mêmes de θ. L'optimisation requiert
        #     donc des dérivées d'ordre deux.
        #     shape résultante : [B, 1, H, W]
        grads = torch.autograd.grad(
            log_probs.sum(),   # scalaire : somme sur batch et classes
            images_with_grad,
            create_graph=True,
            retain_graph=True,
        )[0]

        # (5) Construire le masque des régions interdites.
        #     A = 1 - H : là où la heatmap est élevée (zone pertinente),
        #     le masque est proche de 0 (gradient libre).
        #     Là où la heatmap est faible (fond, distracteurs),
        #     le masque est proche de 1 (gradient pénalisé).
        #     shape : [B, 1, H, W]
        forbidden_mask = 1.0 - heatmaps

        # (6) Appliquer le masque aux gradients (produit de Hadamard),
        #     puis calculer la pénalité RRR : norme L2 au carré des gradients
        #     dans les zones interdites, moyennée sur le batch.
        masked_grads = forbidden_mask * grads
        rrr_penalty = (masked_grads ** 2).sum() / B

        # (7) Combiner cross-entropy et pénalité RRR.
        total_loss = ce_loss + self.lambda_rrr * rrr_penalty

        metrics = {
            'ce_loss': ce_loss.item(),
            'rrr_penalty': rrr_penalty.item(),
            'logits': logits.detach()
        }

        return total_loss, metrics

    def get_tracking_metrics(self):
        return ['rrr_penalty']