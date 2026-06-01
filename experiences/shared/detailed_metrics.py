"""
detailed_metrics.py - Calcul des métriques détaillées (accuracy, precision,
recall, F1, IoU explication) à partir d'un modèle entraîné et d'un test_loader.

Utilisé par run_exp1 (pour Normal + Guided GradCAM) et run_exp3.
"""
import numpy as np
import torch

from shared.model import compute_gradcam_numpy


def compute_metrics_manual(labels, preds):
    """Accuracy / Precision / Recall / F1 + matrice de confusion (TN, FP, FN, TP)."""
    labels = np.array(labels).flatten()
    preds = np.array(preds).flatten()

    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tp = int(np.sum((labels == 1) & (preds == 1)))

    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': [[tn, fp], [fn, tp]],
    }


def compute_iou(pred_map, gt_map, threshold=0.3):
    """IoU entre deux heatmaps après binarisation."""
    pred_binary = (pred_map >= threshold).astype(float)
    gt_binary = (gt_map >= threshold).astype(float)
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(np.maximum(pred_binary, gt_binary))
    if union == 0:
        return 0.0
    return intersection / union


def compute_detailed_metrics(model, test_loader, device, compute_iou_flag=False):
    """
    Calcule accuracy/precision/recall/F1/(IoU) sur le test_loader.

    L'IoU est calculé sur les images positives (label == 0 = Inside) avec un
    seuil de 0.3 (GradCAM peut être diffus).
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_ious = []

    with torch.no_grad():
        for images, labels, heatmaps, _ in test_loader:
            images = images.to(device)
            labels_np = labels.numpy()

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels_np.tolist())

            if compute_iou_flag:
                for i in range(len(images)):
                    if labels_np[i] == 0:  # positif = classe 0 (Inside)
                        img_tensor = images[i]
                        with torch.enable_grad():
                            gradcam, _, _ = compute_gradcam_numpy(model, img_tensor, device=device)
                        gt_heatmap = heatmaps[i].squeeze().numpy()
                        iou = compute_iou(gradcam, gt_heatmap, threshold=0.3)
                        all_ious.append(iou)

    cls = compute_metrics_manual(all_labels, all_preds)

    metrics = {
        'accuracy': cls['accuracy'],
        'precision': cls['precision'],
        'recall': cls['recall'],
        'f1': cls['f1'],
        'confusion_matrix': cls['confusion_matrix'],
        'predictions': all_preds,
        'labels': all_labels,
    }

    if compute_iou_flag and all_ious:
        metrics['iou'] = float(np.mean(all_ious))
        metrics['iou_std'] = float(np.std(all_ious))
    else:
        metrics['iou'] = None
        metrics['iou_std'] = None

    return metrics
