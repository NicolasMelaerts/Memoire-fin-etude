"""
run_exp3.py - Expérience 3 : Métriques détaillées Normal vs GradCAM

Compare Normal vs GradCAM-Guided (λ=0.1) avec 750 images d'entraînement.
Calcule des métriques détaillées :
  - Accuracy : performance classification
  - F1-score : robustesse
  - Precision : faux positifs
  - Recall : triangles manqués
  - IoU explication : qualité XAI

Usage :
    python run_exp3.py [--clean] [--dataset NOM]

Options :
  --clean          Supprime Résultats/exp3/ pour repartir à zéro
  --dataset NOM    Nom du dossier dataset dans dataset_creator/
                   (défaut : generated_dataset)
                   Ex: --dataset dataset_reel

Résultats :
    Résultats/exp3/index.html (double-cliquer pour ouvrir)
"""
import os
import sys
import json
import shutil
import argparse
import numpy as np
import torch

# Ajouter le dossier courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Résolution du dataset AVANT les imports shared (config.py lit DATASET_PATH à l'import)
_exp_dir = os.path.dirname(os.path.abspath(__file__))
_root    = os.path.dirname(_exp_dir)
if '--dataset' in sys.argv:
    _idx = sys.argv.index('--dataset')
    if _idx + 1 < len(sys.argv):
        _ds_name = sys.argv[_idx + 1]
        # Chercher d'abord dans Expériences/, puis dans dataset_creator/
        _candidate = os.path.join(_exp_dir, _ds_name)
        if not os.path.isdir(_candidate):
            _candidate = os.path.join(_root, 'dataset_creator', _ds_name)
        os.environ['DATASET_PATH'] = _candidate
elif 'DATASET_PATH' not in os.environ:
    os.environ['DATASET_PATH'] = os.path.join(_root, 'dataset_creator', 'generated_dataset')

from utils import set_seed
from shared.config import DEVICE, EPOCHS, SEED
from shared.dataset import get_dataloaders
from shared.model import compute_gradcam_numpy

from shared.trainer import Trainer
from shared.strategies import NormalStrategy, GradCAMStrategy

# Support custom results dir for parallel execution (BIG_EXPERIENCE)
RESULTS_DIR = os.path.join(
    os.environ.get('CUSTOM_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Résultats')),
    'exp3'
)

# Palette de couleurs
PALETTE = ['#2196F3', '#FF9800']


def compute_metrics_manual(labels, preds):
    """
    Calcule accuracy, precision, recall, f1, confusion_matrix manuellement.

    Args:
        labels: liste de vrais labels (0 ou 1)
        preds: liste de prédictions (0 ou 1)

    Returns:
        dict: {'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'}
    """
    labels = np.array(labels).flatten()
    preds = np.array(preds).flatten()

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    tp = np.sum((labels == 1) & (preds == 1))

    # Accuracy
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1-score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
    }


def compute_iou(pred_map, gt_map, threshold=0.5):
    """
    Calcule l'IoU (Intersection over Union) entre deux heatmaps.

    Args:
        pred_map: Heatmap prédite (valeurs [0, 1])
        gt_map: Ground truth heatmap (valeurs [0, 1])
        threshold: Seuil pour binariser les heatmaps

    Returns:
        float: IoU score [0, 1]
    """
    # Binariser
    pred_binary = (pred_map >= threshold).astype(float)
    gt_binary = (gt_map >= threshold).astype(float)

    # Intersection et union
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(np.maximum(pred_binary, gt_binary))

    if union == 0:
        return 0.0

    return intersection / union


def compute_detailed_metrics(model, test_loader, device, compute_iou_flag=False):
    """
    Calcule toutes les métriques détaillées pour un modèle.

    Returns:
        dict: {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'confusion_matrix': [[TN, FP], [FN, TP]],
            'iou': float (si compute_iou_flag=True),
            'predictions': list,
            'labels': list
        }
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_ious = []

    with torch.no_grad():
        for images, labels, heatmaps, _ in test_loader:
            images = images.to(device)
            labels_np = labels.numpy()

            # Prédictions (argmax pour CrossEntropy avec 2 sorties)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels_np.tolist())

            # Calcul IoU pour les images positives si demandé
            if compute_iou_flag:
                for i in range(len(images)):
                    if labels_np[i] == 0:  # Positifs = classe 0 (Inside)
                        # Générer GradCAM (nécessite les gradients)
                        img_tensor = images[i]
                        with torch.enable_grad():
                            gradcam, _, _ = compute_gradcam_numpy(model, img_tensor, device=device)

                        # Ground truth heatmap
                        gt_heatmap = heatmaps[i].squeeze().numpy()

                        # Calculer IoU (seuil plus bas car GradCAM peut être diffus)
                        iou = compute_iou(gradcam, gt_heatmap, threshold=0.3)
                        all_ious.append(iou)

    # Métriques de classification (calcul manuel)
    metrics_dict = compute_metrics_manual(all_labels, all_preds)

    metrics = {
        'accuracy': metrics_dict['accuracy'],
        'precision': metrics_dict['precision'],
        'recall': metrics_dict['recall'],
        'f1': metrics_dict['f1'],
        'confusion_matrix': metrics_dict['confusion_matrix'],
        'predictions': all_preds,
        'labels': all_labels
    }

    if compute_iou_flag and all_ious:
        metrics['iou'] = float(np.mean(all_ious))
        metrics['iou_std'] = float(np.std(all_ious))
    else:
        metrics['iou'] = None
        metrics['iou_std'] = None

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expérience 3 : Métriques détaillées")
    parser.add_argument('--clean', action='store_true',
                        help="Supprime les anciens résultats")
    parser.add_argument('--force', action='store_true',
                        help="Force l'entraînement même si déjà présent")
    parser.add_argument('--dataset', default='generated_dataset',
                        help="Nom du dossier dataset dans dataset_creator/ (défaut: generated_dataset)")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXPÉRIENCE 3 : Métriques détaillées Normal vs GradCAM")
    print("=" * 70)
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {EPOCHS}")
    print(f"  λ_gc=0.1")
    if args.clean:
        print("  [!] Option --clean : suppression des anciens résultats")
    print()

    if args.clean and os.path.exists(RESULTS_DIR):
        # Supprimer uniquement les fichiers non-HTML
        for item in os.listdir(RESULTS_DIR):
            item_path = os.path.join(RESULTS_DIR, item)
            if item.endswith('.html'):
                continue  # Préserver les fichiers HTML
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    # Charger les anciennes métriques si elles existent
    all_metrics = {}
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                all_metrics = json.load(f)
            print(f"  [i] {len(all_metrics)} modèles chargés depuis metrics.json\n")
        except json.JSONDecodeError:
            print("  [!] metrics.json corrompu, ignoré\n")

    # Dataset : 750 train / 250 test
    test_ds = None  # Sera chargé lors du premier entraînement
    print(f"  [i] Les DataLoaders seront recréés avant chaque modèle pour garantir la synchronisation RNG.\n")

    # ===== MODÈLE 1 : NORMAL =====
    model_name = "Normal"
    model_normal = None

    must_train = model_name not in all_metrics or args.force

    if must_train:
        print(f"▶ Entraînement {model_name}...")
        set_seed(SEED)
        train_loader, test_loader, test_ds = get_dataloaders(seed=SEED, train_subset_size=750)

        trainer = Trainer(NormalStrategy(), verbose=True)
        history, model_normal, duration = trainer.run(train_loader, test_loader)

        print(f"  → Calcul des métriques détaillées (avec IoU)...")
        # Recréer le test_loader pour être sûr de l'état
        _, test_loader, _ = get_dataloaders(seed=SEED, train_subset_size=750)
        detailed = compute_detailed_metrics(model_normal, test_loader, DEVICE, compute_iou_flag=True)

        all_metrics[model_name] = {
            'history': history,
            'duration': duration,
            'detailed_metrics': detailed
        }

        # Sauvegarder immédiatement
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        iou_str = f"{detailed['iou']:.3f}±{detailed['iou_std']:.3f}" if detailed['iou'] is not None else "N/A"
        print(f"  ✓ {duration:.1f}s - Accuracy={detailed['accuracy']:.1%} F1={detailed['f1']:.3f} IoU={iou_str}\n")
    else:
        print(f"⏭️  {model_name} : déjà entraîné (utilisez --force pour réentraîner)\n")

    # ===== MODÈLE 2 : GRADCAM λ=0.1 =====
    model_name = "GradCAM λ=0.1"
    model_gc = None

    must_train = model_name not in all_metrics or args.force

    if must_train:
        print(f"▶ Entraînement {model_name}...")
        set_seed(SEED)
        train_loader, test_loader, test_ds = get_dataloaders(seed=SEED, train_subset_size=750)

        trainer = Trainer(GradCAMStrategy(lambda_gc=0.1), verbose=True)
        history, model_gc, duration = trainer.run(train_loader, test_loader)

        print(f"  → Calcul des métriques détaillées (avec IoU)...")
        # Recréer le test_loader
        _, test_loader, _ = get_dataloaders(seed=SEED, train_subset_size=750)
        detailed = compute_detailed_metrics(model_gc, test_loader, DEVICE, compute_iou_flag=True)

        all_metrics[model_name] = {
            'history': history,
            'duration': duration,
            'detailed_metrics': detailed
        }

        # Sauvegarder immédiatement
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        iou_str = f"{detailed['iou']:.3f}±{detailed['iou_std']:.3f}" if detailed['iou'] is not None else "N/A"
        print(f"  ✓ {duration:.1f}s - Accuracy={detailed['accuracy']:.1%} F1={detailed['f1']:.3f} IoU={iou_str}\n")
    else:
        print(f"⏭️  {model_name} : déjà entraîné (utilisez --force pour réentraîner)\n")

    # ===== GÉNÉRATION DES DONNÉES JS =====
    print("▶ Génération des données JS...")

    # Sauvegarder data.js pour la page HTML (Format spécifique Exp3)
    with open(os.path.join(RESULTS_DIR, 'data.js'), 'w', encoding='utf-8') as f:
        f.write('window.RESULTS_DATA = ')
        json.dump(all_metrics, f, indent=2)
        f.write(';\n')
    print(f"  → Données JS  : {os.path.join(RESULTS_DIR, 'data.js')}")
    print(f"  → Métriques : {metrics_path}")

    # ===== AFFICHAGE DES RÉSULTATS =====
    print("\n" + "=" * 70)
    print("  RÉSULTATS FINAUX")
    print("=" * 70)

    for model_name, data in all_metrics.items():
        detailed = data['detailed_metrics']
        duration = data['duration']

        acc = detailed['accuracy']
        prec = detailed['precision']
        rec = detailed['recall']
        f1 = detailed['f1']
        iou = detailed.get('iou')

        iou_str = f"IoU={iou:.3f}" if iou is not None else "IoU=N/A"

        print(f"  {model_name:30s}  Acc={acc:.1%}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  {iou_str}  {duration:.0f}s")

    print(f"\n  Pour afficher les résultats :")
    print(f"    Double-cliquer sur : Résultats/exp3/index.html")
