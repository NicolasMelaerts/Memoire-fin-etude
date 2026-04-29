"""
run_exp4.py — Expérience 4 : Robustesse face au biais de flèche

Protocole :
  - Train  : 90 % des positifs ont une flèche, 10 % des négatifs ont une flèche
             → la flèche est un fort prédicteur de la classe POSITIVE
  - Test   : Distribution INVERSÉE — 10 % des positifs ont une flèche,
             90 % des négatifs ont une flèche
             → la flèche prédit maintenant la classe NÉGATIVE

Objectif : évaluer si le modèle Normal exploite la flèche comme raccourci
           et si la supervision GradCAM le rend plus robuste à ce biais.

Modèles comparés :
  - Normal   : CrossEntropy pure (peut apprendre le raccourci "flèche → positif")
  - GradCAM  : Guidé pour regarder le cercle/triangle (λ = 0.1)

Usage :
    python run_exp4.py [--clean] [--force]

Options :
  --clean        Supprime Résultats/exp4/ pour repartir à zéro
  --force        Force le réentraînement même si déjà présent

Résultats :
    Résultats/exp4/exp4.html (double-cliquer pour ouvrir)
"""
import os
import sys
import json
import shutil
import argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Ajouter le dossier courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from utils import set_seed, generate_gradcam_examples, plot_comparison
from shared.config import DEVICE, EPOCHS, SEED, ANNOTATIONS, IMAGES_DIR, HEATMAPS_DIR, BATCH_SIZE
from shared.dataset import ShapeDataset
from shared.model import compute_gradcam_numpy

from shared.trainer import Trainer
from shared.strategies import NormalStrategy, GradCAMStrategy


def compute_confusion_matrix(model, test_loader, device, df_ref=None):
    """Calcule la matrice de confusion et les métriques par sous-groupe (flèche/sans flèche).

    Returns:
        dict: {
            'matrix': [[TN, FP], [FN, TP]],
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float,
            'arrow_stats': {  # si df_ref fourni
                'with_arrow': {'correct': int, 'total': int, 'acc': float},
                'without_arrow': {'correct': int, 'total': int, 'acc': float},
            }
        }
    """
    model.eval()
    all_preds, all_labels, all_img_names = [], [], []

    with torch.no_grad():
        for images, labels, heatmaps, img_names in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_img_names.extend(img_names)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tn = int(((all_labels == 1) & (all_preds == 1)).sum())
    fp = int(((all_labels == 1) & (all_preds == 0)).sum())
    fn = int(((all_labels == 0) & (all_preds == 1)).sum())
    tp = int(((all_labels == 0) & (all_preds == 0)).sum())

    accuracy = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        'matrix': [[tn, fp], [fn, tp]],
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
    }

    # Statistiques par sous-groupe flèche
    if df_ref is not None:
        arrow_correct, arrow_total = 0, 0
        no_arrow_correct, no_arrow_total = 0, 0

        for i, img_name in enumerate(all_img_names):
            row = df_ref[df_ref['filename'] == img_name]
            if len(row) == 0:
                continue
            has_arrow = bool(row['has_arrow'].values[0])
            correct = (all_preds[i] == all_labels[i])
            if has_arrow:
                arrow_total += 1
                arrow_correct += int(correct)
            else:
                no_arrow_total += 1
                no_arrow_correct += int(correct)

        result['arrow_stats'] = {
            'with_arrow': {
                'correct': arrow_correct, 'total': arrow_total,
                'acc': round(arrow_correct / max(arrow_total, 1), 4),
            },
            'without_arrow': {
                'correct': no_arrow_correct, 'total': no_arrow_total,
                'acc': round(no_arrow_correct / max(no_arrow_total, 1), 4),
            },
        }

    return result

# Support custom results dir pour exécution parallèle (BIG_EXPERIENCE)
RESULTS_DIR = os.path.join(
    os.environ.get('CUSTOM_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Résultats')),
    'exp4_bis'
)

# Palette : bleu pour Normal, orange pour GradCAM
PALETTE = ['#2196F3', '#FF9800']


def get_color_and_style(name, idx):
    color = PALETTE[idx % len(PALETTE)]
    style = '--' if 'GradCAM' in name else '-'
    return color, style


# ===========================================================================
# Dataset biaisé — Distribution shift entre train et test
# ===========================================================================

def get_biased_dataloaders(seed=SEED):
    """
    Crée des DataLoaders avec un biais de flèche asymétrique entre train et test.

    Répartition mathématiquement équilibrée :
      Train (750 images) :
        - Positifs : 338 avec flèche  +  37 sans flèche  = 375  (90.1 % flèche)
        - Négatifs :  37 avec flèche  + 338 sans flèche  = 375  ( 9.9 % flèche)

      Test (250 images) :
        - Positifs :  12 avec flèche  + 113 sans flèche  = 125  ( 9.6 % flèche)
        - Négatifs : 113 avec flèche  +  12 sans flèche  = 125  (90.4 % flèche)

    Le modèle voit en train : flèche → positif
    Le modèle est évalué en test sur : flèche → négatif  (inversion totale)
    """
    df = pd.read_csv(ANNOTATIONS)

    # Indices par groupe (classe × flèche)
    pos_arrow    = df[(df['class'] == 'positive') & (df['has_arrow'] == True)].index.tolist()
    pos_no_arrow = df[(df['class'] == 'positive') & (df['has_arrow'] == False)].index.tolist()
    neg_arrow    = df[(df['class'] == 'negative') & (df['has_arrow'] == True)].index.tolist()
    neg_no_arrow = df[(df['class'] == 'negative') & (df['has_arrow'] == False)].index.tolist()

    # Mélange reproductible
    rng = np.random.RandomState(seed)
    for group in [pos_arrow, pos_no_arrow, neg_arrow, neg_no_arrow]:
        rng.shuffle(group)

    # ---- Allocation train / test ----
    # Positifs train : 90 % flèche  →  338 arrow + 37 no-arrow = 375
    n_pos_arrow_tr    = 338
    n_pos_no_arrow_tr = 37
    # Négatifs train : 10 % flèche  →   37 arrow + 338 no-arrow = 375
    n_neg_arrow_tr    = 37
    n_neg_no_arrow_tr = 338

    train_indices = (
        pos_arrow[:n_pos_arrow_tr]       +
        pos_no_arrow[:n_pos_no_arrow_tr] +
        neg_arrow[:n_neg_arrow_tr]       +
        neg_no_arrow[:n_neg_no_arrow_tr]
    )
    test_indices = (
        pos_arrow[n_pos_arrow_tr:]       +
        pos_no_arrow[n_pos_no_arrow_tr:] +
        neg_arrow[n_neg_arrow_tr:]       +
        neg_no_arrow[n_neg_no_arrow_tr:]
    )

    # Vérification santé
    assert len(set(train_indices) & set(test_indices)) == 0, "Fuite train/test détectée !"

    # Datasets (train avec augmentation, test sans)
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_full_ds = ShapeDataset(ANNOTATIONS, IMAGES_DIR, HEATMAPS_DIR, transform=train_transform)
    test_full_ds  = ShapeDataset(ANNOTATIONS, IMAGES_DIR, HEATMAPS_DIR, transform=test_transform)

    train_ds = Subset(train_full_ds, train_indices)
    test_ds  = Subset(test_full_ds,  test_indices)

    dl_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, generator=dl_gen)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # Statistiques de biais pour la visualisation HTML
    bias_stats = _compute_bias_stats(df, train_indices, test_indices)

    return train_loader, test_loader, test_ds, bias_stats


def get_radical_biased_dataloaders(seed=SEED):
    """
    Crée des DataLoaders avec un biais RADICAL de flèche (100%/0%).

    Répartition :
      Train :
        - Positifs : UNIQUEMENT ceux avec flèche (350 images)
        - Négatifs : UNIQUEMENT ceux sans flèche  (350 images)
        → 100 % des positifs ont une flèche, 0 % des négatifs

      Test (distribution INVERSÉE) :
        - Positifs : UNIQUEMENT ceux sans flèche  (150 images)
        - Négatifs : UNIQUEMENT ceux avec flèche   (150 images)
        → 0 % des positifs ont une flèche, 100 % des négatifs

    Le biais est maximal : en train, flèche = positif, en test c'est l'exact inverse.
    """
    df = pd.read_csv(ANNOTATIONS)

    pos_arrow    = df[(df['class'] == 'positive') & (df['has_arrow'] == True)].index.tolist()
    pos_no_arrow = df[(df['class'] == 'positive') & (df['has_arrow'] == False)].index.tolist()
    neg_arrow    = df[(df['class'] == 'negative') & (df['has_arrow'] == True)].index.tolist()
    neg_no_arrow = df[(df['class'] == 'negative') & (df['has_arrow'] == False)].index.tolist()

    rng = np.random.RandomState(seed)
    for group in [pos_arrow, pos_no_arrow, neg_arrow, neg_no_arrow]:
        rng.shuffle(group)

    # Train : positifs=arrow, négatifs=no_arrow
    train_indices = pos_arrow + neg_no_arrow
    # Test : positifs=no_arrow, négatifs=arrow
    test_indices = pos_no_arrow + neg_arrow

    assert len(set(train_indices) & set(test_indices)) == 0, "Fuite train/test détectée !"

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_full_ds = ShapeDataset(ANNOTATIONS, IMAGES_DIR, HEATMAPS_DIR, transform=train_transform)
    test_full_ds  = ShapeDataset(ANNOTATIONS, IMAGES_DIR, HEATMAPS_DIR, transform=test_transform)

    train_ds = Subset(train_full_ds, train_indices)
    test_ds  = Subset(test_full_ds,  test_indices)

    dl_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, generator=dl_gen)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    bias_stats = _compute_bias_stats(df, train_indices, test_indices)
    return train_loader, test_loader, test_ds, bias_stats


def _compute_bias_stats(df, train_indices, test_indices):
    """Calcule les statistiques de distribution de flèches dans chaque split."""
    def _stats(indices, split_name):
        sub = df.iloc[indices]
        pos = sub[sub['class'] == 'positive']
        neg = sub[sub['class'] == 'negative']

        pos_arrow_n    = int((pos['has_arrow'] == True).sum())
        pos_no_arrow_n = int((pos['has_arrow'] == False).sum())
        neg_arrow_n    = int((neg['has_arrow'] == True).sum())
        neg_no_arrow_n = int((neg['has_arrow'] == False).sum())

        return {
            'split': split_name,
            'n_total': len(indices),
            'n_positive': len(pos),
            'n_negative': len(neg),
            'pos_arrow':    pos_arrow_n,
            'pos_no_arrow': pos_no_arrow_n,
            'neg_arrow':    neg_arrow_n,
            'neg_no_arrow': neg_no_arrow_n,
            'pos_arrow_pct':    round(100 * pos_arrow_n / max(len(pos), 1), 1),
            'neg_arrow_pct':    round(100 * neg_arrow_n / max(len(neg), 1), 1),
        }

    return {
        'train': _stats(train_indices, 'Train'),
        'test':  _stats(test_indices,  'Test'),
    }


# ===========================================================================
# Sauvegarde data.js (format exp4 — incluant bias_stats)
# ===========================================================================

def save_data_js_exp4(histories, durations, examples, bias_stats, results_dir, epochs,
                      confusion_matrices=None,
                      radical_histories=None, radical_durations=None,
                      radical_bias_stats=None, radical_confusion_matrices=None,
                      radical_examples=None):
    """Écrit data.js avec les métriques, exemples GradCAM et statistiques de biais."""
    epochs_js = list(range(1, epochs + 1))
    chart_data = {'epochs': epochs_js, 'datasets': {}}

    for i, (name, hist) in enumerate(histories.items()):
        color, _ = get_color_and_style(name, i)
        chart_data['datasets'][name] = {
            'test_loss':  hist.get('test_loss', []),
            'test_acc':   hist.get('test_acc', []),
            'train_loss': hist['train_loss'],
            'train_ce':   hist.get('train_ce', hist['train_loss']),
            'train_acc':  hist['train_acc'],
            'color':      color,
        }

    data = {
        'histories':  histories,
        'durations':  durations,
        'chart_data': chart_data,
        'bias_stats': bias_stats,
        'confusion_matrices': confusion_matrices or {},
        'epochs':     epochs,
    }

    # Données du scénario radical (100%/0%)
    radical_data = None
    if radical_histories:
        radical_chart_data = {'epochs': epochs_js, 'datasets': {}}
        for i, (name, hist) in enumerate(radical_histories.items()):
            color, _ = get_color_and_style(name, i)
            radical_chart_data['datasets'][name] = {
                'test_loss':  hist.get('test_loss', []),
                'test_acc':   hist.get('test_acc', []),
                'train_loss': hist['train_loss'],
                'train_ce':   hist.get('train_ce', hist['train_loss']),
                'train_acc':  hist['train_acc'],
                'color':      color,
            }
        radical_data = {
            'histories':  radical_histories,
            'durations':  radical_durations or {},
            'chart_data': radical_chart_data,
            'bias_stats': radical_bias_stats,
            'confusion_matrices': radical_confusion_matrices or {},
            'epochs':     epochs,
        }

    js_path = os.path.join(results_dir, 'data.js')
    with open(js_path, 'w', encoding='utf-8') as f:
        f.write('window.EXP4_DATA = ')
        json.dump(data, f, indent=2)
        f.write(';\n')
        f.write('window.globalExamples = ')
        json.dump(examples, f, indent=2)
        f.write(';\n')
        if radical_data:
            f.write('window.EXP4_RADICAL = ')
            json.dump(radical_data, f, indent=2)
            f.write(';\n')
            f.write('window.radicalExamples = ')
            json.dump(radical_examples or [], f, indent=2)
            f.write(';\n')

    print(f"  → Données JS  : {js_path}")


# ===========================================================================
# Main
# ===========================================================================

def run_scenario(scenario_name, get_loaders_fn, strategies, df_ref,
                 saved_data, args):
    """Entraîne les modèles pour un scénario de biais donné.

    Returns:
        (histories, durations, bias_stats, confusion_matrices, models_out, test_ds)
    """
    histories  = saved_data.get('histories', {})
    durations  = saved_data.get('durations', {})
    bias_stats = saved_data.get('bias_stats', None)
    confusion_matrices = saved_data.get('confusion_matrices', {})
    models_out = {}
    test_ds    = None

    for name, strategy in strategies:
        if name in histories and not args.force:
            print(f"  [⏭]  {name} : déjà entraîné (--force pour réentraîner)")
            continue

        print(f"▶ [{scenario_name}] Entraînement {name}...")
        set_seed(SEED)
        train_loader, test_loader, test_ds, bias_stats = get_loaders_fn(seed=SEED)

        if bias_stats:
            tr = bias_stats['train']
            te = bias_stats['test']
            print(f"  [i] Train — positifs : {tr['pos_arrow_pct']}% flèche "
                  f"| négatifs : {tr['neg_arrow_pct']}% flèche")
            print(f"  [i] Test  — positifs : {te['pos_arrow_pct']}% flèche "
                  f"| négatifs : {te['neg_arrow_pct']}% flèche")

        trainer = Trainer(strategy, verbose=True)
        h, m, d = trainer.run(train_loader, test_loader)

        histories[name]  = h
        models_out[name] = m
        durations[name]  = d

        # Matrice de confusion
        print(f"  → Calcul de la matrice de confusion...")
        cm = compute_confusion_matrix(m, test_loader, DEVICE, df_ref)
        confusion_matrices[name] = cm
        print(f"  ✓ {d:.1f}s — test_acc={h['test_acc'][-1]:.1f}%  "
              f"F1={cm['f1']:.3f}  "
              f"Acc flèche={cm['arrow_stats']['with_arrow']['acc']:.1%}  "
              f"Acc sans flèche={cm['arrow_stats']['without_arrow']['acc']:.1%}\n")

    # Recharger test_ds si on a utilisé le cache
    if test_ds is None:
        _, _, test_ds, bias_stats = get_loaders_fn(seed=SEED)

    return histories, durations, bias_stats, confusion_matrices, models_out, test_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expérience 4 : Robustesse face au biais de flèche")
    parser.add_argument('--clean', action='store_true', help="Supprime les anciens résultats")
    parser.add_argument('--force', action='store_true', help="Force le réentraînement")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXPÉRIENCE 4 : Robustesse face au biais de flèche")
    print("=" * 70)
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {EPOCHS}")
    print(f"  Scénario 1 : biais 90/10 (standard)")
    print(f"  Scénario 2 : biais 100/0 (radical)")
    if args.clean:
        print("  [!] Option --clean : suppression des anciens résultats")
    print()

    if args.clean and os.path.exists(RESULTS_DIR):
        for item in os.listdir(RESULTS_DIR):
            item_path = os.path.join(RESULTS_DIR, item)
            if item.endswith('.html'):
                continue
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    df_ref = pd.read_csv(ANNOTATIONS)

    strategies_to_run = [
        ('Normal',          NormalStrategy()),
        ('GradCAM (λ=0.1)', GradCAMStrategy(lambda_gc=0.1)),
    ]

    # ===== SCÉNARIO 1 : Biais standard (90/10) =====
    print("\n" + "=" * 70)
    print("  SCÉNARIO 1 : Biais 90/10 (standard)")
    print("=" * 70)

    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    saved_std = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                saved_std = json.load(f)
            print(f"  [i] {len(saved_std.get('histories', {}))} modèles chargés depuis metrics.json")
        except json.JSONDecodeError:
            print("  [!] metrics.json corrompu, ignoré")

    (histories, durations, bias_stats, confusion_matrices,
     models_out, test_ds) = run_scenario(
        "90/10", get_biased_dataloaders, strategies_to_run, df_ref,
        saved_std, args
    )

    # Sauvegarder scénario standard
    with open(metrics_path, 'w') as f:
        json.dump({'histories': histories, 'durations': durations,
                   'bias_stats': bias_stats,
                   'confusion_matrices': confusion_matrices}, f, indent=2)

    # ===== SCÉNARIO 2 : Biais radical (100/0) =====
    print("\n" + "=" * 70)
    print("  SCÉNARIO 2 : Biais 100/0 (radical)")
    print("=" * 70)

    radical_metrics_path = os.path.join(RESULTS_DIR, 'metrics_radical.json')
    saved_rad = {}
    if os.path.exists(radical_metrics_path):
        try:
            with open(radical_metrics_path, 'r') as f:
                saved_rad = json.load(f)
            print(f"  [i] {len(saved_rad.get('histories', {}))} modèles chargés depuis metrics_radical.json")
        except json.JSONDecodeError:
            print("  [!] metrics_radical.json corrompu, ignoré")

    # Recréer les stratégies (les objets strategy ne sont pas réutilisables)
    radical_strategies = [
        ('Normal',          NormalStrategy()),
        ('GradCAM (λ=0.1)', GradCAMStrategy(lambda_gc=0.1)),
    ]

    (radical_histories, radical_durations, radical_bias_stats,
     radical_confusion_matrices, radical_models_out,
     radical_test_ds) = run_scenario(
        "100/0", get_radical_biased_dataloaders, radical_strategies, df_ref,
        saved_rad, args
    )

    # Sauvegarder scénario radical
    with open(radical_metrics_path, 'w') as f:
        json.dump({'histories': radical_histories, 'durations': radical_durations,
                   'bias_stats': radical_bias_stats,
                   'confusion_matrices': radical_confusion_matrices}, f, indent=2)

    # -----------------------------------------------------------------------
    # Visuels
    # -----------------------------------------------------------------------
    print("\n▶ Génération des visuels...")

    metrics_config = [
        ((0, 0), 'test_loss',  'Loss',         'Test Loss'),
        ((0, 1), 'train_ce',   'Loss',         'Train Loss (CE pure)'),
        ((1, 0), 'test_acc',   'Accuracy (%)', 'Test Accuracy (%)'),
        ((1, 1), 'train_acc',  'Accuracy (%)', 'Train Accuracy (%)'),
    ]

    # Courbes scénario standard
    plot_comparison(
        histories,
        os.path.join(RESULTS_DIR, 'comparison_curves.png'),
        epochs=EPOCHS,
        title="Expérience 4 : Normal vs GradCAM — Biais 90/10",
        get_color_fn=get_color_and_style,
        metrics=metrics_config
    )

    # Courbes scénario radical
    plot_comparison(
        radical_histories,
        os.path.join(RESULTS_DIR, 'comparison_curves_radical.png'),
        epochs=EPOCHS,
        title="Expérience 4 : Normal vs GradCAM — Biais radical 100/0",
        get_color_fn=get_color_and_style,
        metrics=metrics_config
    )

    # Exemples GradCAM — scénario standard
    # Mix positifs + négatifs : on shuffle TOUS les indices du test set
    rng_std = np.random.RandomState(SEED)
    all_idx_std = np.arange(len(test_ds))
    rng_std.shuffle(all_idx_std)
    examples = generate_gradcam_examples(
        models_out, test_ds, RESULTS_DIR, SEED,
        compute_gradcam_numpy, DEVICE, n_samples=100,
        sample_indices=all_idx_std.tolist(),
    )
    for ex in examples:
        img_name = ex['img_name']
        row = df_ref[df_ref['filename'] == img_name]
        ex['has_arrow'] = bool(row['has_arrow'].values[0]) if len(row) > 0 else False

    # Exemples GradCAM — scénario radical (même logique)
    rng_rad = np.random.RandomState(SEED + 1000)
    all_idx_rad = np.arange(len(radical_test_ds))
    rng_rad.shuffle(all_idx_rad)
    radical_examples = generate_gradcam_examples(
        radical_models_out, radical_test_ds, RESULTS_DIR, SEED + 1000,
        compute_gradcam_numpy, DEVICE, n_samples=100,
        subdir='gradcam_comparison_radical',
        sample_indices=all_idx_rad.tolist(),
    )
    for ex in radical_examples:
        img_name = ex['img_name']
        row = df_ref[df_ref['filename'] == img_name]
        ex['has_arrow'] = bool(row['has_arrow'].values[0]) if len(row) > 0 else False

    save_data_js_exp4(
        histories, durations, examples, bias_stats, RESULTS_DIR, EPOCHS,
        confusion_matrices=confusion_matrices,
        radical_histories=radical_histories,
        radical_durations=radical_durations,
        radical_bias_stats=radical_bias_stats,
        radical_confusion_matrices=radical_confusion_matrices,
        radical_examples=radical_examples,
    )

    # -----------------------------------------------------------------------
    # Résultats finaux
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  RÉSULTATS FINAUX")
    print("=" * 70)

    print("\n  --- Scénario 90/10 (standard) ---")
    for name, h in histories.items():
        print(f"  {name:25s}  train_acc={h['train_acc'][-1]:.1f}%  "
              f"test_acc={h['test_acc'][-1]:.1f}%  "
              f"best_test={max(h['test_acc']):.1f}%  {durations[name]:.0f}s")

    print("\n  --- Scénario 100/0 (radical) ---")
    for name, h in radical_histories.items():
        print(f"  {name:25s}  train_acc={h['train_acc'][-1]:.1f}%  "
              f"test_acc={h['test_acc'][-1]:.1f}%  "
              f"best_test={max(h['test_acc']):.1f}%  {radical_durations[name]:.0f}s")

    if bias_stats:
        print("\n  Distribution flèches (90/10) :")
        tr = bias_stats['train']
        te = bias_stats['test']
        print(f"    Train  — pos: {tr['pos_arrow_pct']}% flèche | neg: {tr['neg_arrow_pct']}% flèche")
        print(f"    Test   — pos: {te['pos_arrow_pct']}% flèche | neg: {te['neg_arrow_pct']}% flèche")

    if radical_bias_stats:
        print("\n  Distribution flèches (100/0) :")
        tr = radical_bias_stats['train']
        te = radical_bias_stats['test']
        print(f"    Train  — pos: {tr['pos_arrow_pct']}% flèche | neg: {tr['neg_arrow_pct']}% flèche")
        print(f"    Test   — pos: {te['pos_arrow_pct']}% flèche | neg: {te['neg_arrow_pct']}% flèche")

    print()
    print("  Pour afficher les résultats :")
    print(f"    Double-cliquer sur : Résultats/exp4/exp4.html")
