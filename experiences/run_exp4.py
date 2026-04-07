"""
run_exp4.py — Expérience 4 : Robustesse face au biais de flèche (Distribution Shift)

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

from utils import set_seed, generate_gradcam_examples, plot_comparison
from shared.config import DEVICE, EPOCHS, SEED, ANNOTATIONS, IMAGES_DIR, HEATMAPS_DIR, BATCH_SIZE
from shared.dataset import ShapeDataset
from shared.model import compute_gradcam_numpy

from shared.trainer import Trainer
from shared.strategies import NormalStrategy, GradCAMStrategy

# Support custom results dir pour exécution parallèle (BIG_EXPERIENCE)
RESULTS_DIR = os.path.join(
    os.environ.get('CUSTOM_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Résultats')),
    'exp4'
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

def save_data_js_exp4(histories, durations, examples, bias_stats, results_dir, epochs):
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

    print(f"  → Données JS  : {js_path}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expérience 4 : Robustesse face au biais de flèche")
    parser.add_argument('--clean', action='store_true', help="Supprime les anciens résultats")
    parser.add_argument('--force', action='store_true', help="Force le réentraînement")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXPÉRIENCE 4 : Robustesse face au biais de flèche (Distribution Shift)")
    print("=" * 70)
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {EPOCHS}")
    print(f"  Train  : 90 % des positifs ont une flèche   (biais fort)")
    print(f"  Test   : 10 % des positifs ont une flèche   (distribution inversée)")
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

    # Charger les métriques sauvegardées si elles existent
    histories = {}
    durations = {}
    bias_stats = None
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                saved = json.load(f)
            histories  = saved.get('histories', {})
            durations  = saved.get('durations', {})
            bias_stats = saved.get('bias_stats', None)
            print(f"  [i] {len(histories)} modèles chargés depuis metrics.json")
        except json.JSONDecodeError:
            print("  [!] metrics.json corrompu, ignoré")

    models_out = {}
    test_ds    = None

    # -----------------------------------------------------------------------
    # Modèles à comparer
    # -----------------------------------------------------------------------
    strategies_to_run = [
        ('Normal',          NormalStrategy()),
        ('GradCAM (λ=0.1)', GradCAMStrategy(lambda_gc=0.1)),
    ]

    for name, strategy in strategies_to_run:
        if name in histories and not args.force:
            print(f"  [⏭]  {name} : déjà entraîné (--force pour réentraîner)")
            continue

        print(f"▶ Entraînement {name}...")
        set_seed(SEED)
        train_loader, test_loader, test_ds, bias_stats = get_biased_dataloaders(seed=SEED)

        # Afficher les statistiques de biais au premier modèle
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
        print(f"  ✓ {d:.1f}s — test_acc={h['test_acc'][-1]:.1f}%\n")

        # Sauvegarder immédiatement
        with open(metrics_path, 'w') as f:
            json.dump({'histories': histories, 'durations': durations,
                       'bias_stats': bias_stats}, f, indent=2)

    # Recharger test_ds si on a utilisé le cache
    if test_ds is None:
        _, _, test_ds, bias_stats = get_biased_dataloaders(seed=SEED)

    # -----------------------------------------------------------------------
    # Visuels
    # -----------------------------------------------------------------------
    print("▶ Génération des visuels...")

    metrics_config = [
        ((0, 0), 'test_loss',  'Loss',         'Test Loss'),
        ((0, 1), 'train_ce',   'Loss',         'Train Loss (CE pure)'),
        ((1, 0), 'test_acc',   'Accuracy (%)', 'Test Accuracy (%)'),
        ((1, 1), 'train_acc',  'Accuracy (%)', 'Train Accuracy (%)'),
    ]
    plot_comparison(
        histories,
        os.path.join(RESULTS_DIR, 'comparison_curves.png'),
        epochs=EPOCHS,
        title="Expérience 4 : Normal vs GradCAM — Biais de flèche",
        get_color_fn=get_color_and_style,
        metrics=metrics_config
    )

    examples = generate_gradcam_examples(
        models_out, test_ds, RESULTS_DIR, SEED,
        compute_gradcam_numpy, DEVICE, n_samples=100,
    )

    # Enrichir chaque example avec le flag has_arrow
    df_ref = pd.read_csv(ANNOTATIONS)
    for ex in examples:
        img_name = ex['img_name']
        row = df_ref[df_ref['filename'] == img_name]
        ex['has_arrow'] = bool(row['has_arrow'].values[0]) if len(row) > 0 else False

    save_data_js_exp4(histories, durations, examples, bias_stats, RESULTS_DIR, EPOCHS)

    # Sauvegarder metrics.json final
    with open(metrics_path, 'w') as f:
        json.dump({'histories': histories, 'durations': durations,
                   'bias_stats': bias_stats}, f, indent=2)
    print(f"  → Métriques : {metrics_path}")

    print()
    print("=" * 70)
    print("  RÉSULTATS FINAUX")
    print("=" * 70)
    for name, h in histories.items():
        print(f"  {name:25s}  train_acc={h['train_acc'][-1]:.1f}%  "
              f"test_acc={h['test_acc'][-1]:.1f}%  "
              f"best_test={max(h['test_acc']):.1f}%  {durations[name]:.0f}s")

    if bias_stats:
        print()
        print("  Distribution flèches :")
        tr = bias_stats['train']
        te = bias_stats['test']
        print(f"    Train  — pos: {tr['pos_arrow_pct']}% flèche | neg: {tr['neg_arrow_pct']}% flèche")
        print(f"    Test   — pos: {te['pos_arrow_pct']}% flèche | neg: {te['neg_arrow_pct']}% flèche")

    print()
    print("  Pour afficher les résultats :")
    print(f"    Double-cliquer sur : Résultats/exp4/exp4.html")
