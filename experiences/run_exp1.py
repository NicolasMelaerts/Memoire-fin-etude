"""
run_exp1.py — Expérience 1 : Comparaison des stratégies d'entraînement

Compare 5 approches :
  - Normal (baseline)
  - Double Backpropagation (λ = 50)
  - GradCAM-Guided (λ = 0.1)
  - GAIN (λ = 0.1)
  - RRR - Right for the Right Reasons (λ = 1.0)

Usage :
    python run_exp1.py [--clean] [--dataset NOM]

Options :
  --clean          Supprime Résultats/exp1/ pour repartir à zéro
  --dataset NOM    Nom du dossier dataset dans dataset_creator/
                   (défaut : generated_dataset)
                   Ex: --dataset dataset_reel

Résultats :
    Résultats/exp1/exp1.html (double-cliquer pour ouvrir)
"""
import os
import sys
import json
import shutil
import argparse

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

from utils import set_seed, generate_gradcam_examples, save_data_js, plot_comparison
from shared.config import DEVICE, EPOCHS, SEED
from shared.dataset import get_dataloaders
from shared.model import compute_gradcam_numpy

from shared.trainer import Trainer
from shared.strategies import (
    NormalStrategy,
    DoubleBackpropStrategy,
    GradCAMStrategy,
    GAINStrategy,
    RRRStrategy
)

# Support custom results dir for parallel execution (BIG_EXPERIENCE)
RESULTS_DIR = os.path.join(
    os.environ.get('CUSTOM_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Résultats')),
    'exp1'
)

# Palette de couleurs
PALETTE = [
    '#2196F3', '#FF9800', '#FF5722', '#F44336', '#E91E63',
    '#9C27B0', '#673AB7', '#3F51B5', '#4CAF50', '#8BC34A',
]


def get_color_and_style(name, idx):
    """Retourne couleur et style de ligne pour un modèle."""
    color = PALETTE[idx % len(PALETTE)]
    if 'Normal' in name:
        style = '-'
    elif 'Double BP' in name:
        style = '--'
    elif 'GAIN' in name:
        style = ':'
    else:
        style = '-.'
    return color, style


def extract_case_type(test_ds, idx):
    """Extrait le case_type depuis le dataset."""
    orig_idx = test_ds.indices[idx]
    return test_ds.dataset.df.iloc[orig_idx]['case']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expérience 1 : Comparaison des stratégies")
    parser.add_argument('--clean', action='store_true',
                        help="Supprime les anciens résultats")
    parser.add_argument('--dataset', default='generated_dataset',
                        help="Nom du dossier dataset dans dataset_creator/ (défaut: generated_dataset)")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXPÉRIENCE 1 : Comparaison des stratégies d'entraînement")
    print("=" * 70)
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {EPOCHS}")
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
    histories = {}
    durations = {}
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                saved = json.load(f)
            histories = saved.get('histories', {})
            durations = saved.get('durations', {})
            print(f"  [i] {len(histories)} modèles chargés depuis metrics.json")
        except json.JSONDecodeError:
            print("  [!] metrics.json corrompu, ignoré")

    models_out = {}
    print(f"\n  [i] Les DataLoaders seront recréés avant chaque modèle pour garantir la synchronisation RNG.\n")

    # =======================================================================
    # Configuration des modèles à entraîner
    # =======================================================================

    strategies_to_run = [
        ('Normal',          NormalStrategy()),
        ('Double BP (λ=50.0)', DoubleBackpropStrategy(lambda_bp=50.0)),
        ('GradCAM (λ=0.1)', GradCAMStrategy(lambda_gc=0.1)),
        ('GAIN (λ=0.1)',    GAINStrategy(lambda_gain=0.1)),
        ('RRR (λ=1.0)',     RRRStrategy(lambda_rrr=1.0)),
    ]

    # =======================================================================
    # Entraînement
    # =======================================================================

    for name, strategy in strategies_to_run:

        print(f"▶ Entraînement {name}...")
        set_seed(SEED)
        train_loader, test_loader, test_ds = get_dataloaders(seed=SEED)

        trainer = Trainer(strategy, verbose=True)
        h, m, d = trainer.run(train_loader, test_loader)

        histories[name] = h
        models_out[name] = m
        durations[name] = d
        print(f"  ✓ {d:.1f}s — test_acc={h['test_acc'][-1]:.1f}%\n")

    # ---- Génération des visuels ----
    print("▶ Génération des visuels...")

    metrics_config = [
        ((0, 0), 'test_loss',  'Loss',         'Test Loss (Cross Entropy)'),
        ((0, 1), 'train_ce',   'Loss',         'Train Loss (Pure Cross Entropy)'),
        ((1, 0), 'test_acc',   'Accuracy (%)', 'Test Accuracy (%)'),
        ((1, 1), 'train_acc',  'Accuracy (%)', 'Train Accuracy (%)'),
    ]
    plot_comparison(
        histories,
        os.path.join(RESULTS_DIR, 'comparison_curves.png'),
        epochs=EPOCHS,
        title="Expérience 1 : Comparaison des stratégies d'entraînement",
        get_color_fn=get_color_and_style,
        metrics=metrics_config
    )

    examples = generate_gradcam_examples(
        models_out, test_ds, RESULTS_DIR, SEED,
        compute_gradcam_numpy, DEVICE, n_samples=200,
        extract_case_fn=extract_case_type
    )

    save_data_js(histories, durations, examples, RESULTS_DIR, EPOCHS,
                 get_color_fn=get_color_and_style)

    # Sauvegarder metrics.json
    with open(metrics_path, 'w') as f:
        json.dump({'histories': histories, 'durations': durations}, f, indent=2)
    print(f"  → Métriques : {metrics_path}")

    print()
    print("=" * 70)
    print("  RÉSULTATS FINAUX")
    print("=" * 70)
    for name, h in histories.items():
        print(f"  {name:25s} test_acc={h['test_acc'][-1]:.1f}%  "
              f"best={max(h['test_acc']):.1f}%  {durations[name]:.0f}s")

    print()
    print("  Pour afficher les résultats :")
    print(f"    Double-cliquer sur : Résultats/exp1/exp1.html")
