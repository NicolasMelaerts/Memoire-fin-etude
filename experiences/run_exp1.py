"""
run_exp1.py - Expérience 1 : Comparaison des stratégies d'entraînement

Compare 5 approches :
  - Normal (baseline)
  - Double Backpropagation
  - Guided GradCAM
  - GAIN
  - RRR - Right for the Right Reasons

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
import json
import shutil
import argparse

import init_env  # noqa: F401  (side effects: sys.path + DATASET_PATH)

from utils import set_seed, generate_gradcam_examples, save_data_js, plot_comparison
from shared.config import DEVICE, EPOCHS, SEED, STRATEGY_COLORS
from shared.dataset import get_dataloaders
from shared.model import compute_gradcam_numpy
from shared.detailed_metrics import compute_detailed_metrics

from shared.trainer import Trainer
from shared.strategies import (
    NormalStrategy,
    DoubleBackpropStrategy,
    GradCAMStrategy,
    GAINStrategy,
    RRRStrategy
)

# Modèles dont les detailed_metrics seront calculées et exposées pour exp3
# (mêmes données, même seed, même stratégie -> exp3 peut les recycler).
SHARED_WITH_EXP3 = {'Normal', 'Guided GradCAM (λ=0.1)'}

# Prise en charge d'un dossier de résultats personnalisé pour l'exécution parallèle (run_multidataset)
RESULTS_DIR = os.path.join(
    os.environ.get('CUSTOM_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Résultats')),
    'exp1'
)

def get_color_and_style(name, idx):
    """Retourne couleur (issue de shared.config.STRATEGY_COLORS) et style de ligne."""
    color = STRATEGY_COLORS.get(name, '#888')
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
    detailed_metrics = {}
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                saved = json.load(f)
            histories = saved.get('histories', {})
            durations = saved.get('durations', {})
            detailed_metrics = saved.get('detailed_metrics', {})
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
        ('Double BP (λ=200.0)', DoubleBackpropStrategy(lambda_bp=200.0)),
        ('Guided GradCAM (λ=0.1)', GradCAMStrategy(lambda_gc=0.1)),
        ('GAIN (λ=0.1)',   GAINStrategy(lambda_gain=0.1)),
        ('RRR (λ=0.1)',    RRRStrategy(lambda_rrr=0.1)),
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
        print(f"  ✓ {d:.1f}s - test_acc={h['test_acc'][-1]:.1f}%")

        # Pour les modèles partagés avec exp3, calcule les métriques détaillées
        # (precision/recall/F1/IoU) afin qu'exp3 puisse les recycler.
        if name in SHARED_WITH_EXP3:
            print(f"  → Calcul des métriques détaillées (pour recyclage par exp3)...")
            _, test_loader_eval, _ = get_dataloaders(seed=SEED)
            detailed_metrics[name] = compute_detailed_metrics(
                m, test_loader_eval, DEVICE, compute_iou_flag=True
            )
            dm = detailed_metrics[name]
            iou_str = f"IoU={dm['iou']:.3f}" if dm['iou'] is not None else "IoU=N/A"
            print(f"    Acc={dm['accuracy']:.1%}  F1={dm['f1']:.3f}  {iou_str}")
        print()

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
        json.dump({
            'histories': histories,
            'durations': durations,
            'detailed_metrics': detailed_metrics,
        }, f, indent=2)
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
