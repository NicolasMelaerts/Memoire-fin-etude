"""
run_exp0.py - Expérience 0 : Sélection du λ optimal par stratégie via grid search.

Pour chaque stratégie régularisée (Double BP, GradCAM-Guided, GAIN, RRR), on entraîne
le modèle avec une grille de valeurs de λ et on évalue chacun de ces modèles
sur un set de validation séparé. Le critère de sélection est la
Cross-Entropy sur la validation, calculée par le trainer en mode eval (sans
pénalité), pour ne pas biaiser le choix de λ par la grandeur de la pénalité.

Le test set reste intouché : il ne sert qu'aux expériences finales (exp1-4).

Split :
    Test = 25% du dataset complet (figé, identique à exp1)
    Pool train = 75% du dataset complet → re-divisé en :
        Train = 75% du pool   (= 56.25% du total)
        Val   = 25% du pool   (= 18.75% du total)

Usage :
    python run_exp0.py [--clean] [--dataset NOM]

Options :
  --clean          Supprime le contenu de Résultats/exp0/ pour repartir à zéro,
                   en préservant les fichiers .html.
  --dataset NOM    Nom du dossier dataset (défaut : dataset_seed13).
                   Cherche d'abord dans experiences/, puis dans dataset_creator/.

Résultats :
    Résultats/exp0/exp0.html (double-cliquer pour ouvrir)
"""
import os
import sys
import json
import shutil
import argparse

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Ajouter le dossier courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Résolution du dataset AVANT les imports shared (config.py lit DATASET_PATH
# au moment de l'import).
# -----------------------------------------------------------------------------
_exp_dir = os.path.dirname(os.path.abspath(__file__))
_root    = os.path.dirname(_exp_dir)
if '--dataset' in sys.argv:
    _idx = sys.argv.index('--dataset')
    if _idx + 1 < len(sys.argv):
        _ds_name = sys.argv[_idx + 1]
        _candidate = os.path.join(_exp_dir, _ds_name)
        if not os.path.isdir(_candidate):
            _candidate = os.path.join(_root, 'dataset_creator', _ds_name)
        os.environ['DATASET_PATH'] = _candidate
elif 'DATASET_PATH' not in os.environ:
    # Défaut : dataset_seed13 dans experiences/
    os.environ['DATASET_PATH'] = os.path.join(_exp_dir, 'dataset_seed13')

from utils import set_seed
from shared.config import (
    DEVICE, EPOCHS, SEED, BATCH_SIZE, TRAIN_RATIO,
    ANNOTATIONS, IMAGES_DIR, HEATMAPS_DIR,
)
from shared.dataset import ShapeDataset
from shared.trainer import Trainer
from shared.strategies import (
    DoubleBackpropStrategy,
    GradCAMStrategy,
    GAINStrategy,
    RRRStrategy,
)


# =============================================================================
# Configuration : grilles de λ par stratégie
# (modifiables ici pour ajouter/supprimer des valeurs)
# =============================================================================

LAMBDA_GRIDS = {
    'Double BP': [50, 75, 100, 150, 175, 200, 250, 1000],
    'GradCAM-Guided':   [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5],
    'GAIN':      [0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
    'RRR':       [0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
}

# Constructeurs de stratégie associés
STRATEGY_FACTORIES = {
    'Double BP': lambda lam: DoubleBackpropStrategy(lambda_bp=lam),
    'GradCAM-Guided': lambda lam: GradCAMStrategy(lambda_gc=lam),
    'GAIN':      lambda lam: GAINStrategy(lambda_gain=lam),
    'RRR':       lambda lam: RRRStrategy(lambda_rrr=lam),
}

# Couleurs cohérentes avec exp1.html
STRATEGY_COLORS = {
    'Double BP': '#FF9800',
    'GradCAM-Guided':   '#9C27B0',
    'GAIN':      '#4CAF50',
    'RRR':       '#F44336',
}

# Ratio train/val *au sein du pool d'entraînement*. Le test reste intouché.
TRAIN_VAL_RATIO = 0.75


# =============================================================================
# Loaders avec split train / val / test
# =============================================================================

def get_dataloaders_train_val_test(seed):
    """
    Retourne ``(train_loader, val_loader, test_loader)``.

    - Le test set est carved en premier (25 % du dataset complet, indices figés
      par ``seed``), exactement comme dans exp1 → il reste intouché.
    - Le pool d'entraînement (75 % restant) est ensuite re-divisé en
      ``TRAIN_VAL_RATIO`` (train) / ``1 - TRAIN_VAL_RATIO`` (val).
    - Les transforms d'augmentation ne s'appliquent qu'au train ; val et test
      utilisent le même ``ToTensor()`` simple.
    """
    generator = torch.Generator().manual_seed(seed)

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Deux instances : l'une avec augmentation (train), l'autre sans (val/test).
    train_full_ds = ShapeDataset(
        annotations_file=ANNOTATIONS,
        img_dir=IMAGES_DIR,
        heatmaps_dir=HEATMAPS_DIR,
        transform=train_transform,
    )
    eval_full_ds = ShapeDataset(
        annotations_file=ANNOTATIONS,
        img_dir=IMAGES_DIR,
        heatmaps_dir=HEATMAPS_DIR,
        transform=eval_transform,
    )

    n_total = len(train_full_ds)
    indices = torch.randperm(n_total, generator=generator).tolist()

    n_train_pool = int(TRAIN_RATIO * n_total)
    pool_indices = indices[:n_train_pool]
    test_indices = indices[n_train_pool:]

    n_train = int(TRAIN_VAL_RATIO * len(pool_indices))
    train_indices = pool_indices[:n_train]
    val_indices   = pool_indices[n_train:]

    train_ds = Subset(train_full_ds, train_indices)
    val_ds   = Subset(eval_full_ds, val_indices)
    test_ds  = Subset(eval_full_ds, test_indices)

    dataloader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False,
        generator=dataloader_generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# Grid search par stratégie
# =============================================================================

def run_grid_search(strategy_name, lambdas, train_loader, val_loader):
    """Entraîne ``strategy_name`` avec chaque λ et retourne les métriques de val."""
    factory = STRATEGY_FACTORIES[strategy_name]
    results = []

    for lam in lambdas:
        # Reset du seed avant chaque entraînement → la seule variable est λ.
        set_seed(SEED)
        strategy = factory(lam)

        # `Trainer` évalue avec une CE pure (cf. trainer.py:181) : c'est
        # exactement ce qu'on veut comme critère de sélection.
        trainer = Trainer(strategy, verbose=False)
        history, _, duration = trainer.run(train_loader, val_loader)

        # Le trainer enregistre la perte de validation sous la clé `test_loss`
        # (le trainer ne distingue pas test/val ; ici on lui passe val_loader).
        val_loss_curve = history['test_loss']
        val_acc_curve  = history['test_acc']

        # Meilleur epoch sur la val
        best_epoch_idx = min(range(len(val_loss_curve)),
                             key=lambda i: val_loss_curve[i])

        results.append({
            'lambda':         lam,
            'val_loss_final': val_loss_curve[-1],
            'val_acc_final':  val_acc_curve[-1],
            'val_loss_best':  val_loss_curve[best_epoch_idx],
            'val_acc_best':   val_acc_curve[best_epoch_idx],
            'best_epoch':     best_epoch_idx + 1,
            'val_loss_curve': val_loss_curve,
            'val_acc_curve':  val_acc_curve,
            'duration':       duration,
        })

        print(f"    λ={lam:>10.4g}  "
              f"val_loss={val_loss_curve[-1]:.4f}  "
              f"val_acc={val_acc_curve[-1]:.1f}%  "
              f"({duration:.1f}s)")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expérience 0 : Grid search λ par stratégie"
    )
    parser.add_argument('--clean', action='store_true',
                        help="Supprime les anciens résultats (sauf .html)")
    parser.add_argument('--dataset', default='dataset_seed13',
                        help="Nom du dossier dataset (défaut: dataset_seed13)")
    args = parser.parse_args()

    RESULTS_DIR = os.path.join(
        os.environ.get('CUSTOM_RESULTS_DIR',
                       os.path.join(_exp_dir, 'Résultats')),
        'exp0'
    )

    print("=" * 70)
    print("  EXPÉRIENCE 0 : Grid search λ par stratégie")
    print("=" * 70)
    print(f"  Device         : {DEVICE}")
    print(f"  Dataset        : {args.dataset}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Train/Val/Test : "
          f"{TRAIN_RATIO * TRAIN_VAL_RATIO:.0%} / "
          f"{TRAIN_RATIO * (1 - TRAIN_VAL_RATIO):.0%} / "
          f"{1 - TRAIN_RATIO:.0%}")
    if args.clean:
        print("  [!] Option --clean : suppression des fichiers non-HTML")
    print()

    # Création + clean éventuel (en préservant les .html)
    if args.clean and os.path.exists(RESULTS_DIR):
        for item in os.listdir(RESULTS_DIR):
            if item.endswith('.html'):
                continue  # Préserver les fichiers HTML
            item_path = os.path.join(RESULTS_DIR, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    # Charger les anciens résultats si possible (reprise par stratégie)
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    all_results = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                all_results = json.load(f)
            print(f"  [i] Reprise : {len(all_results)} stratégie(s) déjà calculée(s)")
        except json.JSONDecodeError:
            print("  [!] metrics.json corrompu, on repart à zéro")

    # Loaders (une seule fois pour toute la grid search)
    print("▶ Préparation des DataLoaders (split train/val/test)...")
    train_loader, val_loader, test_loader = get_dataloaders_train_val_test(SEED)
    n_train = sum(b[0].size(0) for b in train_loader)
    n_val   = sum(b[0].size(0) for b in val_loader)
    n_test  = sum(b[0].size(0) for b in test_loader)
    print(f"  train={n_train}   val={n_val}   test={n_test}\n")

    # Grid search par stratégie
    for strategy_name, lambdas in LAMBDA_GRIDS.items():
        print(f"▶ {strategy_name} ({len(lambdas)} valeurs de λ)")
        results = run_grid_search(strategy_name, lambdas,
                                  train_loader, val_loader)
        all_results[strategy_name] = results

        # Sauvegarde après chaque stratégie pour limiter la perte en cas de crash
        with open(metrics_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print()

    # Génération du data.js pour exp0.html
    print("▶ Génération du data.js...")
    chart_data = {}
    for strategy_name, results in all_results.items():
        chart_data[strategy_name] = {
            'lambdas':        [r['lambda']         for r in results],
            'val_loss_final': [r['val_loss_final'] for r in results],
            'val_acc_final':  [r['val_acc_final']  for r in results],
            'val_loss_best':  [r['val_loss_best']  for r in results],
            'val_acc_best':   [r['val_acc_best']   for r in results],
            'best_epoch':     [r['best_epoch']     for r in results],
            'duration':       [r['duration']       for r in results],
            'color':          STRATEGY_COLORS.get(strategy_name, '#888'),
        }

    js_path = os.path.join(RESULTS_DIR, 'data.js')
    with open(js_path, 'w', encoding='utf-8') as f:
        f.write('window.GRIDSEARCH_DATA = ')
        json.dump(chart_data, f, indent=2)
        f.write(';\n')
    print(f"  → Données : {js_path}")

    # Récap console
    print()
    print("=" * 70)
    print("  λ OPTIMAL PAR STRATÉGIE  (selon CE finale sur validation)")
    print("=" * 70)
    for strategy_name, results in all_results.items():
        best = min(results, key=lambda r: r['val_loss_final'])
        print(f"  {strategy_name:12s}  "
              f"λ*={best['lambda']:>10.4g}   "
              f"val_loss={best['val_loss_final']:.4f}   "
              f"val_acc={best['val_acc_final']:.1f}%")
    print()
    print("  Pour visualiser :")
    print(f"    Double-cliquer sur : Résultats/exp0/exp0.html")
