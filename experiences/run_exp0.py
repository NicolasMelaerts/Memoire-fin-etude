"""
run_exp0.py - Expérience 0 : Sélection du λ optimal par stratégie via grid search,
agrégée sur plusieurs datasets (multi-seed) pour réduire la variance.

Pour chaque stratégie régularisée (Double BP, Guided GradCAM, GAIN, RRR), on
entraîne le modèle avec une grille de valeurs de λ, sur N datasets distincts
(un par seed de génération), et on agrège les résultats par moyenne ± écart-type.
Le critère de sélection est la Cross-Entropy sur la validation, calculée par le
trainer en mode eval (sans pénalité), pour ne pas biaiser le choix de λ par la
grandeur de la pénalité.

EPOCHS est fixé dans shared/config.py (par décision méthodologique) et ne fait
pas l'objet d'une recherche dans ce script.

Le test set reste intouché : il ne sert qu'aux expériences finales (exp1-4).

Split (par dataset) :
    Test = 25% du dataset (figé, identique à exp1)
    Pool train = 75% restant → re-divisé en :
        Train = 75% du pool   (= 56.25% du total)
        Val   = 25% du pool   (= 18.75% du total)

Usage :
    python run_exp0.py [--clean]

Options :
  --clean             Supprime le contenu de Résultats/exp0/ pour repartir à zéro,
                      en préservant les fichiers .html.

Les datasets utilisés sont définis dans `DATASETS_DEFAULT` dans ce fichier.
"""
import os
import json
import shutil
import argparse
import statistics

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from init_env import EXP_DIR, ROOT

from utils import set_seed
from shared.config import (
    DEVICE, EPOCHS, SEED, BATCH_SIZE, TRAIN_RATIO, STRATEGY_COLORS,
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
# =============================================================================

# Chaque grille comporte 6 valeurs en progression géométrique. La plage est
# centrée sur l'ordre de grandeur indiqué par la littérature pour chaque
# pénalité, puis affinée en fonction de tests préliminaires sur les datasets
# afin de rester dans un régime produisant des résultats cohérents. Le
# facteur multiplicatif entre deux valeurs successives est choisi de manière
# à conserver 6 points tout en couvrant cette plage.
LAMBDA_GRIDS = {
    # 6 valeurs de 50 à 1600, facteur 2 entre deux valeurs successives.
    # Plage autour de la centaine, cohérente avec les ordres de grandeur
    # employés par Drucker & LeCun (1992) pour Double Backpropagation.
    'Double BP':      [50, 100, 200, 400, 800, 1600],

    # 6 valeurs de 0.03 à 10, facteur ≈ 3.16 (= √10) entre deux valeurs.
    # Plage autour de l'unité, typique des pondérations utilisées pour
    # une pénalité sur carte d'attention normalisée entre 0 et 1.
    'Guided GradCAM': [0.03, 0.1, 0.3, 1, 3, 10],

    # 6 valeurs de 0.003 à 1, même facteur ≈ 3.16 que Guided GradCAM,
    # plage décalée vers les petites valeurs.
    'GAIN':           [0.003, 0.01, 0.03, 0.1, 0.3, 1],

    # 6 valeurs de 0.01 à 1000, facteur 10 entre deux valeurs successives.
    # Plage large autour de λ=1000 (valeur retenue par Ross et al. 2017
    # sur MLP) pour situer l'ordre de grandeur dans le cas CNN, où la
    # littérature ne fournit pas de valeur de référence stable.
    'RRR':            [0.01, 0.1, 1, 10, 100, 1000],
}

STRATEGY_FACTORIES = {
    'Double BP':      lambda lam: DoubleBackpropStrategy(lambda_bp=lam),
    'Guided GradCAM': lambda lam: GradCAMStrategy(lambda_gc=lam),
    'GAIN':           lambda lam: GAINStrategy(lambda_gain=lam),
    'RRR':            lambda lam: RRRStrategy(lambda_rrr=lam),
}

DATASETS_DEFAULT = ['dataset_seed3', 'dataset_seed11', 'dataset_seed13', 'dataset_seed25', 'dataset_seed33']

TRAIN_VAL_RATIO = 0.75


# =============================================================================
# Loaders
# =============================================================================

def resolve_dataset_dir(name):
    """Cherche le dataset dans experiences/datasets/<name> puis dataset_creator/<name>."""
    candidate = os.path.join(EXP_DIR, 'datasets', name)
    if os.path.isdir(candidate):
        return candidate
    candidate = os.path.join(ROOT, 'dataset_creator', name)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"Dataset introuvable : {name}\n"
        f"  Cherché dans :\n"
        f"    - {os.path.join(EXP_DIR, 'datasets', name)}\n"
        f"    - {os.path.join(ROOT, 'dataset_creator', name)}"
    )


def get_dataloaders_train_val_test(seed, dataset_dir):
    """Retourne ``(train_loader, val_loader, test_loader)`` pour un dataset donné."""
    annotations  = os.path.join(dataset_dir, 'annotations.csv')
    images_dir   = os.path.join(dataset_dir, 'images')
    heatmaps_dir = os.path.join(dataset_dir, 'heatmaps')

    generator = torch.Generator().manual_seed(seed)

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor()])

    train_full_ds = ShapeDataset(
        annotations_file=annotations,
        img_dir=images_dir,
        heatmaps_dir=heatmaps_dir,
        transform=train_transform,
    )
    eval_full_ds = ShapeDataset(
        annotations_file=annotations,
        img_dir=images_dir,
        heatmaps_dir=heatmaps_dir,
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

    dl_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False, generator=dl_gen,
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
# Grid search multi-dataset
# =============================================================================

def safe_std(xs):
    """Écart-type d'échantillon ; renvoie 0 si moins de 2 valeurs."""
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def run_grid_search(strategy_name, lambdas, datasets):
    """Pour chaque λ, entraîne sur chacun des `datasets` et agrège mean/std des
    métriques de validation."""
    factory = STRATEGY_FACTORIES[strategy_name]
    results = []

    for lam in lambdas:
        per_seed = []
        for ds_name in datasets:
            dataset_dir = resolve_dataset_dir(ds_name)
            train_loader, val_loader, _ = get_dataloaders_train_val_test(
                SEED, dataset_dir
            )

            # Init du modèle / shuffle déterministe pour fixer la variance à
            # la seule génération du dataset.
            set_seed(SEED)
            strategy = factory(lam)
            trainer  = Trainer(strategy, verbose=False)
            history, _, duration = trainer.run(train_loader, val_loader)

            val_loss_curve = history['test_loss']
            val_acc_curve  = history['test_acc']
            best_idx = min(range(len(val_loss_curve)),
                           key=lambda i: val_loss_curve[i])

            per_seed.append({
                'dataset':        ds_name,
                'val_loss_final': val_loss_curve[-1],
                'val_acc_final':  val_acc_curve[-1],
                'val_loss_best':  val_loss_curve[best_idx],
                'val_acc_best':   val_acc_curve[best_idx],
                'best_epoch':     best_idx + 1,
                'duration':       duration,
                'val_loss_curve': val_loss_curve,
                'val_acc_curve':  val_acc_curve,
            })

            print(f"      [{ds_name:>16s}] λ={lam:>10.4g}  "
                  f"val_loss={val_loss_curve[-1]:.4f}  "
                  f"val_acc={val_acc_curve[-1]:.1f}%  "
                  f"({duration:.1f}s)")

        loss_finals = [r['val_loss_final'] for r in per_seed]
        acc_finals  = [r['val_acc_final']  for r in per_seed]
        loss_bests  = [r['val_loss_best']  for r in per_seed]
        acc_bests   = [r['val_acc_best']   for r in per_seed]
        best_epochs = [r['best_epoch']     for r in per_seed]
        durations   = [r['duration']       for r in per_seed]

        agg = {
            'lambda':               lam,
            'n_seeds':              len(per_seed),
            'datasets':             list(datasets),
            'val_loss_final_mean':  statistics.mean(loss_finals),
            'val_loss_final_std':   safe_std(loss_finals),
            'val_acc_final_mean':   statistics.mean(acc_finals),
            'val_acc_final_std':    safe_std(acc_finals),
            'val_loss_best_mean':   statistics.mean(loss_bests),
            'val_loss_best_std':    safe_std(loss_bests),
            'val_acc_best_mean':    statistics.mean(acc_bests),
            'val_acc_best_std':     safe_std(acc_bests),
            'best_epoch_mean':      statistics.mean(best_epochs),
            'duration_mean':        statistics.mean(durations),
            'duration_total':       sum(durations),
            'per_seed':             per_seed,
        }
        results.append(agg)

        print(f"    λ={lam:>10.4g}  "
              f"loss={agg['val_loss_final_mean']:.4f}±{agg['val_loss_final_std']:.4f}   "
              f"acc={agg['val_acc_final_mean']:.1f}%±{agg['val_acc_final_std']:.1f}%")

    return results


def is_already_done(strategy_results, datasets):
    """Vrai si la stratégie est déjà calculée pour exactement la même liste de datasets."""
    if not strategy_results:
        return False
    first = strategy_results[0]
    return (
        'datasets' in first
        and list(first['datasets']) == list(datasets)
        and first.get('n_seeds') == len(datasets)
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expérience 0 : grid search λ multi-seed"
    )
    parser.add_argument('--clean', action='store_true',
                        help="Supprime les anciens résultats (sauf .html)")

    args = parser.parse_args()

    RESULTS_DIR = os.path.join(
        os.environ.get('CUSTOM_RESULTS_DIR',
                       os.path.join(EXP_DIR, 'Résultats')),
        'exp0'
    )
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')

    # Vérifier que tous les datasets existent avant de commencer
    for ds in DATASETS_DEFAULT:
        resolve_dataset_dir(ds)  # raise si pas trouvé

    print("=" * 70)
    print("  EXPÉRIENCE 0 : Grid search λ multi-seed")
    print("=" * 70)
    print(f"  Device         : {DEVICE}")
    print(f"  Datasets ({len(DATASETS_DEFAULT)}) : {', '.join(DATASETS_DEFAULT)}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Train/Val/Test : "
          f"{TRAIN_RATIO * TRAIN_VAL_RATIO:.0%} / "
          f"{TRAIN_RATIO * (1 - TRAIN_VAL_RATIO):.0%} / "
          f"{1 - TRAIN_RATIO:.0%}  (par dataset)")

    n_runs = sum(len(g) for g in LAMBDA_GRIDS.values()) * len(DATASETS_DEFAULT)
    print(f"  # runs prévus  : {n_runs}  "
          f"(4 stratégies × ~{n_runs // (4 * len(DATASETS_DEFAULT))} λ × {len(DATASETS_DEFAULT)} datasets)")

    if args.clean:
        print("  [!] Option --clean : suppression des fichiers non-HTML")
    print()

    if args.clean and os.path.exists(RESULTS_DIR):
        for item in os.listdir(RESULTS_DIR):
            if item.endswith('.html'):
                continue
            item_path = os.path.join(RESULTS_DIR, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    # Reprise éventuelle (par stratégie, si la liste de datasets correspond)
    all_results = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                all_results = json.load(f)
            print(f"  [i] Reprise : {len(all_results)} stratégie(s) déjà présente(s) dans metrics.json")
        except json.JSONDecodeError:
            print("  [!] metrics.json corrompu, on repart à zéro")
            all_results = {}

        # Purge des entrées au vieux format mono-seed (incompatibles avec le
        # pipeline multi-seed et la génération du data.js).
        for k in list(all_results.keys()):
            entries = all_results[k]
            if not entries or 'val_loss_final_mean' not in entries[0]:
                print(f"  [i] Purge ancienne entrée mono-seed : {k}")
                del all_results[k]
        print()

    # Grid search par stratégie
    for strategy_name, lambdas in LAMBDA_GRIDS.items():
        if (strategy_name in all_results
                and is_already_done(all_results[strategy_name], DATASETS_DEFAULT)):
            print(f"▶ {strategy_name} — déjà calculé sur ces datasets, on saute.\n")
            continue

        print(f"▶ {strategy_name} ({len(lambdas)} valeurs × {len(DATASETS_DEFAULT)} datasets)")
        results = run_grid_search(strategy_name, lambdas, DATASETS_DEFAULT)
        all_results[strategy_name] = results

        # Sauvegarde après chaque stratégie pour limiter la perte en cas de crash
        with open(metrics_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print()

    # Génération du data.js
    print("▶ Génération du data.js...")
    chart_data = {}
    for strategy_name, results in all_results.items():
        n_seeds = results[0]['n_seeds'] if results else 0
        per_seed_loss_final = [
            [r['per_seed'][s_idx]['val_loss_final'] for r in results]
            for s_idx in range(n_seeds)
        ]
        per_seed_acc_final = [
            [r['per_seed'][s_idx]['val_acc_final'] for r in results]
            for s_idx in range(n_seeds)
        ]

        chart_data[strategy_name] = {
            'lambdas':              [r['lambda']               for r in results],
            'val_loss_final_mean':  [r['val_loss_final_mean']  for r in results],
            'val_loss_final_std':   [r['val_loss_final_std']   for r in results],
            'val_acc_final_mean':   [r['val_acc_final_mean']   for r in results],
            'val_acc_final_std':    [r['val_acc_final_std']    for r in results],
            'val_loss_best_mean':   [r['val_loss_best_mean']   for r in results],
            'val_loss_best_std':    [r['val_loss_best_std']    for r in results],
            'val_acc_best_mean':    [r['val_acc_best_mean']    for r in results],
            'val_acc_best_std':     [r['val_acc_best_std']     for r in results],
            'best_epoch_mean':      [r['best_epoch_mean']      for r in results],
            'duration_mean':        [r['duration_mean']        for r in results],
            'duration_total':       [r['duration_total']       for r in results],
            'per_seed_loss_final':  per_seed_loss_final,
            'per_seed_acc_final':   per_seed_acc_final,
            'datasets':             results[0]['datasets'] if results else [],
            'n_seeds':              n_seeds,
            'color':                STRATEGY_COLORS.get(strategy_name, '#888'),
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
    print("  λ OPTIMAL PAR STRATÉGIE  (selon CE moyenne sur validation)")
    print("=" * 70)
    for strategy_name, results in all_results.items():
        best = min(results, key=lambda r: r['val_loss_final_mean'])
        print(f"  {strategy_name:14s}  "
              f"λ*={best['lambda']:>10.4g}   "
              f"val_loss={best['val_loss_final_mean']:.4f}±{best['val_loss_final_std']:.4f}   "
              f"val_acc={best['val_acc_final_mean']:.1f}%±{best['val_acc_final_std']:.1f}%")
    print()
