"""
run_exp2.py - Expérience 2 : Influence de la taille du dataset

Compare Normal vs Guided GradCAM avec différentes tailles de dataset :
  - 750, 500, 250, 200, 150, 100, 50, 25 images d'entraînement
  - Test fixé à 250 images (25% de 1000)

Usage :
    python run_exp2.py [--clean] [--dataset NOM]

Options :
  --clean          Supprime Résultats/exp2/ pour repartir à zéro
  --dataset NOM    Nom du dossier dataset dans dataset_creator/
                   (défaut : generated_dataset)
                   Ex: --dataset dataset_reel

Résultats :
    Résultats/exp2/exp2.html (double-cliquer pour ouvrir)
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

from shared.trainer import Trainer
from shared.strategies import NormalStrategy, GradCAMStrategy

# Prise en charge d'un dossier de résultats personnalisé pour l'exécution parallèle (run_multidataset)
RESULTS_DIR = os.path.join(
    os.environ.get('CUSTOM_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Résultats')),
    'exp2'
)

# Modèles d'exp2 (taille 750) recyclables depuis exp1 : même set_seed(SEED),
# même get_dataloaders(seed=SEED) (les 750 indices entraînement coïncident avec
# le train par défaut), même stratégie. Map: nom exp2 -> nom exp1.
RECYCLE_FROM_EXP1 = {
    'Normal (750 imgs)':                 'Normal',
    'Guided GradCAM (λ=0.1) (750 imgs)': 'Guided GradCAM (λ=0.1)',
}


def try_recycle_from_exp1(exp2_model_name):
    """Récupère history + duration depuis le metrics.json d'exp1 voisin
    (None si indisponible ou modèle non recyclable)."""
    if exp2_model_name not in RECYCLE_FROM_EXP1:
        return None
    exp1_name = RECYCLE_FROM_EXP1[exp2_model_name]
    exp1_metrics_path = os.path.join(os.path.dirname(RESULTS_DIR), 'exp1', 'metrics.json')
    if not os.path.exists(exp1_metrics_path):
        return None
    try:
        with open(exp1_metrics_path, 'r') as f:
            exp1_data = json.load(f)
    except json.JSONDecodeError:
        return None
    history = exp1_data.get('histories', {}).get(exp1_name)
    duration = exp1_data.get('durations', {}).get(exp1_name)
    if history is None or duration is None:
        return None
    return {'history': history, 'duration': duration}




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expérience 2 : Influence de la taille du dataset")
    parser.add_argument('--clean', action='store_true',
                        help="Supprime les anciens résultats")
    parser.add_argument('--dataset', default='generated_dataset',
                        help="Nom du dossier dataset dans dataset_creator/ (défaut: generated_dataset)")
    args = parser.parse_args()

    print("=" * 70)
    print("  EXPÉRIENCE 2 : Influence de la taille du dataset")
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

    # Le test set est fixé à 250 images
    test_ds = None
    print(f"\n  [i] Les DataLoaders seront recréés avant chaque modèle pour garantir la synchronisation RNG.\n")

    models_out = {}

    for idx, size in enumerate([750, 500, 250, 200, 150, 100, 50, 25]):
        name = f"Normal ({size} imgs)"

        # Tente de recycler depuis exp1 (uniquement size=750, modèle identique).
        recycled = try_recycle_from_exp1(name)
        if recycled is not None:
            histories[name] = recycled['history']
            durations[name] = recycled['duration']
            # Pas de models_out -> exemples GradCAM non générés pour ce modèle
            # (les exemples sont déjà dans le dashboard d'exp1).
            print(f"♻️  {name} : recyclé depuis exp1 "
                  f"(test_acc={recycled['history']['test_acc'][-1]:.1f}%)\n")
            continue

        print(f"▶ {name}...")
        dataset_seed = SEED
        model_seed = SEED + idx
        train_loader, test_loader, test_ds = get_dataloaders(seed=dataset_seed, train_subset_size=size)
        set_seed(model_seed)
        trainer = Trainer(NormalStrategy(), verbose=True)
        h, m, d = trainer.run(train_loader, test_loader)
        histories[name] = h
        models_out[name] = m
        durations[name] = d
        print(f"  ✓ {d:.1f}s - test_acc={h['test_acc'][-1]:.1f}%\n")

    # Guided GradCAM : on entraîne systématiquement λ=0.1 et λ=0.3 pour chaque
    # taille, afin que les moyennes inter-dataset soient comparables (mêmes
    # `n_runs` pour tous les modèles).
    LAMBDA_CANDIDATES = [0.1, 0.3]

    for idx, size in enumerate([750, 500, 250, 200, 150, 100, 50, 25]):
        train_loader, test_loader, test_ds = get_dataloaders(seed=SEED, train_subset_size=size)

        # Nettoyage des anciennes entrées GradCAM pour cette taille avant ré-écriture
        # (y compris d'éventuelles λ=0.5 issues d'anciennes exécutions).
        old_keys = [k for k in histories.keys() if "GradCAM" in k and f"({size} imgs)" in k]
        for k in old_keys:
            del histories[k]
            if k in durations:  del durations[k]
            if k in models_out: del models_out[k]

        for attempt, lambda_gc in enumerate(LAMBDA_CANDIDATES):
            name = f"Guided GradCAM (λ={lambda_gc}) ({size} imgs)"

            # Tente de recycler depuis exp1 (size=750, λ=0.1 uniquement).
            recycled = try_recycle_from_exp1(name)
            if recycled is not None:
                histories[name] = recycled['history']
                durations[name] = recycled['duration']
                print(f"♻️  {name} : recyclé depuis exp1 "
                      f"(test_acc={recycled['history']['test_acc'][-1]:.1f}%)\n")
                continue

            print(f"▶ {name}...")
            set_seed(SEED + idx + attempt * 100)

            trainer = Trainer(GradCAMStrategy(lambda_gc=lambda_gc), verbose=False)
            h, m, d = trainer.run(train_loader, test_loader)

            histories[name]   = h
            models_out[name]  = m
            durations[name]   = d
            print(f"  ✓ {d:.1f}s - test_acc={h['test_acc'][-1]:.1f}%\n")

    # ---- Génération des visuels ----
    print("▶ Génération des visuels...")

    # Couleurs dynamiques : pour chaque taille, le gagnant passe en vert ; les
    # perdants gardent la couleur identitaire de leur stratégie (STRATEGY_COLORS).
    import re
    WINNER_COLOR = '#4CAF50'  # vert

    def base_key(model_name):
        """Extrait la clé sans la taille : 'Guided GradCAM (λ=0.3) (500 imgs)' -> 'Guided GradCAM (λ=0.3)'."""
        return re.sub(r' \(\d+ imgs\)$', '', model_name)

    dynamic_colors = {}
    dynamic_styles = {}

    for size in [750, 500, 250, 200, 150, 100, 50, 25]:
        candidates = [k for k in histories.keys() if f"({size} imgs)" in k]
        if not candidates:
            continue

        winner = max(candidates, key=lambda k: histories[k]['test_acc'][-1])

        for name in candidates:
            if name == winner:
                dynamic_colors[name] = WINNER_COLOR
            else:
                dynamic_colors[name] = STRATEGY_COLORS.get(base_key(name), '#888')
            dynamic_styles[name] = '-' if 'Normal' in name else '--'

    def get_dynamic_color_and_style(name, idx):
        return dynamic_colors.get(name, '#000000'), dynamic_styles.get(name, '-')

    metrics_config = [
        ((0,), 'test_loss', 'Loss', 'Test Loss'),
        ((1,), 'test_acc', 'Accuracy (%)', 'Test Accuracy (%)'),
    ]
    plot_comparison(
        histories,
        os.path.join(RESULTS_DIR, 'comparison_curves.png'),
        epochs=EPOCHS,
        title="Expérience 2 : Normal vs GradCAM selon la taille du dataset",
        get_color_fn=get_dynamic_color_and_style,
        metrics=metrics_config
    )

    # Recharger test_ds si on a chargé depuis cache
    if test_ds is None:
        _, _, test_ds = get_dataloaders(seed=SEED, train_subset_size=750)

    examples = generate_gradcam_examples(
        models_out, test_ds, RESULTS_DIR, SEED,
        compute_gradcam_numpy, DEVICE, n_samples=50
    )

    save_data_js(histories, durations, examples, RESULTS_DIR, EPOCHS,
                 get_color_fn=get_dynamic_color_and_style)

    # Sauvegarder metrics.json
    with open(metrics_path, 'w') as f:
        json.dump({'histories': histories, 'durations': durations}, f, indent=2)
    print(f"  → Métriques : {metrics_path}")

    print()
    print("=" * 70)
    print("  RÉSULTATS FINAUX")
    print("=" * 70)
    for name, h in histories.items():
        print(f"  {name:35s} test_acc={h['test_acc'][-1]:.1f}%  "
              f"best={max(h['test_acc']):.1f}%  {durations[name]:.0f}s")

    print()
    print("  Pour afficher les résultats :")
    print(f"    Double-cliquer sur : Résultats/exp2/exp2.html")
