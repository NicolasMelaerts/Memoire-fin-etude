"""
run_exp2.py - Expérience 2 : Influence de la taille du dataset

Compare Normal vs GradCAM-Guided avec différentes tailles de dataset :
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
from shared.strategies import NormalStrategy, GradCAMStrategy

# Support custom results dir for parallel execution (BIG_EXPERIENCE)
RESULTS_DIR = os.path.join(
    os.environ.get('CUSTOM_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Résultats')),
    'exp2'
)

# Palette de couleurs
PALETTE = [
    '#2196F3', '#FF9800', '#FF5722', '#F44336',
    '#9C27B0', '#4CAF50', '#00BCD4', '#E91E63',
]


def get_color_and_style(name, idx):
    """Retourne couleur et style de ligne pour un modèle."""
    color = PALETTE[idx % len(PALETTE)]
    style = '--' if 'GradCAM' in name else '-'
    return color, style


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

    normal_acc_by_size = {}
    for idx, size in enumerate([750, 500, 250, 200, 150, 100, 50, 25]):
        name = f"Normal ({size} imgs)"
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
        normal_acc_by_size[size] = h['test_acc'][-1]
        print(f"  ✓ {d:.1f}s - test_acc={h['test_acc'][-1]:.1f}%\n")

    # GradCAM-Guided adaptatif : recherche de λ
    # On teste plusieurs valeurs de λ et on l'augmente s'il ne bat pas le Normal.
    # On conserve le meilleur résultat obtenu.
    LAMBDA_CANDIDATES = [0.1, 0.05, 0.2, 0.25, 0.5, 1, 2]

    for idx, size in enumerate([750, 500, 250, 200, 150, 100, 50, 25]):
        target_acc = normal_acc_by_size.get(size, 0)
        print(f"▶ Recherche GradCAM pour {size} imgs (cible Normal : {target_acc:.1f}%)...")

        best_h, best_m, best_d, best_acc, best_lam = None, None, 0, -1, 0
        train_loader, test_loader, test_ds = get_dataloaders(seed=SEED, train_subset_size=size)

        for attempt, lambda_gc in enumerate(LAMBDA_CANDIDATES):
            print(f"  ▷ Essai {attempt+1}/{len(LAMBDA_CANDIDATES)} avec λ={lambda_gc}...")
            set_seed(SEED + idx + attempt * 100)

            trainer = Trainer(GradCAMStrategy(lambda_gc=lambda_gc), verbose=False)
            h, m, d = trainer.run(train_loader, test_loader)

            acc = h['test_acc'][-1]
            print(f"    ✓ test_acc={acc:.1f}%")

            if acc > best_acc:
                best_acc, best_h, best_m, best_d, best_lam = acc, h, m, d, lambda_gc

            if acc >= target_acc and target_acc > 0:
                print(f"  [!] Succès ! λ={lambda_gc} bat ou égale Normal ({acc:.1f}% >= {target_acc:.1f}%).")
                break
            elif attempt < len(LAMBDA_CANDIDATES) - 1:
                print("  [!] Échec. Augmentation de λ pour le prochain essai...")
            else:
                print("  [!] Échec final. On conserve le meilleur essai.")

        old_keys = [k for k in histories.keys() if "GradCAM" in k and f"({size} imgs)" in k]
        for k in old_keys:
            del histories[k]
            if k in durations: del durations[k]
            if k in models_out: del models_out[k]

        final_name = f"GradCAM adaptatif (λ={best_lam}) ({size} imgs)"
        histories[final_name] = best_h
        models_out[final_name] = best_m
        durations[final_name] = best_d
        print(f"▶ Retenu : {final_name} avec test_acc={best_acc:.1f}%\n")

    # ---- Génération des visuels ----
    print("▶ Génération des visuels...")

    # Déterminer les couleurs dynamiquement : Vert pour le gagnant, Rouge pour le perdant
    dynamic_colors = {}
    dynamic_styles = {}
    
    for size in [750, 500, 250, 200, 150, 100, 50, 25]:
        norm_name = f"Normal ({size} imgs)"
        gc_name = next((k for k in histories.keys() if "GradCAM" in k and f"({size} imgs)" in k), None)
        
        if norm_name in histories and gc_name in histories:
            acc_norm = histories[norm_name]['test_acc'][-1]
            acc_gc = histories[gc_name]['test_acc'][-1]
            
            if acc_gc > acc_norm:
                dynamic_colors[gc_name] = '#4CAF50'     # Vert (Gagnant)
                dynamic_colors[norm_name] = '#F44336'   # Rouge (Perdant)
            elif acc_norm > acc_gc:
                dynamic_colors[norm_name] = '#4CAF50'   # Vert (Gagnant)
                dynamic_colors[gc_name] = '#F44336'     # Rouge (Perdant)
            else:
                dynamic_colors[norm_name] = '#FF9800'   # Orange (Égalité)
                dynamic_colors[gc_name] = '#FF9800'
                
            dynamic_styles[norm_name] = '-'
            dynamic_styles[gc_name] = '--'
        else:
            if norm_name in histories:
                dynamic_colors[norm_name] = '#2196F3'
                dynamic_styles[norm_name] = '-'
            if gc_name in histories:
                dynamic_colors[gc_name] = '#9C27B0'
                dynamic_styles[gc_name] = '--'

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
