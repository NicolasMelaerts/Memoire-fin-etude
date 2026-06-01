"""
run_exp3.py - Expérience 3 : Métriques détaillées Normal vs GradCAM

Compare Normal vs Guided GradCAM (λ=0.1, 0.3, 0.5) avec 750 images d'entraînement.
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
import json
import shutil
import argparse

import init_env  # noqa: F401  (side effects: sys.path + DATASET_PATH)

from utils import set_seed
from shared.config import DEVICE, EPOCHS, SEED
from shared.dataset import get_dataloaders
from shared.detailed_metrics import compute_detailed_metrics

from shared.trainer import Trainer
from shared.strategies import NormalStrategy, GradCAMStrategy

# Prise en charge d'un dossier de résultats personnalisé pour l'exécution parallèle (run_multidataset)
RESULTS_DIR = os.path.join(
    os.environ.get('CUSTOM_RESULTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Résultats')),
    'exp3'
)

# Modèles d'exp3 dont les résultats peuvent être recyclés depuis exp1
# (mêmes données, même seed, même stratégie). Map: nom exp3 -> nom exp1.
RECYCLE_FROM_EXP1 = {
    'Normal':                  'Normal',
    'Guided GradCAM (λ=0.1)':  'Guided GradCAM (λ=0.1)',
}


def try_recycle_from_exp1(exp3_model_name):
    """
    Tente de récupérer history + duration + detailed_metrics depuis le
    metrics.json d'exp1 (dossier voisin). Retourne None si indisponible.
    """
    if exp3_model_name not in RECYCLE_FROM_EXP1:
        return None

    exp1_name = RECYCLE_FROM_EXP1[exp3_model_name]
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
    detailed = exp1_data.get('detailed_metrics', {}).get(exp1_name)
    if history is None or duration is None or detailed is None:
        return None

    return {
        'history': history,
        'duration': duration,
        'detailed_metrics': detailed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expérience 3 : Métriques détaillées")
    parser.add_argument('--clean', action='store_true',
                        help="Supprime les anciens résultats")
    parser.add_argument('--force', action='store_true',
                        help="Force l'entraînement même si déjà présent")
    parser.add_argument('--dataset', default='generated_dataset',
                        help="Nom du dossier dataset dans dataset_creator/ (défaut: generated_dataset)")
    args = parser.parse_args()

    LAMBDAS = [0.1, 0.3, 0.5]

    print("=" * 70)
    print("  EXPÉRIENCE 3 : Métriques détaillées Normal vs GradCAM")
    print("=" * 70)
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {EPOCHS}")
    print(f"  λ_gc  : {LAMBDAS}")
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
    print(f"  [i] Les DataLoaders seront recréés avant chaque modèle pour garantir la synchronisation RNG.\n")

    def train_and_evaluate(model_name, strategy):
        """Entraîne un modèle et calcule ses métriques détaillées (avec IoU).

        Si le modèle peut être recyclé depuis exp1 (même stratégie, mêmes
        données, même seed), on récupère directement le résultat au lieu de
        réentraîner.
        """
        if model_name in all_metrics and not args.force:
            print(f"⏭️  {model_name} : déjà entraîné (utilisez --force pour réentraîner)\n")
            return

        if not args.force:
            recycled = try_recycle_from_exp1(model_name)
            if recycled is not None:
                all_metrics[model_name] = recycled
                with open(metrics_path, 'w') as f:
                    json.dump(all_metrics, f, indent=2)
                dm = recycled['detailed_metrics']
                iou_str = f"{dm['iou']:.3f}±{dm['iou_std']:.3f}" if dm['iou'] is not None else "N/A"
                print(f"♻️  {model_name} : recyclé depuis exp1  "
                      f"(Acc={dm['accuracy']:.1%} F1={dm['f1']:.3f} IoU={iou_str})\n")
                return

        print(f"▶ Entraînement {model_name}...")
        set_seed(SEED)
        train_loader, test_loader, _ = get_dataloaders(seed=SEED, train_subset_size=750)

        trainer = Trainer(strategy, verbose=True)
        history, model, duration = trainer.run(train_loader, test_loader)

        print(f"  → Calcul des métriques détaillées (avec IoU)...")
        _, test_loader, _ = get_dataloaders(seed=SEED, train_subset_size=750)
        detailed = compute_detailed_metrics(model, test_loader, DEVICE, compute_iou_flag=True)

        all_metrics[model_name] = {
            'history': history,
            'duration': duration,
            'detailed_metrics': detailed,
        }
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        iou_str = f"{detailed['iou']:.3f}±{detailed['iou_std']:.3f}" if detailed['iou'] is not None else "N/A"
        print(f"  ✓ {duration:.1f}s - Accuracy={detailed['accuracy']:.1%} F1={detailed['f1']:.3f} IoU={iou_str}\n")

    # ===== MODÈLE 1 : NORMAL =====
    train_and_evaluate("Normal", NormalStrategy())

    # ===== MODÈLES 2..N : GRADCAM λ ∈ LAMBDAS =====
    for lam in LAMBDAS:
        train_and_evaluate(f"Guided GradCAM (λ={lam})", GradCAMStrategy(lambda_gc=lam))

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
