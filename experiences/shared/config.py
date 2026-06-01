"""
config.py - Hyperparamètres partagés pour l'expérience de comparaison.
"""
import os
import torch

# --- Chemins ---
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR       = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
# Supporte un chemin de dataset personnalisé via variable d'environnement (utilisé par run_multidataset)
DATA_DIR       = os.environ.get('DATASET_PATH', os.path.join(ROOT_DIR, 'dataset_creator', 'generated_dataset'))
IMAGES_DIR     = os.path.join(DATA_DIR, 'images')
HEATMAPS_DIR   = os.path.join(DATA_DIR, 'heatmaps')
ANNOTATIONS    = os.path.join(DATA_DIR, 'annotations.csv')
# Supporte un dossier de résultats personnalisé via variable d'environnement (utilisé par run_multidataset pour l'exécution en parallèle)
RESULTS_DIR    = os.environ.get('CUSTOM_RESULTS_DIR', os.path.join(EXPERIMENT_DIR, 'results'))

# --- Matériel ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Entraînement ---
EPOCHS        = 15
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
SEED          = 2026
TRAIN_RATIO   = 0.75  # 75% train (750) / 25% test (250)

# --- Palette partagée par stratégie ---
# Référence unique pour conserver la même couleur d'une expérience à l'autre.
# Pour GradCAM, les trois λ sont nuancés du clair au foncé pour rester
# distinguables visuellement quand ils apparaissent dans un même plot.
STRATEGY_COLORS = {
    # Noms canoniques (exp0)
    'Normal':                   '#2196F3',  # bleu
    'Double BP':                '#FF9800',  # orange
    'Guided GradCAM':           '#9C27B0',  # violet
    'GAIN':                     '#4CAF50',  # vert
    'RRR':                      '#F44336',  # rouge

    # Variants λ de base (exp1) — couleur de la stratégie pour les non-GradCAM
    'Double BP (λ=200.0)':      '#FF9800',
    'GAIN (λ=0.1)':             '#4CAF50',
    'RRR (λ=0.1)':              '#F44336',

    # Variants GradCAM : nuances de violet (clair → foncé selon λ)
    'Guided GradCAM (λ=0.1)':   '#CE93D8',  # violet clair
    'Guided GradCAM (λ=0.3)':   '#9C27B0',  # violet moyen
    'Guided GradCAM (λ=0.5)':   '#6A1B9A',  # violet foncé
}

