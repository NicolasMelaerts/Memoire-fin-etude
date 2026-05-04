"""
config.py - Hyperparamètres partagés pour l'expérience de comparaison.
"""
import os
import torch

# --- Paths ---
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR       = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
# Support custom dataset path via environment variable (used by BIG_EXPERIENCE)
DATA_DIR       = os.environ.get('DATASET_PATH', os.path.join(ROOT_DIR, 'dataset_creator', 'generated_dataset'))
IMAGES_DIR     = os.path.join(DATA_DIR, 'images')
HEATMAPS_DIR   = os.path.join(DATA_DIR, 'heatmaps')
ANNOTATIONS    = os.path.join(DATA_DIR, 'annotations.csv')
# Support custom results dir via environment variable (used by BIG_EXPERIENCE for parallel execution)
RESULTS_DIR    = os.environ.get('CUSTOM_RESULTS_DIR', os.path.join(EXPERIMENT_DIR, 'results'))

# --- Hardware ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Training ---
EPOCHS        = 10
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
SEED          = 42
TRAIN_RATIO   = 0.75  # 75% train (750) / 25% test (250)

