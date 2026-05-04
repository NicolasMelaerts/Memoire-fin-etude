"""
dataset.py - Dataset et DataLoaders partagés pour les 3 expériences.

Le ShapeDataset charge :
  - l'image originale (grayscale → tensor)
  - le label (inside=0, outside=1)
  - la heatmap de supervision (pour GradCAM-guided) :
      * si une heatmap .npy existe dans HEATMAPS_DIR → on la charge (valeurs [0,1])
      * sinon → heatmap uniforme d'importance (fallback)

Note: Les heatmaps sont sauvegardées en .npy pour préserver les valeurs exactes [0,1]
      sans distorsion due aux conversions de couleur.
"""
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms

from .config import (
    ANNOTATIONS, IMAGES_DIR, HEATMAPS_DIR,
    BATCH_SIZE, TRAIN_RATIO, SEED, DEVICE
)


class ShapeDataset(Dataset):
    """Dataset pour les images cercle/triangle (inside / outside)."""

    # Le CSV a des colonnes : filename, class, case
    # Les valeurs de 'class' sont 'positive' (triangle dans le cercle) et 'negative'
    LABEL_MAP = {'positive': 0, 'negative': 1}  # 0=inside, 1=outside

    def __init__(self, annotations_file, img_dir, heatmaps_dir=None, transform=None):
        self.df          = pd.read_csv(annotations_file)
        self.img_dir     = img_dir
        self.heatmaps_dir = heatmaps_dir
        self.transform   = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name  = self.df.iloc[idx, 0]         # colonne 'filename'
        label_str = self.df.iloc[idx]['class']     # colonne 'class' => 'positive' ou 'negative'

        # Image (grayscale → [1, 128, 128])
        img_path = os.path.join(self.img_dir, img_name)
        image    = Image.open(img_path).convert('L')
        image    = self.transform(image)

        # Label
        label = self.LABEL_MAP.get(label_str, 0)

        # Heatmap de supervision  (float32, [1, 128, 128], valeurs dans [0,1])
        heatmap = self._load_heatmap(img_name)

        return image, label, heatmap, img_name

    def _load_heatmap(self, img_name):
        """Charge la heatmap .npy (valeurs brutes) ou retourne une heatmap uniforme.

        Les heatmaps sont sauvegardées en .npy pour préserver les valeurs exactes [0,1]
        sans distorsion due aux conversions de couleur RGB → Grayscale.
        """
        if self.heatmaps_dir:
            # Convention de nommage : image_XXXXX_heatmap.npy
            stem = os.path.splitext(img_name)[0]
            hm_name = f"{stem}_heatmap.npy"
            hm_path = os.path.join(self.heatmaps_dir, hm_name)

            if os.path.exists(hm_path):
                # Charger les valeurs brutes [0, 1] directement
                hm_array = np.load(hm_path)  # Shape: (128, 128)
                hm_tensor = torch.from_numpy(hm_array).float().unsqueeze(0)  # [1, 128, 128]
                return hm_tensor

        # Fallback : heatmap uniforme (pas de supervision spatiale)
        return torch.ones(1, 128, 128, dtype=torch.float32) / (128 * 128)


def get_dataloaders(seed=SEED, train_subset_size=None):
    """Retourne train_loader et test_loader avec un split reproductible.

    Args:
        seed: Random seed for reproducibility
        train_subset_size: If provided, limits the training set to this size
    """
    generator = torch.Generator().manual_seed(seed)

    # Augmentation de données pour réduire l'overfitting
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Rotation ±10°
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Translation légère
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Dataset pour le training (avec augmentation)
    train_full_ds = ShapeDataset(
        annotations_file=ANNOTATIONS,
        img_dir=IMAGES_DIR,
        heatmaps_dir=HEATMAPS_DIR,
        transform=train_transform,
    )

    # Dataset pour le test (sans augmentation)
    test_full_ds = ShapeDataset(
        annotations_file=ANNOTATIONS,
        img_dir=IMAGES_DIR,
        heatmaps_dir=HEATMAPS_DIR,
        transform=test_transform,
    )

    # Calculer le nombre d'images pour le split (75% / 25% par défaut)
    n_total = len(train_full_ds)
    n_train_default = int(TRAIN_RATIO * n_total)
    
    # Générer une permutation unique des indices pour garantir que train/test sont disjoints
    indices = torch.randperm(n_total, generator=generator).tolist()
    
    # Déterminer les indices de test (toujours les mêmes 25% du pool global)
    test_indices = indices[n_train_default:]
    
    # Déterminer les indices de train
    # Si train_subset_size est spécifié, on prend les X premiers indices de la partie 'train'
    # Sinon on prend toute la partie 'train' (75%)
    all_train_indices = indices[:n_train_default]
    if train_subset_size is not None:
        subset_size = min(train_subset_size, len(all_train_indices))
        train_indices = all_train_indices[:subset_size]
    else:
        train_indices = all_train_indices

    # Créer les subsets
    train_ds = Subset(train_full_ds, train_indices)
    test_ds  = Subset(test_full_ds, test_indices)

    # Créer un generator pour le shuffle du DataLoader
    # Cela garantit que l'ordre des batches sera IDENTIQUE entre les exécutions
    dataloader_generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False,
        generator=dataloader_generator  # Fixe l'ordre du shuffle
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=False
    )
    return train_loader, test_loader, test_ds
