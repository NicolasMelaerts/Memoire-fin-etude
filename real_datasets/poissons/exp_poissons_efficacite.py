"""
exp_poissons_efficacite.py — GradCAM avec moins de données (suite de exp_poissons.py)

Réutilise les résultats déjà calculés dans exp_poissons.py :
  - Normal  100%  → chargé depuis results/metrics.json
  - GradCAM 100%  → chargé depuis results/metrics.json

N'entraîne que les variantes GradCAM avec moins d'images :
  90%, 80%, 70%, 60%, 50%, 40%, 30%, 20%, 10%
  (même split exact que exp_poissons.py : DATASET_FRAC=0.25)

Hypothèse : la supervision spatiale GradCAM permet d'atteindre les mêmes
            performances qu'un Normal 100% avec moins d'images.

Usage :
    python3 exp_poissons_efficacite.py [--clean]
"""
import os
import json
import time
import shutil
import argparse
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FISH_DIR     = os.path.join(BASE_DIR, 'fish-dataset')
RESULTS1_DIR = os.path.join(BASE_DIR, 'results')              # résultats exp_poissons.py
RESULTS_DIR  = os.path.join(BASE_DIR, 'results_efficacite')

# ---------------------------------------------------------------------------
# Hyperparamètres — identiques à run_fish.py
# ---------------------------------------------------------------------------
IMG_SIZE     = 128
BATCH_SIZE   = 32
EPOCHS       = 15
LR           = 1e-3
SEED         = 42
LAMBDA_GC    = 0.1
TRAIN_RATIO  = 0.8
DATASET_FRAC = 0.25   # même valeur que run_fish.py → même split
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASSES = sorted([
    d for d in os.listdir(FISH_DIR)
    if os.path.isdir(os.path.join(FISH_DIR, d)) and not d.endswith(' GT')
]) if os.path.isdir(FISH_DIR) else []
N_CLASSES = len(CLASSES)

# Fractions à tester (de 90% à 10%, par pas de 10%)
GC_FRACTIONS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------------------------------------------------------------------------
# Modèle — identique à run_fish.py
# ---------------------------------------------------------------------------
class SimpleCNN_RGB(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.conv1         = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2         = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3         = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool          = nn.MaxPool2d(2, 2)
        self.fc1           = nn.Linear(64 * 16 * 16, 128)
        self.dropout       = nn.Dropout(0.5)
        self.fc2           = nn.Linear(128, n_classes)
        self.gradcam_layer = self.conv3

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return F.relu(self.conv3(x))

    def forward_from_features(self, feat):
        x = self.pool(feat)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Dataset — identique à run_fish.py
# ---------------------------------------------------------------------------
class FishDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform or transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.seg_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        if os.path.exists(gt_path):
            gt = Image.open(gt_path).convert('L')
            heatmap = self.seg_transform(gt)
            heatmap = (heatmap > 0.5).float()
        else:
            heatmap = torch.ones(1, IMG_SIZE, IMG_SIZE) / (IMG_SIZE * IMG_SIZE)
        return image, label, heatmap, img_path


def load_fish_splits():
    """Reproduit exactement le split de run_fish.py (DATASET_FRAC=0.25, SEED=42)."""
    all_samples = []
    for label_idx, class_name in enumerate(CLASSES):
        img_dir = os.path.join(FISH_DIR, class_name, class_name)
        gt_dir  = os.path.join(FISH_DIR, class_name, class_name + ' GT')
        if not os.path.isdir(img_dir):
            continue
        filenames = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        for fname in filenames:
            all_samples.append((
                os.path.join(img_dir, fname),
                os.path.join(gt_dir,  fname),
                label_idx,
            ))

    rng = random.Random(SEED)
    rng.shuffle(all_samples)
    if DATASET_FRAC < 1.0:
        all_samples = all_samples[:max(1, int(len(all_samples) * DATASET_FRAC))]
    split = int(len(all_samples) * TRAIN_RATIO)
    return all_samples[:split], all_samples[split:]


def make_loaders(train_samples, test_samples, frac=1.0):
    """Construit les dataloaders pour une fraction des données d'entraînement."""
    n   = max(1, int(len(train_samples) * frac))
    sub = random.Random(SEED).sample(train_samples, n)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = FishDataset(sub,          transform=train_transform)
    test_ds  = FishDataset(test_samples, transform=test_transform)

    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, generator=gen)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)
    return train_loader, test_loader, test_ds


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------
def compute_gradcam_differentiable(model, image_batch, target_classes):
    B    = image_batch.size(0)
    feat = model.forward_features(image_batch)
    feat.retain_grad()
    logits = model.forward_from_features(feat)
    scores = logits[torch.arange(B), target_classes]
    grads  = torch.autograd.grad(
        scores.sum(), feat, create_graph=True, retain_graph=True
    )[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam     = F.relu((weights * feat).sum(dim=1, keepdim=True))
    cam     = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    cam_flat = cam.view(B, -1)
    mn = cam_flat.min(dim=1, keepdim=True).values
    mx = cam_flat.max(dim=1, keepdim=True).values
    cam = ((cam_flat - mn) / (mx - mn + 1e-8)).view(B, 1, IMG_SIZE, IMG_SIZE)
    return cam, logits


def compute_gradcam_numpy(model, img_tensor):
    model.eval()
    grads_list, acts_list = [], []
    h1 = model.gradcam_layer.register_full_backward_hook(
        lambda m, gi, go: grads_list.append(go[0]))
    h2 = model.gradcam_layer.register_forward_hook(
        lambda m, inp, out: acts_list.append(out))
    x      = img_tensor.unsqueeze(0).to(DEVICE)
    logits = model(x)
    pred   = logits.argmax(dim=1).item()
    model.zero_grad()
    logits[0, pred].backward()
    grads = grads_list[0].detach().cpu().numpy()[0]
    acts  = acts_list[0].detach().cpu().numpy()[0]
    h1.remove(); h2.remove()
    weights = grads.mean(axis=(1, 2))
    cam     = (weights[:, None, None] * acts).sum(0)
    cam     = np.maximum(cam, 0)
    cam     = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    mn, mx  = cam.min(), cam.max()
    return (cam - mn) / (mx - mn + 1e-8), pred, logits.detach()


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------
def localisation_loss(cam_batch, heatmaps):
    B = cam_batch.size(0)
    return 1.0 - F.cosine_similarity(
        cam_batch.view(B, -1),
        heatmaps.view(B, -1).to(cam_batch.device),
        dim=1
    ).mean()


def train_epoch_gradcam(model, loader, optimizer):
    model.train()
    total_loss = total_ce = total_correct = total = 0
    for images, labels, heatmaps, _ in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        heatmaps       = heatmaps.to(DEVICE)
        optimizer.zero_grad()
        cam, logits = compute_gradcam_differentiable(model, images, labels)
        ce   = F.cross_entropy(logits, labels)
        loc  = localisation_loss(cam, heatmaps)
        loss = ce + LAMBDA_GC * loc
        loss.backward()
        optimizer.step()
        total_loss    += loss.item()
        total_ce      += ce.item()
        total_correct += (logits.argmax(1) == labels).sum().item()
        total         += labels.size(0)
    return total_loss / len(loader), total_ce / len(loader), 100. * total_correct / total


def eval_epoch(model, loader):
    model.eval()
    total_loss = total_correct = total = 0
    with torch.no_grad():
        for images, labels, _, _ in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            total_loss    += F.cross_entropy(logits, labels).item()
            total_correct += (logits.argmax(1) == labels).sum().item()
            total         += labels.size(0)
    return total_loss / len(loader), 100. * total_correct / total


def compute_mean_iou(model, test_ds, n_samples=200):
    model.eval()
    indices = random.sample(range(len(test_ds)), min(n_samples, len(test_ds)))
    ious = []
    for idx in indices:
        img_tensor, _, heatmap, _ = test_ds[idx]
        cam, _, _ = compute_gradcam_numpy(model, img_tensor)
        gt = cv2.resize(heatmap.squeeze().numpy().astype(np.float32), (IMG_SIZE, IMG_SIZE))
        inter = ((cam >= 0.5) & (gt >= 0.5)).sum()
        union = ((cam >= 0.5) | (gt >= 0.5)).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def train_model(name, train_loader, test_loader, epochs, verbose=True):
    set_seed(SEED)
    model     = SimpleCNN_RGB(n_classes=N_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {'train_loss': [], 'train_acc': [], 'train_ce': [],
               'test_loss':  [], 'test_acc':  []}
    t0 = time.time()

    for epoch in range(epochs):
        tr_loss, tr_ce, tr_acc = train_epoch_gradcam(model, train_loader, optimizer)
        te_loss, te_acc        = eval_epoch(model, test_loader)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['train_ce'].append(tr_ce)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)

        if verbose:
            print(f"  [{name}] Epoch {epoch+1:02d}/{epochs}  "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.1f}%  "
                  f"test_loss={te_loss:.4f}  test_acc={te_acc:.1f}%")

    return model, history, time.time() - t0


# ---------------------------------------------------------------------------
# Courbes PNG
# ---------------------------------------------------------------------------
def plot_curves(all_histories, all_configs, epochs, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Normal 100% vs GradCAM avec moins de données', fontsize=13, fontweight='bold')
    for name, mode, frac, n_train in all_configs:
        h = all_histories.get(name)
        if h is None:
            continue
        ep = range(1, len(h['test_acc']) + 1)
        if mode == 'normal':
            color, ls, lw, zorder = '#2196F3', '-',  2.5, 10
        elif frac == 1.0:
            color, ls, lw, zorder = '#FF9800', '--', 2.0, 9
        else:
            # dégradé du jaune vers le rouge selon la fraction
            t     = 1.0 - frac           # 0 (100%) → 1 (10%)
            r     = int(255)
            g     = int(152 * (1 - t))
            color = f'#{r:02x}{g:02x}00'
            ls, lw, zorder = '--', 1.5, 5

        label = f"{name} (n={n_train})"
        axes[0].plot(ep, h['test_acc'],  color=color, ls=ls, lw=lw, zorder=zorder, label=label)
        axes[1].plot(ep, h['test_loss'], color=color, ls=ls, lw=lw, zorder=zorder, label=label)

    for ax, title, ylabel in [
        (axes[0], 'Test Accuracy (%)', 'Accuracy (%)'),
        (axes[1], 'Test Loss',         'Loss'),
    ]:
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Courbes PNG : {output_path}")


# ---------------------------------------------------------------------------
# data2.js
# ---------------------------------------------------------------------------
def save_data_js(all_histories, all_durations, all_iou, all_configs,
                 epochs, n_train_full, n_test, results_dir):
    configs_json = [
        {'name': name, 'mode': mode, 'frac': frac, 'n_train': n_train}
        for name, mode, frac, n_train in all_configs
    ]
    data = {
        'epochs':    list(range(1, epochs + 1)),
        'histories': all_histories,
        'durations': all_durations,
        'iou_scores': all_iou,
        'configs':   configs_json,
        'classes':   CLASSES,
        'n_classes': N_CLASSES,
        'n_train_full': n_train_full,
        'n_test':    n_test,
        'baseline':  'Normal',
    }
    js_path = os.path.join(results_dir, 'data2.js')
    with open(js_path, 'w', encoding='utf-8') as f:
        f.write('window.FISH2_DATA = ')
        json.dump(data, f, indent=2)
        f.write(';\n')
    print(f"  → data2.js : {js_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fish : GradCAM avec moins de données (suite run_fish.py)')
    parser.add_argument('--clean',  action='store_true')
    args = parser.parse_args()

    print('=' * 70)
    print('  Fish Dataset : Normal 100% vs GradCAM avec moins de données')
    print('=' * 70)
    print(f'  Device  : {DEVICE}')
    print(f'  Epochs  : {EPOCHS}')
    print(f'  Fractions GradCAM : {[f"{int(f*100)}%" for f in GC_FRACTIONS]}')
    print()

    if args.clean and os.path.exists(RESULTS_DIR):
        for item in os.listdir(RESULTS_DIR):
            p = os.path.join(RESULTS_DIR, item)
            if item.endswith('.html'): continue
            if os.path.isfile(p): os.remove(p)
            elif os.path.isdir(p): shutil.rmtree(p)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Charger les résultats déjà calculés depuis run_fish.py
    # ------------------------------------------------------------------
    metrics1_path = os.path.join(RESULTS1_DIR, 'metrics.json')
    histories1, durations1, iou1 = {}, {}, {}
    if os.path.exists(metrics1_path):
        with open(metrics1_path) as f:
            saved = json.load(f)
        histories1 = saved.get('histories', {})
        durations1 = saved.get('durations', {})
        iou1       = saved.get('iou_scores', {})
        print(f'  [✓] Chargé depuis results/metrics.json : {list(histories1.keys())}')
    else:
        print('  [!] results/metrics.json introuvable — lancez exp_poissons.py d\'abord.')

    # ------------------------------------------------------------------
    # 2. Cache propre à exp_poissons_efficacite.py
    # ------------------------------------------------------------------
    metrics2_path = os.path.join(RESULTS_DIR, 'metrics2.json')
    histories2, durations2, iou2 = {}, {}, {}
    if os.path.exists(metrics2_path):
        try:
            with open(metrics2_path) as f:
                saved2 = json.load(f)
            histories2 = saved2.get('histories', {})
            durations2 = saved2.get('durations', {})
            iou2       = saved2.get('iou_scores', {})
            print(f'  [✓] Cache local : {list(histories2.keys())}')
        except json.JSONDecodeError:
            print('  [!] metrics2.json corrompu, ignoré')
    print()

    # ------------------------------------------------------------------
    # 3. Charger le split identique à run_fish.py
    # ------------------------------------------------------------------
    print('▶ Chargement du dataset (même split que run_fish.py)...')
    set_seed(SEED)
    train_samples, test_samples = load_fish_splits()
    n_train_full = len(train_samples)
    print(f'  Train (100%) : {n_train_full} images  |  Test : {len(test_samples)} images')
    print()

    # ------------------------------------------------------------------
    # 4. Configs complètes = résultats run_fish.py + nouvelles fractions
    # ------------------------------------------------------------------
    # Noms depuis run_fish.py
    NAME_NORMAL = 'Normal'
    NAME_GC100  = 'GradCAM (λ=0.1)'

    all_configs = [
        (NAME_NORMAL, 'normal',  1.0, n_train_full),
        (NAME_GC100,  'gradcam', 1.0, n_train_full),
    ]
    for frac in GC_FRACTIONS:
        n = max(1, int(n_train_full * frac))
        all_configs.append((f'GradCAM {int(frac*100)}%', 'gradcam', frac, n))

    # Fusionner les historiques
    all_histories = {**histories1, **histories2}
    all_durations = {**durations1, **durations2}
    all_iou       = {**iou1,       **iou2}

    # ------------------------------------------------------------------
    # 5. Entraîner uniquement les fractions manquantes
    # ------------------------------------------------------------------
    models_out = {}

    for name, mode, frac, n_train in all_configs:
        # Sauter les modèles déjà disponibles (run_fish.py ou cache)
        if name in all_histories:
            print(f'  [⏭]  {name} : déjà disponible')
            continue
        # Sauter Normal et GradCAM 100% s'ils ne sont pas dans le cache
        # (ils doivent venir de run_fish.py)
        if name in (NAME_NORMAL, NAME_GC100):
            print(f'  [!]  {name} introuvable — lancez exp_poissons.py d\'abord.')
            continue

        train_loader, test_loader, test_ds = make_loaders(train_samples, test_samples, frac)
        print(f'▶ Entraînement {name}  ({n_train} images train)...')
        model, h, d = train_model(name, train_loader, test_loader, EPOCHS)
        all_histories[name] = h
        all_durations[name] = d
        models_out[name]    = (model, test_ds)

        # Sauvegarder uniquement les nouveaux résultats dans metrics2.json
        histories2[name] = h
        durations2[name] = d
        with open(metrics2_path, 'w') as f:
            json.dump({'histories': histories2, 'durations': durations2,
                       'iou_scores': iou2}, f, indent=2)
        print(f'  ✓ {d:.0f}s — test_acc={h["test_acc"][-1]:.1f}%\n')

    # ------------------------------------------------------------------
    # 6. IoU pour les nouveaux modèles seulement
    # ------------------------------------------------------------------
    for name, mode, frac, n_train in all_configs:
        if name in (NAME_NORMAL, NAME_GC100):
            continue   # IoU déjà dans iou1
        if name not in all_iou:
            if name in models_out:
                model, test_ds = models_out[name]
                print(f'▶ Calcul IoU pour {name}...')
                iou2[name]    = compute_mean_iou(model, test_ds)
                all_iou[name] = iou2[name]
                print(f'  IoU {name} = {all_iou[name]:.3f}')
                with open(metrics2_path, 'w') as f:
                    json.dump({'histories': histories2, 'durations': durations2,
                               'iou_scores': iou2}, f, indent=2)

    # ------------------------------------------------------------------
    # 7. Sorties
    # ------------------------------------------------------------------
    plot_curves(all_histories, all_configs, EPOCHS,
                os.path.join(RESULTS_DIR, 'curves2.png'))

    save_data_js(all_histories, all_durations, all_iou, all_configs,
                 EPOCHS, n_train_full, len(test_samples), RESULTS_DIR)

    # ------------------------------------------------------------------
    # 8. Résumé terminal
    # ------------------------------------------------------------------
    print()
    print('=' * 70)
    print('  RÉSULTATS FINAUX')
    print('=' * 70)
    baseline_acc = all_histories.get(NAME_NORMAL, {}).get('test_acc', [0])[-1]
    print(f'  {"Modèle":<22} {"n_train":>8}  {"test_acc":>9}  {"vs Normal 100%":>14}  {"IoU":>6}')
    print(f'  {"-"*22}  {"-"*8}  {"-"*9}  {"-"*14}  {"-"*6}')
    for name, mode, frac, n_train in all_configs:
        if name not in all_histories: continue
        h     = all_histories[name]
        acc   = h['test_acc'][-1]
        iou   = all_iou.get(name, 0)
        delta = acc - baseline_acc
        sign  = '+' if delta >= 0 else ''
        print(f'  {name:<22}  {n_train:>8}  {acc:>8.1f}%  {sign}{delta:>+13.1f}%  {iou:>6.3f}')

    # Trouver le point de croisement
    crossover = None
    for name, mode, frac, n_train in all_configs:
        if mode == 'gradcam' and name in all_histories:
            acc = all_histories[name]['test_acc'][-1]
            if acc >= baseline_acc:
                crossover = (name, frac, n_train, acc)
    print()
    if crossover:
        name, frac, n_train, acc = crossover
        print(f'  ★ GradCAM rattrape Normal 100% dès {int(frac*100)}% des données '
              f'({n_train} images, acc={acc:.1f}% vs {baseline_acc:.1f}%)')
    else:
        print(f'  Aucun modèle GradCAM n\'atteint la baseline Normal 100% '
              f'({baseline_acc:.1f}%) avec moins de données.')

    print()
    print('  Double-cliquer sur : real_datasets/poissons/results_efficacite/fish_results_2.html')
