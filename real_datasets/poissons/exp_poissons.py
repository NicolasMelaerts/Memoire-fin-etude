"""
exp_poissons.py — Expérience Fish Dataset : Normal vs GradCAM-Guided

Dataset réel : A Large Scale Fish Dataset (5 espèces, 1000 images/classe)
Supervision  : masques de segmentation binaires (poisson vs fond)

Hypothèse : en forçant le modèle à regarder le poisson (et non le fond),
            la guidance GradCAM améliore l'accuracy et la qualité des explications.

Usage :
    python3 exp_poissons.py [--clean]
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
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FISH_DIR    = os.path.join(BASE_DIR, 'fish-dataset')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ---------------------------------------------------------------------------
# Hyperparamètres
# ---------------------------------------------------------------------------
IMG_SIZE     = 128
BATCH_SIZE   = 32
EPOCHS       = 15
LR           = 1e-3
SEED         = 42
LAMBDA_GC    = 0.1
TRAIN_RATIO  = 0.8
DATASET_FRAC = 0.25   # fraction du dataset à utiliser (1.0 = tout, 0.1 = 10%)
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Classes du dataset (découvertes dynamiquement)
CLASSES = sorted([
    d for d in os.listdir(FISH_DIR)
    if os.path.isdir(os.path.join(FISH_DIR, d)) and not d.endswith(' GT')
]) if os.path.isdir(FISH_DIR) else []
N_CLASSES = len(CLASSES)


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
# Modèle — SimpleCNN adapté RGB / N classes
# ---------------------------------------------------------------------------
class SimpleCNN_RGB(nn.Module):
    """SimpleCNN pour images RGB 128×128 et N classes."""

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
        return F.relu(self.conv3(x))            # [B, 64, 32, 32]

    def forward_from_features(self, feat):
        x = self.pool(feat)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Dataset Fish
# ---------------------------------------------------------------------------
class FishDataset(Dataset):
    """Charge images Fish Dataset + masques GT comme heatmaps de supervision."""

    def __init__(self, samples, transform=None):
        # samples : liste de (img_path, gt_path, label_idx)
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
    """Parcourt le dossier Fish et construit train/test splits (80/20)."""
    all_samples = []
    for label_idx, class_name in enumerate(CLASSES):
        img_dir = os.path.join(FISH_DIR, class_name, class_name)
        gt_dir  = os.path.join(FISH_DIR, class_name, class_name + ' GT')
        if not os.path.isdir(img_dir):
            continue
        filenames = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        for fname in filenames:
            img_path = os.path.join(img_dir, fname)
            gt_path  = os.path.join(gt_dir, fname)
            all_samples.append((img_path, gt_path, label_idx))

    # Réduction optionnelle du dataset
    rng = random.Random(SEED)
    rng.shuffle(all_samples)
    if DATASET_FRAC < 1.0:
        all_samples = all_samples[:max(1, int(len(all_samples) * DATASET_FRAC))]
    split = int(len(all_samples) * TRAIN_RATIO)
    return all_samples[:split], all_samples[split:]


def get_dataloaders():
    train_samples, test_samples = load_fish_splits()

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

    train_ds = FishDataset(train_samples, transform=train_transform)
    test_ds  = FishDataset(test_samples,  transform=test_transform)

    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, generator=gen)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    return train_loader, test_loader, test_ds


# ---------------------------------------------------------------------------
# GradCAM différentiable
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


def compute_gradcam_numpy(model, img_tensor, device=DEVICE):
    """GradCAM pour visualisation (non différentiable)."""
    model.eval()
    grads_list = []
    acts_list  = []

    h1 = model.gradcam_layer.register_full_backward_hook(lambda m, gi, go: grads_list.append(go[0]))
    h2 = model.gradcam_layer.register_forward_hook(lambda m, inp, out: acts_list.append(out))

    x      = img_tensor.unsqueeze(0).to(device)
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
    cam     = (cam - mn) / (mx - mn + 1e-8)
    return cam, pred, logits.detach()


# ---------------------------------------------------------------------------
# Stratégies d'entraînement
# ---------------------------------------------------------------------------
def train_epoch_normal(model, loader, optimizer):
    model.train()
    total_loss = total_correct = total = 0
    for images, labels, _, _ in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item()
        total_correct += (logits.argmax(1) == labels).sum().item()
        total         += labels.size(0)
    return total_loss / len(loader), 100. * total_correct / total


def localisation_loss(cam_batch, heatmaps):
    B = cam_batch.size(0)
    cam_flat = cam_batch.view(B, -1)
    sup_flat = heatmaps.view(B, -1).to(cam_flat.device)
    return 1.0 - F.cosine_similarity(cam_flat, sup_flat, dim=1).mean()


def train_epoch_gradcam(model, loader, optimizer, lambda_gc=LAMBDA_GC):
    model.train()
    total_loss = total_ce = total_correct = total = 0
    for images, labels, heatmaps, _ in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        heatmaps       = heatmaps.to(DEVICE)
        optimizer.zero_grad()
        cam, logits = compute_gradcam_differentiable(model, images, labels)
        ce   = F.cross_entropy(logits, labels)
        loc  = localisation_loss(cam, heatmaps)
        loss = ce + lambda_gc * loc
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
    per_class_correct = np.zeros(N_CLASSES)
    per_class_total   = np.zeros(N_CLASSES)
    with torch.no_grad():
        for images, labels, _, _ in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss   = F.cross_entropy(logits, labels)
            preds  = logits.argmax(1)
            total_loss    += loss.item()
            total_correct += (preds == labels).sum().item()
            total         += labels.size(0)
            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                per_class_total[l]   += 1
                per_class_correct[l] += (p == l)
    acc = 100. * total_correct / total
    per_class_acc = np.where(per_class_total > 0,
                             100. * per_class_correct / per_class_total, 0.)
    return total_loss / len(loader), acc, per_class_acc


def compute_mean_iou(model, test_ds, n_samples=200):
    """Calcule l'IoU moyen entre GradCAM et masque GT."""
    model.eval()
    indices = list(range(min(n_samples, len(test_ds))))
    random.shuffle(indices)
    ious = []
    for idx in indices[:n_samples]:
        img_tensor, label, heatmap, _ = test_ds[idx]
        cam, pred, _ = compute_gradcam_numpy(model, img_tensor, device=DEVICE)
        gt = heatmap.squeeze().numpy()
        gt_resized = cv2.resize(gt.astype(np.float32), (IMG_SIZE, IMG_SIZE))
        pred_bin = (cam >= 0.5).astype(float)
        gt_bin   = (gt_resized >= 0.5).astype(float)
        inter = (pred_bin * gt_bin).sum()
        union = np.maximum(pred_bin, gt_bin).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# Entraînement complet d'un modèle
# ---------------------------------------------------------------------------
def train_model(name, mode, train_loader, test_loader, epochs, verbose=True):
    """mode: 'normal' ou 'gradcam'"""
    set_seed(SEED)
    model = SimpleCNN_RGB(n_classes=N_CLASSES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {'train_loss': [], 'train_acc': [], 'train_ce': [],
               'test_loss': [], 'test_acc': []}
    t0 = time.time()

    for epoch in range(epochs):
        if mode == 'normal':
            tr_loss, tr_acc = train_epoch_normal(model, train_loader, optimizer)
            tr_ce = tr_loss
        else:
            tr_loss, tr_ce, tr_acc = train_epoch_gradcam(model, train_loader, optimizer)

        te_loss, te_acc, _ = eval_epoch(model, test_loader)
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

    duration = time.time() - t0
    return model, history, duration


# ---------------------------------------------------------------------------
# Génération des exemples GradCAM
# ---------------------------------------------------------------------------
DENORM_MEAN = np.array([0.485, 0.456, 0.406])
DENORM_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * DENORM_STD + DENORM_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def generate_examples(models_dict, test_ds, results_dir, n=60):
    examples_dir = os.path.join(results_dir, 'gradcam_examples')
    os.makedirs(examples_dir, exist_ok=True)

    indices = list(range(len(test_ds)))
    random.seed(SEED)
    random.shuffle(indices)
    indices = indices[:n]

    examples = []
    for i, idx in enumerate(indices):
        img_tensor, label, heatmap, img_path = test_ds[idx]

        img_bgr   = denormalize(img_tensor)
        orig_file = f"sample_{i:03d}_orig.jpg"
        cv2.imwrite(os.path.join(examples_dir, orig_file), img_bgr)

        seg_np      = heatmap.squeeze().numpy().astype(np.float32)
        seg_resized = cv2.resize(seg_np, (IMG_SIZE, IMG_SIZE))
        seg_colored = cv2.applyColorMap((seg_resized * 255).astype(np.uint8), cv2.COLORMAP_BONE)
        seg_file    = f"sample_{i:03d}_seg.jpg"
        cv2.imwrite(os.path.join(examples_dir, seg_file), seg_colored)

        sample = {
            'id': i, 'true_class': CLASSES[label], 'true_label': label,
            'orig_img': orig_file, 'seg_img': seg_file,
            'models': {}
        }

        for model_name, model in models_dict.items():
            cam, pred, logits = compute_gradcam_numpy(model, img_tensor, device=DEVICE)
            probs   = torch.softmax(logits, dim=1)[0]
            conf    = probs[pred].item()
            correct = (pred == label)

            heatmap_col = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay     = (heatmap_col.astype(np.float32) / 255. * 0.5 +
                           img_bgr.astype(np.float32) / 255. * 0.5)
            overlay     = (overlay * 255).astype(np.uint8)

            cam_file = f"sample_{i:03d}_{model_name}_cam.jpg"
            cv2.imwrite(os.path.join(examples_dir, cam_file), overlay)

            sample['models'][model_name] = {
                'cam_img':    cam_file,
                'pred_class': CLASSES[pred],
                'pred_label': pred,
                'confidence': f"{conf:.1%}",
                'correct':    bool(correct),
            }

        examples.append(sample)
    return examples


# ---------------------------------------------------------------------------
# Sauvegarde data.js
# ---------------------------------------------------------------------------
def save_data_js(histories, durations, per_class_accs, examples,
                 iou_scores, epochs, results_dir, n_train=0, n_test=0):
    data = {
        'epochs':         list(range(1, epochs + 1)),
        'histories':      histories,
        'durations':      durations,
        'per_class_accs': {k: v.tolist() for k, v in per_class_accs.items()},
        'iou_scores':     iou_scores,
        'classes':        CLASSES,
        'n_classes':      N_CLASSES,
        'n_train':        n_train,
        'n_test':         n_test,
    }
    js_path = os.path.join(results_dir, 'data.js')
    with open(js_path, 'w', encoding='utf-8') as f:
        f.write('window.FISH_DATA = ')
        json.dump(data, f, indent=2)
        f.write(';\n')
        f.write('window.FISH_EXAMPLES = ')
        json.dump(examples, f, indent=2)
        f.write(';\n')
    print(f"  → data.js : {js_path}")


# ---------------------------------------------------------------------------
# Génération des courbes PNG
# ---------------------------------------------------------------------------
def plot_curves(histories, epochs, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fish Dataset : Normal vs GradCAM-Guided', fontsize=13, fontweight='bold')
    epochs_range = range(1, epochs + 1)
    colors = {
        'Normal': '#2196F3', 'GradCAM (λ=0.1)': '#FF9800',
    }
    styles = {
        'Normal': '-', 'GradCAM (λ=0.1)': '--',
    }

    for name, h in histories.items():
        c = colors.get(name, '#aaa')
        s = styles.get(name, '-')
        axes[0, 0].plot(epochs_range, h['test_loss'],  color=c, ls=s, lw=2, label=name)
        axes[0, 1].plot(epochs_range, h['train_ce'],   color=c, ls=s, lw=2, label=name)
        axes[1, 0].plot(epochs_range, h['test_acc'],   color=c, ls=s, lw=2, label=name)
        axes[1, 1].plot(epochs_range, h['train_acc'],  color=c, ls=s, lw=2, label=name)

    for ax, title, ylabel in [
        (axes[0, 0], 'Test Loss',         'Loss'),
        (axes[0, 1], 'Train Loss (CE)',    'Loss'),
        (axes[1, 0], 'Test Accuracy (%)',  'Accuracy (%)'),
        (axes[1, 1], 'Train Accuracy (%)', 'Accuracy (%)'),
    ]:
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Courbes PNG : {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fish Dataset : SimpleCNN · Normal vs GradCAM-Guided')
    parser.add_argument('--clean',  action='store_true', help='Supprime les anciens résultats')
    args = parser.parse_args()

    print('=' * 70)
    print('  Fish Dataset : SimpleCNN · Normal vs GradCAM-Guided')
    print('=' * 70)
    print(f'  Device  : {DEVICE}')
    print(f'  Epochs  : {EPOCHS}')
    print(f'  Classes : {N_CLASSES}  ({", ".join(CLASSES)})')
    print(f'  Lambda  : {LAMBDA_GC}')
    print()

    if args.clean and os.path.exists(RESULTS_DIR):
        for item in os.listdir(RESULTS_DIR):
            p = os.path.join(RESULTS_DIR, item)
            if item.endswith('.html'): continue
            if os.path.isfile(p): os.remove(p)
            elif os.path.isdir(p): shutil.rmtree(p)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Charger cache
    metrics_path = os.path.join(RESULTS_DIR, 'metrics.json')
    histories, durations, per_class_accs, iou_scores = {}, {}, {}, {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                saved = json.load(f)
            histories      = saved.get('histories', {})
            durations      = saved.get('durations', {})
            iou_scores     = saved.get('iou_scores', {})
            per_class_accs = {k: np.array(v) for k, v in saved.get('per_class_accs', {}).items()}
            print(f'  [i] {len(histories)} modèles chargés depuis metrics.json')
        except json.JSONDecodeError:
            print('  [!] metrics.json corrompu, ignoré')

    print('▶ Chargement du dataset...')
    set_seed(SEED)
    train_loader, test_loader, test_ds = get_dataloaders()
    print(f'  Train : {len(train_loader.dataset)} images')
    print(f'  Test  : {len(test_loader.dataset)} images')
    print()

    models_out = {}
    configs = [
        ('Normal',          'normal'),
        ('GradCAM (λ=0.1)', 'gradcam'),
    ]

    for name, mode in configs:
        if name in histories:
            print(f'  [⏭]  {name} : déjà entraîné (--clean pour réentraîner)')
            continue
        print(f'▶ Entraînement {name}...')
        model, h, d = train_model(name, mode, train_loader, test_loader, EPOCHS)

        _, _, pca = eval_epoch(model, test_loader)

        histories[name]      = h
        durations[name]      = d
        per_class_accs[name] = pca
        models_out[name]     = model

        with open(metrics_path, 'w') as f:
            json.dump({
                'histories': histories, 'durations': durations,
                'iou_scores': iou_scores,
                'per_class_accs': {k: v.tolist() for k, v in per_class_accs.items()}
            }, f, indent=2)
        print(f'  ✓ {d:.0f}s — test_acc={h["test_acc"][-1]:.1f}%\n')

    # IoU GradCAM ↔ Segmentation
    for name, mode in configs:
        if name not in iou_scores:
            if name in models_out:
                print(f'▶ Calcul IoU GradCAM ↔ Segmentation pour {name}...')
                iou_scores[name] = compute_mean_iou(models_out[name], test_ds, n_samples=300)
                print(f'  IoU {name} = {iou_scores[name]:.3f}')
                with open(metrics_path, 'w') as f:
                    json.dump({
                        'histories': histories, 'durations': durations,
                        'iou_scores': iou_scores,
                        'per_class_accs': {k: v.tolist() for k, v in per_class_accs.items()}
                    }, f, indent=2)

    # Exemples GradCAM
    print('▶ Génération des exemples GradCAM...')
    if not models_out:
        print('  [!] Aucun modèle entraîné dans cette session — galerie vide.')
        examples = []
    else:
        examples = generate_examples(models_out, test_ds, RESULTS_DIR, n=60)

    # Courbes PNG
    plot_curves(histories, EPOCHS, os.path.join(RESULTS_DIR, 'curves.png'))

    # data.js
    n_train = len(train_loader.dataset)
    n_test  = len(test_loader.dataset)
    save_data_js(histories, durations, per_class_accs, examples,
                 iou_scores, EPOCHS, RESULTS_DIR,
                 n_train=n_train, n_test=n_test)
    print(f"  → HTML   : {os.path.join(RESULTS_DIR, 'fish_results.html')}  (fichier statique, toujours à jour)")

    print()
    print('=' * 70)
    print('  RÉSULTATS FINAUX')
    print('=' * 70)
    for name in [n for n, _ in configs]:
        if name not in histories: continue
        h   = histories[name]
        iou = iou_scores.get(name, 0)
        print(f'  {name:20s}  train_acc={h["train_acc"][-1]:.1f}%  '
              f'test_acc={h["test_acc"][-1]:.1f}%  '
              f'IoU={iou:.3f}  {durations[name]:.0f}s')

    print()
    print('  Double-cliquer sur : real_datasets/poissons/results/fish_results.html')
