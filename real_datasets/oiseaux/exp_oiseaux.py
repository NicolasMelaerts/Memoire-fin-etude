"""
exp_oiseaux.py - Expérience CUB-200-2011 : Normal vs Guided GradCAM

Dataset réel : Caltech-UCSD Birds 200 (11 788 images, 200 espèces)
Supervision  : masques de segmentation binaires (oiseau vs fond)

Hypothèse : en forçant le modèle à regarder l'oiseau (et non le fond),
            la guidance GradCAM améliore l'accuracy et la qualité des explications.

Usage :
    python3 exp_oiseaux.py [--clean]

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
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CUB_DIR    = os.path.join(BASE_DIR, 'CUB_200_2011')
SEG_DIR    = os.path.join(BASE_DIR, 'segmentations')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# ---------------------------------------------------------------------------
# Hyperparamètres
# ---------------------------------------------------------------------------
IMG_SIZE       = 128
N_CLASSES      = 5
BATCH_SIZE     = 32
EPOCHS         = 15
LR             = 1e-3
SEED           = 42
LAMBDA_GC_LIST = [0.1, 0.3, 0.5]     # 3 valeurs testées par architecture
LAMBDA_GC      = LAMBDA_GC_LIST[1]   # défaut pour les helpers
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
# Modèle - SimpleCNN adapté RGB / N classes
# (même architecture que shared/model.py, 1→3 canaux, 2→200 sorties)
# ---------------------------------------------------------------------------
class SimpleCNN_RGB(nn.Module):
    """SimpleCNN adapté pour images RGB 128×128 et N classes."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.conv1        = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # 3 canaux RGB
        self.conv2        = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3        = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool         = nn.MaxPool2d(2, 2)
        self.fc1          = nn.Linear(64 * 16 * 16, 128)
        self.dropout      = nn.Dropout(0.5)
        self.fc2          = nn.Linear(128, n_classes)                     # N sorties
        self.gradcam_layer = self.conv3                                    # couche cible GradCAM

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 32, 64, 64]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 64, 32, 32]
        x = self.pool(F.relu(self.conv3(x)))   # [B, 64, 16, 16]
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def forward_features(self, x):
        """Retourne les feature maps de conv3 AVANT pooling (pour GradCAM diff.)."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return F.relu(self.conv3(x))            # [B, 64, 32, 32]

    def forward_from_features(self, feat):
        x = self.pool(feat)                     # [B, 64, 16, 16]
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Modèle - ResNet18 pré-entraîné (Transfer Learning)
# ---------------------------------------------------------------------------
class ResNet18_CUB(nn.Module):
    """ResNet18 pré-entraîné sur ImageNet, fine-tuné pour N classes."""

    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        from torchvision import models
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet = models.resnet18(weights=weights)

        # Geler les premières couches (conv1, bn1, layer1, layer2)
        for name, param in self.resnet.named_parameters():
            if not name.startswith('layer3') and not name.startswith('layer4') and not name.startswith('fc'):
                param.requires_grad = False

        # Remplacer la couche FC finale
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, n_classes)
        )

        # GradCAM cible la dernière couche convolutive (layer4)
        self.gradcam_layer = self.resnet.layer4[-1].conv2

    def forward(self, x):
        return self.resnet(x)

    def forward_features(self, x):
        """Retourne les feature maps de layer4 (avant avgpool)."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x  # [B, 512, H, W]

    def forward_from_features(self, feat):
        """Classification depuis les feature maps de layer4."""
        x = self.resnet.avgpool(feat)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x


# ---------------------------------------------------------------------------
# Dataset CUB-200-2011
# ---------------------------------------------------------------------------
class CUBDataset(Dataset):
    """Charge images CUB-200 + masques de segmentation comme heatmaps."""

    def __init__(self, image_ids, img_paths, class_ids, transform=None):
        self.image_ids = image_ids
        self.img_paths = img_paths       # relatif à CUB_DIR/images/
        self.class_ids = class_ids       # 0-indexed
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
        return len(self.image_ids)

    def __getitem__(self, idx):
        rel_path  = self.img_paths[idx]
        img_path  = os.path.join(CUB_DIR, 'images', rel_path)
        image     = Image.open(img_path).convert('RGB')
        image     = self.transform(image)

        label = self.class_ids[idx]

        # Masque de segmentation → heatmap de supervision [0, 1]
        seg_rel  = rel_path.replace('.jpg', '.png')
        seg_path = os.path.join(SEG_DIR, seg_rel)
        if os.path.exists(seg_path):
            seg = Image.open(seg_path).convert('L')
            heatmap = self.seg_transform(seg)        # [1, H, W], 0 ou 1
            heatmap = (heatmap > 0.5).float()
        else:
            heatmap = torch.ones(1, IMG_SIZE, IMG_SIZE) / (IMG_SIZE * IMG_SIZE)

        return image, label, heatmap, rel_path


def load_cub_splits():
    """Lit les fichiers CUB et retourne train/test splits."""
    # images.txt : id → relative_path
    with open(os.path.join(CUB_DIR, 'images.txt')) as f:
        img_paths = {int(l.split()[0]): l.split()[1] for l in f}
    # image_class_labels.txt : id → class_id (1-indexed)
    with open(os.path.join(CUB_DIR, 'image_class_labels.txt')) as f:
        img_labels = {int(l.split()[0]): int(l.split()[1]) - 1 for l in f}  # 0-indexed
    # train_test_split.txt : id → 1=train, 0=test
    with open(os.path.join(CUB_DIR, 'train_test_split.txt')) as f:
        splits = {int(l.split()[0]): int(l.split()[1]) for l in f}
    # classes.txt
    with open(os.path.join(CUB_DIR, 'classes.txt')) as f:
        classes = {int(l.split()[0]) - 1: l.split()[1] for l in f}  # 0-indexed

    train_ids = [i for i, s in splits.items() if s == 1 and img_labels[i] < N_CLASSES]
    test_ids  = [i for i, s in splits.items() if s == 0 and img_labels[i] < N_CLASSES]

    return (
        train_ids, [img_paths[i] for i in train_ids], [img_labels[i] for i in train_ids],
        test_ids,  [img_paths[i] for i in test_ids],  [img_labels[i] for i in test_ids],
        classes
    )


def get_dataloaders():
    train_ids, train_paths, train_labels, test_ids, test_paths, test_labels, classes = load_cub_splits()

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

    train_ds = CUBDataset(train_ids, train_paths, train_labels, transform=train_transform)
    test_ds  = CUBDataset(test_ids,  test_paths,  test_labels,  transform=test_transform)

    gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False, generator=gen)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    return train_loader, test_loader, test_ds, classes


# ---------------------------------------------------------------------------
# GradCAM différentiable (même logique que shared/strategies.py)
# ---------------------------------------------------------------------------
def compute_gradcam_differentiable(model, image_batch, target_classes):
    B    = image_batch.size(0)
    feat = model.forward_features(image_batch)     # [B, 64, 32, 32]
    feat.retain_grad()
    logits = model.forward_from_features(feat)     # [B, N_CLASSES]
    scores = logits[torch.arange(B), target_classes]
    grads  = torch.autograd.grad(
        scores.sum(), feat, create_graph=True, retain_graph=True
    )[0]                                           # [B, 64, 32, 32]
    weights = grads.mean(dim=(2, 3), keepdim=True) # [B, 64, 1, 1]
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
    """Calcule l'IoU moyen entre GradCAM et masque de segmentation."""
    model.eval()
    indices = list(range(min(n_samples, len(test_ds))))
    random.shuffle(indices)
    ious = []
    for idx in indices[:n_samples]:
        img_tensor, label, heatmap, _ = test_ds[idx]
        cam, pred, _ = compute_gradcam_numpy(model, img_tensor, device=DEVICE)
        gt = heatmap.squeeze().numpy()
        gt_resized = cv2.resize(gt, (IMG_SIZE, IMG_SIZE))
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
def train_model(name, mode, train_loader, test_loader, epochs, verbose=True,
                model_class=None, lambda_gc=LAMBDA_GC):
    """mode: 'normal' ou 'gradcam'. model_class: classe du modèle (défaut: SimpleCNN_RGB).
    lambda_gc: coefficient Guided GradCAM (ignoré si mode='normal')."""
    set_seed(SEED)
    cls = model_class or SimpleCNN_RGB
    model = cls(n_classes=N_CLASSES).to(DEVICE)

    # Learning rate plus bas pour le fine-tuning ResNet
    lr = LR * 0.1 if cls is ResNet18_CUB else LR
    # Seuls les paramètres non gelés sont optimisés
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {'train_loss': [], 'train_acc': [], 'train_ce': [],
               'test_loss': [], 'test_acc': []}
    t0 = time.time()

    for epoch in range(epochs):
        if mode == 'normal':
            tr_loss, tr_acc = train_epoch_normal(model, train_loader, optimizer)
            tr_ce = tr_loss
        else:
            tr_loss, tr_ce, tr_acc = train_epoch_gradcam(model, train_loader, optimizer,
                                                        lambda_gc=lambda_gc)

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
    """Convertit un tensor normalisé ImageNet en image uint8 BGR."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * DENORM_STD + DENORM_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def generate_examples(models_dict, test_ds, classes, results_dir, n=60):
    examples_dir = os.path.join(results_dir, 'gradcam_examples')
    os.makedirs(examples_dir, exist_ok=True)

    indices = list(range(len(test_ds)))
    random.seed(SEED)
    random.shuffle(indices)
    indices = indices[:n]

    examples = []
    for i, idx in enumerate(indices):
        img_tensor, label, heatmap, rel_path = test_ds[idx]

        # Image originale dénormalisée
        img_bgr  = denormalize(img_tensor)
        orig_file = f"sample_{i:03d}_orig.jpg"
        cv2.imwrite(os.path.join(examples_dir, orig_file), img_bgr)

        # Masque de segmentation
        seg_np    = heatmap.squeeze().numpy()
        seg_resized = cv2.resize(seg_np, (IMG_SIZE, IMG_SIZE))
        seg_colored = cv2.applyColorMap((seg_resized * 255).astype(np.uint8), cv2.COLORMAP_BONE)
        seg_file  = f"sample_{i:03d}_seg.jpg"
        cv2.imwrite(os.path.join(examples_dir, seg_file), seg_colored)

        sample = {
            'id': i, 'true_class': classes[label], 'true_label': label,
            'orig_img': orig_file, 'seg_img': seg_file,
            'models': {}
        }

        for model_name, model in models_dict.items():
            cam, pred, logits = compute_gradcam_numpy(model, img_tensor, device=DEVICE)
            probs    = torch.softmax(logits, dim=1)[0]
            conf     = probs[pred].item()
            correct  = (pred == label)

            heatmap_col = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay     = (heatmap_col.astype(np.float32) / 255. * 0.5 +
                           img_bgr.astype(np.float32) / 255. * 0.5)
            overlay     = (overlay * 255).astype(np.uint8)

            cam_file = f"sample_{i:03d}_{model_name}_cam.jpg"
            cv2.imwrite(os.path.join(examples_dir, cam_file), overlay)

            sample['models'][model_name] = {
                'cam_img': cam_file,
                'pred_class': classes[pred],
                'pred_label': pred,
                'confidence': f"{conf:.1%}",
                'correct': bool(correct),
            }

        examples.append(sample)
    return examples


# ---------------------------------------------------------------------------
# Sauvegarde data.js
# ---------------------------------------------------------------------------
def save_data_js(histories, durations, per_class_accs, examples,
                 iou_scores, classes_list, epochs, results_dir, n_train=0, n_test=0):
    data = {
        'epochs':         list(range(1, epochs + 1)),
        'histories':      histories,
        'durations':      durations,
        'per_class_accs': {k: v.tolist() for k, v in per_class_accs.items()},
        'iou_scores':     iou_scores,
        'classes':        classes_list,
        'n_classes':      N_CLASSES,
        'n_train':        n_train,
        'n_test':         n_test,
    }
    js_path = os.path.join(results_dir, 'data.js')
    with open(js_path, 'w', encoding='utf-8') as f:
        f.write('window.CUB_DATA = ')
        json.dump(data, f, indent=2)
        f.write(';\n')
        f.write('window.CUB_EXAMPLES = ')
        json.dump(examples, f, indent=2)
        f.write(';\n')
    print(f"  → data.js : {js_path}")


# ---------------------------------------------------------------------------
# Génération des courbes PNG
# ---------------------------------------------------------------------------
def plot_curves(histories, epochs, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CUB-200 : SimpleCNN & ResNet18 · Normal vs Guided GradCAM', fontsize=13, fontweight='bold')
    epochs_range = range(1, epochs + 1)
    # Palette : bleus pour SimpleCNN, verts pour ResNet18.
    # Pour GradCAM, dégradés selon λ (clair = 0.1, moyen = 0.3, foncé = 0.5).
    colors = {
        'Normal':                    '#2196F3',
        'GradCAM (λ=0.1)':           '#FFCC80',
        'GradCAM (λ=0.3)':           '#FF9800',
        'GradCAM (λ=0.5)':           '#E65100',
        'Normal-ResNet18':           '#4CAF50',
        'GradCAM-ResNet18 (λ=0.1)':  '#F8BBD0',
        'GradCAM-ResNet18 (λ=0.3)':  '#E91E63',
        'GradCAM-ResNet18 (λ=0.5)':  '#AD1457',
    }
    styles = {name: ('-' if 'Normal' in name else '--') for name in colors}

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
    parser = argparse.ArgumentParser(description='CUB-200 : SimpleCNN & ResNet18 · Normal vs Guided GradCAM')
    parser.add_argument('--n-classes', type=int, default=N_CLASSES, help=f'Nombre de classes CUB à utiliser (défaut: {N_CLASSES})')
    parser.add_argument('--clean',  action='store_true', help='Supprime les anciens résultats')
    args = parser.parse_args()

    N_CLASSES = args.n_classes

    print('=' * 70)
    print('  CUB-200-2011 : SimpleCNN & ResNet18 · Normal vs Guided GradCAM')
    print('=' * 70)
    print(f'  Device  : {DEVICE}')
    print(f'  Epochs  : {EPOCHS}')
    print(f'  Classes : {N_CLASSES}')
    print(f'  Lambdas : {LAMBDA_GC_LIST}')
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
    train_loader, test_loader, test_ds, classes = get_dataloaders()
    classes_list = [classes[i] for i in range(N_CLASSES)]
    print(f'  Train : {len(train_loader.dataset)} images')
    print(f'  Test  : {len(test_loader.dataset)} images')
    print()

    models_out = {}
    # Tuple : (nom, mode, classe_modèle, lambda_gc)
    # Pour les configurations Normal, lambda_gc est ignoré (None).
    configs = [
        ('Normal',                      'normal',  SimpleCNN_RGB, None),
        ('GradCAM (λ=0.1)',             'gradcam', SimpleCNN_RGB, 0.1),
        ('GradCAM (λ=0.3)',             'gradcam', SimpleCNN_RGB, 0.3),
        ('GradCAM (λ=0.5)',             'gradcam', SimpleCNN_RGB, 0.5),
        ('Normal-ResNet18',             'normal',  ResNet18_CUB,  None),
        ('GradCAM-ResNet18 (λ=0.1)',    'gradcam', ResNet18_CUB,  0.1),
        ('GradCAM-ResNet18 (λ=0.3)',    'gradcam', ResNet18_CUB,  0.3),
        ('GradCAM-ResNet18 (λ=0.5)',    'gradcam', ResNet18_CUB,  0.5),
    ]

    for name, mode, model_cls, lam in configs:
        if name in histories:
            print(f'  [⏭]  {name} : déjà entraîné (--clean pour réentraîner)')
            continue
        print(f'▶ Entraînement {name}...')
        model, h, d = train_model(name, mode, train_loader, test_loader, EPOCHS,
                                  model_class=model_cls,
                                  lambda_gc=(lam if lam is not None else LAMBDA_GC))

        # Métriques per-class
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
        print(f'  ✓ {d:.0f}s - test_acc={h["test_acc"][-1]:.1f}%\n')

    # IoU GradCAM ↔ Segmentation
    for name, mode, _, _ in configs:
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
        print('  [!] Aucun modèle entraîné dans cette session - galerie vide.')
        examples = []
    else:
        examples = generate_examples(models_out, test_ds, classes, RESULTS_DIR, n=60)

    # Courbes PNG
    plot_curves(histories, EPOCHS, os.path.join(RESULTS_DIR, 'curves.png'))

    # data.js + HTML
    n_train = len(train_loader.dataset)
    n_test  = len(test_loader.dataset)
    save_data_js(histories, durations, per_class_accs, examples,
                 iou_scores, classes_list, EPOCHS, RESULTS_DIR,
                 n_train=n_train, n_test=n_test)
    print(f"  → HTML   : {os.path.join(RESULTS_DIR, 'cub_results.html')}  (fichier statique, toujours à jour)")

    print()
    print('=' * 70)
    print('  RÉSULTATS FINAUX')
    print('=' * 70)
    for name in [n for n, _, _, _ in configs]:
        if name not in histories: continue
        h   = histories[name]
        iou = iou_scores.get(name, 0)
        print(f'  {name:20s}  train_acc={h["train_acc"][-1]:.1f}%  '
              f'test_acc={h["test_acc"][-1]:.1f}%  '
              f'IoU={iou:.3f}  {durations[name]:.0f}s')

    print()
    print('  Double-cliquer sur : real_datasets/oiseaux/results/cub_results.html')
