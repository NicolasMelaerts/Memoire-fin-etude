"""
utils.py — Utilitaires partagés entre les expériences.

Contient les fonctions communes pour :
  - Reproductibilité (set_seed)
  - Génération des exemples GradCAM
  - Sauvegarde des données pour HTML (data.js)
  - Génération des courbes matplotlib
"""
import os
import json
import random
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
def set_seed(seed):
    """Fixe les seeds pour la reproductibilité COMPLÈTE.

    Cela garantit que tous les résultats (split, shuffle, augmentation)
    sont identiques entre les exécutions et entre les expériences.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # IMPORTANT : Force le déterminisme complet dans PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Pour Python 3.7+ : garantit l'ordre des dictionnaires
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------------------------------------------------------------------------
# Génération des exemples GradCAM
# ---------------------------------------------------------------------------
def generate_gradcam_examples(models_dict, test_ds, results_dir, seed,
                              compute_gradcam_fn, device, n_samples=10,
                              extract_case_fn=None):
    """
    Génère des exemples GradCAM pour chaque modèle sur les mêmes images.

    Args:
        models_dict:       dict {model_name: model}
        test_ds:           dataset de test
        results_dir:       dossier où sauvegarder (ex: 'results/')
        seed:              seed pour la reproductibilité
        compute_gradcam_fn: fonction pour calculer GradCAM (ex: compute_gradcam_numpy)
        device:            device PyTorch ('cuda' ou 'cpu')
        n_samples:         nombre d'exemples à générer
        extract_case_fn:   fonction optionnelle pour extraire le 'case_type' depuis le dataset

    Returns:
        examples : liste de dicts avec les infos pour chaque sample
    """
    examples_dir = os.path.join(results_dir, 'gradcam_comparison')
    os.makedirs(examples_dir, exist_ok=True)

    indices = list(range(min(n_samples, len(test_ds))))
    random.seed(seed)
    random.shuffle(indices)
    indices = indices[:n_samples]

    classes = ['Inside', 'Outside']
    examples = []

    for i, idx in enumerate(indices):
        img_tensor, label, _, img_name = test_ds[idx]
        true_label = classes[label]

        # Image originale → PNG
        img_np  = (img_tensor.cpu().numpy().squeeze() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        orig_file = f"sample_{i:02d}_original.png"
        cv2.imwrite(os.path.join(examples_dir, orig_file), img_bgr)

        sample_info = {
            'id':         i,
            'true_label': true_label,
            'img_name':   img_name if isinstance(img_name, str) else img_name[0],
            'orig_img':   orig_file,
            'models':     {},
        }

        # Extraire le case_type si une fonction est fournie
        if extract_case_fn is not None:
            try:
                sample_info['case_type'] = extract_case_fn(test_ds, idx)
            except Exception:
                sample_info['case_type'] = "unknown"

        # Générer GradCAM pour chaque modèle
        for model_name, model in models_dict.items():
            model.eval()
            cam, pred_idx, logits = compute_gradcam_fn(model, img_tensor, device=device)

            # Superposition GradCAM sur l'image
            heatmap   = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            overlay   = heatmap.astype(np.float32) / 255.0 + img_bgr.astype(np.float32) / 255.0
            overlay   = (overlay / overlay.max() * 255).astype(np.uint8)

            cam_file = f"sample_{i:02d}_{model_name}_cam.png"
            cv2.imwrite(os.path.join(examples_dir, cam_file), overlay)

            probs      = torch.softmax(logits, dim=1)[0]
            pred_label = classes[pred_idx]
            conf       = probs[pred_idx].item()

            sample_info['models'][model_name] = {
                'cam_img':    cam_file,
                'pred_label': pred_label,
                'confidence': f"{conf:.1%}",
                'correct':    (pred_label == true_label),
            }

        examples.append(sample_info)

    return examples


# ---------------------------------------------------------------------------
# Sauvegarde des données pour index.html
# ---------------------------------------------------------------------------
def save_data_js(histories, durations, examples, results_dir, epochs,
                 get_color_fn=None):
    """
    Écrit data.js dans results_dir.
    Ce fichier définit les variables window chargées par index.html.

    Args:
        histories:     dict {model_name: {metric: [values]}}
        durations:     dict {model_name: duration_seconds}
        examples:      liste de dicts (résultats de generate_gradcam_examples)
        results_dir:   dossier de sortie
        epochs:        nombre d'époques
        get_color_fn:  fonction optionnelle (name, idx) -> (color, style)
                       Si None, utilise une palette par défaut
    """
    # Structure enrichie pour ChartJS
    epochs_js = list(range(1, epochs + 1))
    chart_data = {
        'epochs': epochs_js,
        'datasets': {}
    }

    # Palette par défaut si aucune fonction fournie
    default_palette = [
        '#2196F3', '#FF9800', '#FF5722', '#F44336',
        '#9C27B0', '#4CAF50', '#00BCD4', '#E91E63',
    ]

    for i, (name, hist) in enumerate(histories.items()):
        if get_color_fn is not None:
            color, _ = get_color_fn(name, i)
        else:
            color = default_palette[i % len(default_palette)]

        chart_data['datasets'][name] = {
            'test_loss':  hist.get('test_loss', hist.get('val_loss', [])),  # Support nouveau et ancien format
            'test_acc':   hist.get('test_acc', hist.get('val_acc', [])),    # Support nouveau et ancien format
            'train_loss': hist['train_loss'],
            'train_ce':   hist.get('train_ce', hist['train_loss']),
            'train_acc':  hist['train_acc'],
            'color':      color,
            # Rétrocompatibilité avec les anciens HTML qui utilisent val_*
            'val_loss':   hist.get('test_loss', hist.get('val_loss', [])),
            'val_acc':    hist.get('test_acc', hist.get('val_acc', [])),
        }

    data_to_save = {
        'histories':  histories,
        'durations':  durations,
        'chart_data': chart_data
    }

    # Variable window selon le contexte (globalMetrics ou RESULTS_DATA)
    js_path = os.path.join(results_dir, 'data.js')
    with open(js_path, 'w', encoding='utf-8') as f:
        # Détection automatique : si 'case_type' présent → experiment 1
        has_case_type = any('case_type' in ex for ex in examples)

        if has_case_type:
            # Format experiment 1
            f.write('window.globalMetrics = ')
            json.dump(data_to_save, f, indent=2)
            f.write(';\n')
            f.write('window.globalExamples = ')
            json.dump(examples, f, indent=2)
            f.write(';\n')
        else:
            # Format experiment 2
            data_to_save['examples'] = examples
            data_to_save['epochs'] = epochs
            # Ajouter les couleurs directement
            colors = {name: chart_data['datasets'][name]['color']
                     for name in histories.keys()}
            data_to_save['colors'] = colors

            f.write('window.RESULTS_DATA = ')
            json.dump(data_to_save, f, indent=2)
            f.write(';\n')

    print(f"  → Données JS  : {js_path}")


# ---------------------------------------------------------------------------
# Génération des courbes matplotlib
# ---------------------------------------------------------------------------
def plot_comparison(histories, output_path, epochs, title="Comparaison",
                   get_color_fn=None, metrics=None):
    """
    Génère les courbes comparatives loss/accuracy et les sauvegarde en PNG.

    Args:
        histories:     dict {model_name: {metric: [values]}}
        output_path:   chemin du fichier PNG de sortie
        epochs:        nombre d'époques
        title:         titre du graphique
        get_color_fn:  fonction optionnelle (name, idx) -> (color, style)
        metrics:       liste de tuples (subplot_pos, metric_key, ylabel, title)
                       Si None, utilise une config par défaut 2x2
    """
    # Palette par défaut
    default_palette = [
        '#2196F3', '#FF9800', '#FF5722', '#F44336',
        '#9C27B0', '#4CAF50', '#00BCD4', '#E91E63',
    ]

    # Configuration par défaut : 2x2 grid
    if metrics is None:
        metrics = [
            ((0, 0), 'val_loss',   'Loss',        'Validation Loss'),
            ((0, 1), 'val_acc',    'Accuracy (%)', 'Validation Accuracy (%)'),
            ((1, 0), 'train_loss', 'Loss',        'Train Loss'),
            ((1, 1), 'train_acc',  'Accuracy (%)', 'Train Accuracy (%)'),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        # Adapter selon le nombre de métriques
        n_metrics = len(metrics)
        if n_metrics == 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        elif n_metrics == 4:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        else:
            fig, axes = plt.subplots(1, n_metrics, figsize=(7*n_metrics, 5))

    fig.suptitle(title, fontsize=13, fontweight='bold')
    epochs_range = range(1, epochs + 1)

    for i, (name, hist) in enumerate(histories.items()):
        if get_color_fn is not None:
            color, style = get_color_fn(name, i)
        else:
            color = default_palette[i % len(default_palette)]
            style = '--' if 'GradCAM' in name or 'Double BP' in name else '-'

        for pos, metric_key, ylabel, subtitle in metrics:
            # Récupérer l'axe
            if isinstance(pos, tuple) and len(pos) == 2:
                ax = axes[pos] if axes.ndim > 1 else axes[pos[1]]
            else:
                ax = axes[pos] if hasattr(axes, '__len__') else axes

            # Tracer
            data = hist.get(metric_key, [])
            if data:
                ax.plot(epochs_range, data, color=color, linestyle=style,
                       linewidth=2, label=name)

    # Configurer chaque subplot
    for pos, metric_key, ylabel, subtitle in metrics:
        if isinstance(pos, tuple) and len(pos) == 2:
            ax = axes[pos] if axes.ndim > 1 else axes[pos[1]]
        else:
            ax = axes[pos] if hasattr(axes, '__len__') else axes

        ax.set_title(subtitle)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Courbes PNG : {output_path}")
