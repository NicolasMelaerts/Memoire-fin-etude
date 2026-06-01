"""
run_multidataset.py - Exécution parallèle des 3 expériences sur N datasets.

- Génère les datasets dans experiences/datasets/, lance les expériences en
  parallèle (3 workers), agrège les résultats et produit un fichier data.js.
- Sortie : experiences/Résultats/multidataset/{runs/, *.json, data.js, index.html}.
- Utilise caffeinate pour empêcher la mise en veille du Mac.

Usage:
    python run_multidataset.py          # Exécution (skip si déjà fait)
    python run_multidataset.py --clean  # Nettoyer et tout réexécuter
"""
import os
import sys
import json
import random
import shutil
import argparse
import subprocess
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset_creator'))
from dataset_generator import DatasetGenerator  # type: ignore  # noqa: E402 (dynamic sys.path)
from config import Config                       # type: ignore  # noqa: E402 (dynamic sys.path)

# ===== CONFIGURATION =====
# Le script vit désormais dans experiences/ ; les datasets et résultats sont
# regroupés sous experiences/datasets/ et experiences/Résultats/multidataset/.
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))     # experiences/
DATASETS_DIR    = os.path.join(BASE_DIR, 'datasets')
EXPERIENCES_DIR = BASE_DIR                                       # alias historique
AGGREGATED_DIR  = os.path.join(BASE_DIR, 'Résultats', 'multidataset')
RESULTS_DIR     = os.path.join(AGGREGATED_DIR, 'runs')

SEEDS = [3, 11, 13, 25, 33, 35, 55, 201, 421, 610, 987, 2001, 2026, 2584, 4181]


# ===== ÉTAPE 1 : GÉNÉRATION DES DATASETS =====

def set_seed(seed):
    """Fixe tous les seeds pour la reproductibilité"""
    random.seed(seed)
    np.random.seed(seed)


def generate_datasets():
    """Génère les datasets avec des seeds différents"""
    print(f"\n[1/4] Génération des {len(SEEDS)} datasets")

    os.makedirs(DATASETS_DIR, exist_ok=True)

    for idx, seed in enumerate(SEEDS):
        output_dir = os.path.join(DATASETS_DIR, f'dataset_seed{seed}')

        if os.path.exists(output_dir):
            print(f"  [{idx+1}/{len(SEEDS)}] seed={seed} ⏭️  skip")
            continue

        print(f"  [{idx+1}/{len(SEEDS)}] seed={seed}...", end=' ', flush=True)

        class CustomConfig:
            NUM_IMAGES = Config.NUM_IMAGES
            IMAGE_SIZE = Config.IMAGE_SIZE
            OUTPUT_DIR = output_dir
            POSITIVE_RATIO = Config.POSITIVE_RATIO
            ARROW_RATIO_POSITIVE = Config.ARROW_RATIO_POSITIVE
            ARROW_RATIO_NEGATIVE = Config.ARROW_RATIO_NEGATIVE
            POSITIVE_WITH_NOISE_RATIO = Config.POSITIVE_WITH_NOISE_RATIO
            MIN_CIRCLE_RADIUS = Config.MIN_CIRCLE_RADIUS
            MAX_CIRCLE_RADIUS = Config.MAX_CIRCLE_RADIUS
            MIN_TRIANGLE_SIDE = Config.MIN_TRIANGLE_SIDE
            MAX_TRIANGLE_SIDE = Config.MAX_TRIANGLE_SIDE
            ARROW_HEAD_SIZE = Config.ARROW_HEAD_SIZE
            ARROW_LINE_WIDTH = Config.ARROW_LINE_WIDTH
            NEGATIVE_CASE_TYPES = Config.NEGATIVE_CASE_TYPES
            MAX_PLACEMENT_ATTEMPTS = Config.MAX_PLACEMENT_ATTEMPTS

        set_seed(seed)
        gen = DatasetGenerator(CustomConfig)
        gen.run()
        print("✓")


# ===== ÉTAPE 2 : EXÉCUTION DES EXPÉRIENCES =====

def run_all_experiments(num_workers=2):
    """Lance les 3 expériences sur chaque dataset en parallèle (pool de workers).

    Exécution en deux phases pour permettre à exp3 de recycler Normal + GradCAM
    λ=0.1 depuis exp1 :
      - Phase 1 : toutes les exp1 (en parallèle).
      - Phase 2 : toutes les exp2 + exp3 (en parallèle) — exp3 retrouve alors
                  le metrics.json d'exp1 dans le même run_X et skip l'entraînement
                  des modèles partagés.
    """
    experiments = [
        ('exp1', os.path.join(EXPERIENCES_DIR, 'run_exp1.py')),
        ('exp2', os.path.join(EXPERIENCES_DIR, 'run_exp2.py')),
        ('exp3', os.path.join(EXPERIENCES_DIR, 'run_exp3.py')),
    ]

    n_cpu = os.cpu_count() or 4
    threads_per_worker = max(1, n_cpu // max(1, num_workers))

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Détermine d'abord ce qui est à faire vs déjà fait, séparé par phase
    phase1_tasks = []  # exp1
    phase2_tasks = []  # exp2 + exp3 (exp3 dépend d'exp1)
    skipped = 0
    for run_idx, seed in enumerate(SEEDS):
        dataset_path = os.path.join(DATASETS_DIR, f'dataset_seed{seed}')
        run_output_dir = os.path.join(RESULTS_DIR, f'run_{run_idx}')

        for exp_name, exp_script in experiments:
            metrics_file = os.path.join(run_output_dir, exp_name, 'metrics.json')
            if os.path.exists(metrics_file):
                print(f"  run_{run_idx} (seed={seed}) | {exp_name} ⏭️  skip")
                skipped += 1
                continue
            task = (run_idx, seed, exp_name, exp_script, dataset_path, run_output_dir)
            if exp_name == 'exp1':
                phase1_tasks.append(task)
            else:
                phase2_tasks.append(task)

    total_todo = len(phase1_tasks) + len(phase2_tasks)
    total = len(SEEDS) * len(experiments)
    print(f"\n[2/4] Exécution des expériences ({len(SEEDS)} datasets × {len(experiments)} exp = {total} tâches, "
          f"{num_workers} workers × {threads_per_worker} threads)")
    print(f"      Phase 1 (exp1) : {len(phase1_tasks)} | Phase 2 (exp2+exp3) : {len(phase2_tasks)} | skip : {skipped}")

    if total_todo == 0:
        print(f"\n  📊 Résumé: 0 à faire, {skipped} skip\n")
        return True

    def run_one(task):
        _, _, _, exp_script, dataset_path, run_output_dir = task
        env = os.environ.copy()
        env['DATASET_PATH'] = dataset_path
        env['CUSTOM_RESULTS_DIR'] = run_output_dir
        env['OMP_NUM_THREADS']        = str(threads_per_worker)
        env['MKL_NUM_THREADS']        = str(threads_per_worker)
        env['OPENBLAS_NUM_THREADS']   = str(threads_per_worker)
        env['VECLIB_MAXIMUM_THREADS'] = str(threads_per_worker)
        env['NUMEXPR_NUM_THREADS']    = str(threads_per_worker)
        result = subprocess.run(
            [sys.executable, exp_script],
            cwd=EXPERIENCES_DIR,
            env=env,
            capture_output=True,
            text=True,
        )
        return task, result

    def run_phase(label, phase_tasks):
        if not phase_tasks:
            return 0, 0
        print(f"\n  ── {label} ({len(phase_tasks)} tâches) ──")
        ok, ko, done = 0, 0, 0
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(run_one, t) for t in phase_tasks]
            for fut in as_completed(futures):
                task, result = fut.result()
                run_idx, seed, exp_name, *_ = task
                done += 1
                tag = f"  [{done}/{len(phase_tasks)}] run_{run_idx} (seed={seed}) | {exp_name}"
                if result.returncode == 0:
                    print(f"{tag} ✓")
                    ok += 1
                else:
                    print(f"{tag} ❌ (code {result.returncode})")
                    if result.stderr:
                        print(f"      stderr: {result.stderr[-200:]}")
                    ko += 1
        return ok, ko

    ok1, ko1 = run_phase("Phase 1 : exp1", phase1_tasks)
    ok2, ko2 = run_phase("Phase 2 : exp2 + exp3", phase2_tasks)

    completed = ok1 + ok2
    failed = ko1 + ko2
    print(f"\n  📊 Résumé: {completed} succès, {skipped} skip, {failed} échecs\n")
    return failed == 0


# ===== ÉTAPE 3 : AGRÉGATION DES RÉSULTATS =====

def aggregate_results():
    """Agrège les résultats des runs pour chaque expérience"""
    print(f"\n[3/4] Agrégation des résultats")

    os.makedirs(AGGREGATED_DIR, exist_ok=True)

    from collections import defaultdict

    def load_metrics(run_dir, exp_name):
        metrics_file = os.path.join(run_dir, exp_name, 'metrics.json')
        if not os.path.exists(metrics_file):
            return None
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except:
            return None

    def aggregate_exp_generic(runs_data, is_exp3=False):
        """Agrège les résultats pour exp1/exp2/exp3"""
        models_metrics = defaultdict(lambda: {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': [],
            'duration': [],
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'iou': []
        })

        for data in runs_data:
            if data is None:
                continue

            # Format peut varier selon l'expérience
            if is_exp3:
                # Exp3 : {model_name: {'history': {...}, 'duration': ..., 'detailed_metrics': {...}}}
                items = data.items()
            else:
                # Exp1/Exp2 : {'histories': {...}, 'durations': {...}}
                histories = data.get('histories', data)
                durations = data.get('durations', {})
                items = [(name, {'history': h, 'duration': durations.get(name, 0)}) for name, h in histories.items()]

            for model_name, model_data in items:
                if isinstance(model_data, dict) and 'history' in model_data:
                    history = model_data['history']
                    duration = model_data.get('duration', 0)
                else:
                    history = model_data
                    duration = 0

                models_metrics[model_name]['train_loss'].append(history.get('train_loss', []))
                models_metrics[model_name]['train_acc'].append(history.get('train_acc', []))
                models_metrics[model_name]['test_loss'].append(history.get('test_loss', history.get('val_loss', [])))
                models_metrics[model_name]['test_acc'].append(history.get('test_acc', history.get('val_acc', [])))
                models_metrics[model_name]['duration'].append(duration)

                # Métriques détaillées (exp3 uniquement)
                if is_exp3 and 'detailed_metrics' in model_data:
                    dm = model_data['detailed_metrics']
                    models_metrics[model_name]['accuracy'].append(dm.get('accuracy', 0))
                    models_metrics[model_name]['precision'].append(dm.get('precision', 0))
                    models_metrics[model_name]['recall'].append(dm.get('recall', 0))
                    models_metrics[model_name]['f1'].append(dm.get('f1', 0))
                    if dm.get('iou') is not None:
                        models_metrics[model_name]['iou'].append(dm['iou'])

        # Calculer moyennes et écarts-types
        aggregated = {}
        for model_name, metrics in models_metrics.items():
            valid_test_acc = [x for x in metrics['test_acc'] if len(x) > 0]
            if not valid_test_acc:
                continue

            train_loss_array = np.array([x for x in metrics['train_loss'] if len(x) > 0])
            train_acc_array = np.array([x for x in metrics['train_acc'] if len(x) > 0])
            test_loss_array = np.array([x for x in metrics['test_loss'] if len(x) > 0])
            test_acc_array = np.array(valid_test_acc)
            duration_array = np.array(metrics['duration'])

            result = {
                'history': {
                    'train_loss_mean': train_loss_array.mean(axis=0).tolist(),
                    'train_loss_std': train_loss_array.std(axis=0).tolist(),
                    'train_acc_mean': train_acc_array.mean(axis=0).tolist(),
                    'train_acc_std': train_acc_array.std(axis=0).tolist(),
                    'test_loss_mean': test_loss_array.mean(axis=0).tolist(),
                    'test_loss_std': test_loss_array.std(axis=0).tolist(),
                    'test_acc_mean': test_acc_array.mean(axis=0).tolist(),
                    'test_acc_std': test_acc_array.std(axis=0).tolist(),
                },
                'duration_mean': float(duration_array.mean()),
                'duration_std': float(duration_array.std()),
                'final_test_acc_mean': float(test_acc_array[:, -1].mean()),
                'final_test_acc_std': float(test_acc_array[:, -1].std()),
                'n_runs': len(valid_test_acc)
            }

            if is_exp3 and metrics['accuracy']:
                result['detailed_metrics'] = {
                    'accuracy_mean': float(np.mean(metrics['accuracy'])),
                    'accuracy_std': float(np.std(metrics['accuracy'])),
                    'precision_mean': float(np.mean(metrics['precision'])),
                    'precision_std': float(np.std(metrics['precision'])),
                    'recall_mean': float(np.mean(metrics['recall'])),
                    'recall_std': float(np.std(metrics['recall'])),
                    'f1_mean': float(np.mean(metrics['f1'])),
                    'f1_std': float(np.std(metrics['f1'])),
                }
                if metrics['iou']:
                    result['detailed_metrics']['iou_mean'] = float(np.mean(metrics['iou']))
                    result['detailed_metrics']['iou_std'] = float(np.std(metrics['iou']))

            aggregated[model_name] = result

        return aggregated

    experiments = [('exp1', False), ('exp2', False), ('exp3', True)]

    for exp_name, is_exp3 in experiments:
        print(f"  {exp_name}...", end=' ', flush=True)

        runs_data = []
        for run_idx in range(len(SEEDS)):
            run_dir = os.path.join(RESULTS_DIR, f'run_{run_idx}')
            runs_data.append(load_metrics(run_dir, exp_name))

        valid_runs = sum(1 for d in runs_data if d is not None)
        if valid_runs == 0:
            print("❌")
            continue

        aggregated = aggregate_exp_generic(runs_data, is_exp3)

        output_file = os.path.join(AGGREGATED_DIR, f'{exp_name}_aggregated.json')
        with open(output_file, 'w') as f:
            json.dump(aggregated, f, indent=2)

        print(f"✓ ({valid_runs} runs)")

    return True


# ===== ÉTAPE 4 : GÉNÉRATION DES DONNÉES JS =====

def generate_data_js():
    """Génère le fichier data.js avec les résultats agrégés"""
    print(f"\n[4/4] Génération de data.js")

    exp1_file = os.path.join(AGGREGATED_DIR, 'exp1_aggregated.json')
    exp2_file = os.path.join(AGGREGATED_DIR, 'exp2_aggregated.json')
    exp3_file = os.path.join(AGGREGATED_DIR, 'exp3_aggregated.json')

    exp1_data = json.load(open(exp1_file)) if os.path.exists(exp1_file) else None
    exp2_data = json.load(open(exp2_file)) if os.path.exists(exp2_file) else None
    exp3_data = json.load(open(exp3_file)) if os.path.exists(exp3_file) else None

    data_js_file = os.path.join(AGGREGATED_DIR, 'data.js')
    with open(data_js_file, 'w') as f:
        f.write('// Données agrégées des 3 expériences\n')
        f.write('// Généré le ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n\n')
        if exp1_data:
            f.write(f'window.EXP1_DATA = {json.dumps(exp1_data, indent=2)};\n\n')
        if exp2_data:
            f.write(f'window.EXP2_DATA = {json.dumps(exp2_data, indent=2)};\n\n')
        if exp3_data:
            f.write(f'window.EXP3_DATA = {json.dumps(exp3_data, indent=2)};\n\n')

    html_file = os.path.join(AGGREGATED_DIR, 'index.html')
    print(f"  ✓ {html_file}\n")
    return html_file


# ===== MAIN =====

def main():
    parser = argparse.ArgumentParser(description="Exécution des expériences sur plusieurs datasets")
    parser.add_argument('--clean', action='store_true', help="Nettoyer et tout réexécuter")
    parser.add_argument('--workers', type=int, default=3,
                        help="Nombre de sous-processus en parallèle (défaut: 3)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  {len(SEEDS)} datasets × 3 expériences")
    print(f"{'='*60}")

    if args.clean:
        print("\n🧹 Nettoyage (datasets conservés)...")
        if os.path.exists(RESULTS_DIR):
            shutil.rmtree(RESULTS_DIR)
        for f in ['exp1_aggregated.json', 'exp2_aggregated.json', 'exp3_aggregated.json', 'data.js']:
            path = os.path.join(AGGREGATED_DIR, f)
            if os.path.exists(path):
                os.remove(path)
        print("  ✓ Nettoyage terminé")

    start_time = datetime.now()

    generate_datasets()
    run_all_experiments(num_workers=args.workers)
    aggregate_results()
    html_file = generate_data_js()

    elapsed = datetime.now() - start_time
    hours, remainder = divmod(elapsed.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"{'='*60}")
    print(f"  ✅ Terminé!")
    print(f"  ⏱️  Temps total: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"{'='*60}")
    print(f"  📊 {html_file}\n")


if __name__ == "__main__":
    # caffeinate empêche la mise en veille du Mac pendant l'exécution
    caffeinate_proc = subprocess.Popen(['caffeinate', '-dims'])
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Ctrl+C - Résultats partiels conservés\n")
        sys.exit(0)
    finally:
        caffeinate_proc.terminate()
