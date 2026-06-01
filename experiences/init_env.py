"""
init_env.py - Initialisation de l'environnement Python pour les run_exp*.py.

Effets de bord (déclenchés à l'import) :
  1. Ajoute experiences/ à sys.path pour que `from shared.* import ...`
     fonctionne quel que soit le répertoire de lancement.
  2. Si l'argument CLI `--dataset NOM` est présent, résout NOM en chemin absolu
     (cherche d'abord dans experiences/datasets/, puis dans dataset_creator/)
     et l'expose via la variable d'environnement DATASET_PATH — qui est lue
     par shared/config.py à son propre import.

Doit être importé EN PREMIER dans run_exp*.py, AVANT tout `from shared.* import`.

Constantes exposées :
  - EXP_DIR : chemin absolu du dossier experiences/
  - ROOT    : chemin absolu du dossier racine du repo (parent d'experiences/)
"""
import os
import sys

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(EXP_DIR)

# 1. sys.path : permet `from shared.* import ...` depuis n'importe où.
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

# 2. DATASET_PATH : doit être défini AVANT que shared/config.py soit importé
#    (config lit cette variable au moment de son import pour résoudre les
#    chemins images/, heatmaps/, annotations.csv).
if '--dataset' in sys.argv:
    _idx = sys.argv.index('--dataset')
    if _idx + 1 < len(sys.argv):
        _ds_name = sys.argv[_idx + 1]
        # Cherche d'abord dans experiences/datasets/, puis dans dataset_creator/
        _candidate = os.path.join(EXP_DIR, 'datasets', _ds_name)
        if not os.path.isdir(_candidate):
            _candidate = os.path.join(ROOT, 'dataset_creator', _ds_name)
        os.environ['DATASET_PATH'] = _candidate
elif 'DATASET_PATH' not in os.environ:
    os.environ['DATASET_PATH'] = os.path.join(ROOT, 'dataset_creator', 'generated_dataset')
