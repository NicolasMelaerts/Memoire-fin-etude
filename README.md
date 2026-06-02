# Améliorer les prédictions d'un réseau de neurones sur base d'explications a priori

Ce dépôt contient le code source associé au mémoire de fin d'études décrit ci-dessous.

## 🎓 Informations sur le Mémoire

*   **Titre :** Améliorer les prédictions d'un réseau de neurones sur base d'explications a priori
*   **Auteur :** Nicolas Melaerts (<nicolas.melaerts@student.umons.ac.be>)
*   **Institution :** Université de Mons (UMONS) — Faculté des Sciences
*   **Cursus :** Master en Sciences Informatiques, finalité spécialisée
*   **Directeur :** Pierre Vandenhove
*   **Co-directeur :** Stéphane Dupont
*   **Année académique :** 2025–2026

---

## 📊 Visualisation des Résultats

Pour ouvrir directement dans votre navigateur par défaut toutes les visualisations HTML interactives des expériences déjà exécutées et leurs résultats associés, exécutez la commande suivante à la racine du dépôt :

```bash
./open_results.sh
```

---

## 🧪 Dataset Synthétique (Inside/Outside)

### 1. Génération du dataset

Pour générer le dataset synthétique principal (géométries cercle/triangle) :

```bash
python3 dataset_creator/dataset_generator.py
```

### 2. Lancement des expériences 1, 2 et 3 (avec `--clean` et `dataset_seed13`)

Ces expériences comparent les performances et la robustesse des modèles entraînés sur le dataset d'expérience `dataset_seed13` (le paramètre `--dataset` accepte le nom du dossier présent dans `experiences/datasets/`) :

```bash
# Expérience 1
python3 experiences/run_exp1.py --clean --dataset dataset_seed13

# Expérience 2
python3 experiences/run_exp2.py --clean --dataset dataset_seed13

# Expérience 3
python3 experiences/run_exp3.py --clean --dataset dataset_seed13
```

### 3. Expérience 4

Cette expérience évalue la robustesse du modèle face à un biais statistique inversé en phase de test (la flèche prédit la classe positive en entraînement mais la classe négative en test) :

```bash
python3 experiences/run_exp4.py --clean
```

### 4. Expérience Multi-Datasets

Pour lancer l'évaluation et l'agrégation des résultats sur l'ensemble des 15 datasets synthétiques générés avec différentes graines (seeds) :

```bash
python3 experiences/run_multidataset.py --clean
```

---

## 🖼️ Datasets Réels

Les datasets réels ne sont pas inclus dans ce dépôt en raison de leur taille. Pour reproduire ces expériences, vous devez les télécharger et les placer dans les répertoires indiqués ci-dessous.

### 1. CUB-200-2011 (Oiseaux)

*   **Téléchargement :**
    *   Dataset complet (contenant les images et les segmentations) : [CUB2002011 (Kaggle)](https://www.kaggle.com/datasets/wenewone/cub2002011)
*   **Installation :** Décompressez les fichiers de sorte à obtenir la structure suivante :
    ```
    real_datasets/
    └── oiseaux/
        ├── CUB_200_2011/      # Contient images.txt, classes.txt, images/, etc.
        └── segmentations/     # Contient les dossiers de masques d'oiseaux (.png)
    ```

### 2. Large-Scale Fish Dataset (Poissons)

*   **Téléchargement :**
    *   [A Large Scale Fish Dataset (Kaggle)](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)
*   **Installation :** Décompressez le dataset et renommez le dossier racine en `fish-dataset` de sorte à obtenir la structure suivante :
    ```
    real_datasets/
    └── poissons/
        └── fish-dataset/      # Contient Gilt-Head Bream/, Gilt-Head Bream GT/, etc.
    ```

### 3. Lancement des expériences sur datasets réels

Une fois les datasets installés, vous pouvez lancer les trois scripts d'expériences réelles :

```bash
# 1. Classification d'oiseaux (CUB-200-2011) : Normal vs Guided GradCAM (SimpleCNN et ResNet18)
python3 real_datasets/oiseaux/exp_oiseaux.py --clean

# 2. Classification de poissons (Fish Dataset) : Normal vs Guided GradCAM (SimpleCNN et ResNet18)
python3 real_datasets/poissons/exp_poissons.py --clean

# 3. Efficacité des données sur les poissons (Fish Dataset avec moins de données d'entraînement)
python3 real_datasets/poissons/exp_poissons_efficacite.py --clean
```