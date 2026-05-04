"""
config.py - Configuration du générateur de dataset

Modifiez ces paramètres pour personnaliser la génération du dataset.
"""
import os

class Config:
    # ===== CONFIGURATION GÉNÉRALE =====

    # Nombre total d'images à générer
    NUM_IMAGES = 1000

    # Taille des images (largeur, hauteur) en pixels
    IMAGE_SIZE = (128, 128)

    # Dossier de sortie
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'generated_dataset')


    # ===== RÉPARTITION DES CLASSES =====

    # Pourcentage de la classe positive (0-100)
    # Le reste sera négatif
    POSITIVE_RATIO = 50  # 50% positif, 50% négatif

    # Pourcentage d'images avec flèche dans chaque classe (0-100)
    ARROW_RATIO_POSITIVE = 70  # 70% des positifs ont une flèche
    ARROW_RATIO_NEGATIVE = 30  # 30% des négatifs ont une flèche

    # Pourcentage de cas positifs avec "bruit" (formes supplémentaires en dehors)
    # Ces images restent positives mais contiennent des distracteurs
    POSITIVE_WITH_NOISE_RATIO = 30  # 30% des positifs ont du bruit


    # ===== PARAMÈTRES VISUELS =====

    # Taille des cercles (rayon en pixels)
    MIN_CIRCLE_RADIUS = 20
    MAX_CIRCLE_RADIUS = 50

    # Taille des triangles (côté en pixels)
    MIN_TRIANGLE_SIDE = 20
    MAX_TRIANGLE_SIDE = 30

    # Paramètres des flèches
    ARROW_HEAD_SIZE = 8      # Taille de la tête de flèche
    ARROW_LINE_WIDTH = 2     # Épaisseur de la ligne de flèche (en pixels)

    # Types de cas négatifs à générer
    # Supprimez un type de cette liste pour ne pas le générer
    NEGATIVE_CASE_TYPES = [
        'neg_circle_only',       # Cercle seul
        'neg_two_triangles',     # 2 triangles dans un cercle
        'neg_triangle_out',      # Triangle hors du cercle
        'neg_multiple_circles',  # Plusieurs cercles
        'neg_disjoint',          # Cercles et triangles séparés
        'neg_empty',             # Fond vide
    ]

    # Nombre maximal de tentatives pour placer une forme
    MAX_PLACEMENT_ATTEMPTS = 50
