"""
dataset_generator.py - Génère un dataset synthétique Inside/Outside

Classes :
  - Positive : Triangle DANS un cercle (heatmap générée)
  - Negative : 6 types différents (cercle seul, triangle dehors, etc.)

Usage :
    python3 dataset_generator.py

Configuration :
    Modifiez config.py pour ajuster le nombre d'images, tailles, ratios, etc.
"""
import numpy as np
from PIL import Image, ImageDraw
import os
import random
import json
import csv
from config import Config
from shape_generators import ShapeGenerator
from heatmap_generator import HeatmapGenerator
from sample_generators import SampleGenerator


class DatasetGenerator:
    """Orchestrateur principal pour la génération du dataset."""

    def __init__(self, config):
        self.config = config
        self.images_dir = os.path.join(config.OUTPUT_DIR, 'images')
        self.heatmaps_dir = os.path.join(config.OUTPUT_DIR, 'heatmaps')

        # Créer les répertoires de sortie
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        if not os.path.exists(self.heatmaps_dir):
            os.makedirs(self.heatmaps_dir)

        # Initialiser les générateurs
        self.shape_gen = ShapeGenerator(config)
        self.heatmap_gen = HeatmapGenerator(config)
        self.sample_gen = SampleGenerator(config, self.shape_gen, self.heatmap_gen)

    def generate_sample(self, sample_id, case_type, force_arrow):
        """
        Génère un échantillon (image + heatmap si positif).

        Args:
            sample_id: identifiant unique de l'échantillon
            case_type: type d'échantillon à générer
            force_arrow: True pour forcer l'ajout d'une flèche

        Returns:
            (filename, label, has_arrow) ou (None, None, False) en cas d'échec
        """
        w, h = self.config.IMAGE_SIZE
        img = Image.new('L', (w, h), 255)
        draw = ImageDraw.Draw(img)

        heatmap = None
        label = 'negative'

        # Aiguiller vers le générateur approprié
        if case_type == 'positive':
            success_result = self.sample_gen.generate_positive_sample(w, h, draw, force_arrow)
            label = 'positive'
        elif case_type == 'positive_noise':
            success_result = self.sample_gen.generate_positive_sample_with_noise(w, h, draw, force_arrow)
            label = 'positive'
        elif case_type == 'neg_circle_only':
            success_result = self.sample_gen.generate_negative_circle_only(w, h, draw, force_arrow)
        elif case_type == 'neg_two_triangles':
            success_result = self.sample_gen.generate_negative_two_triangles_in(w, h, draw, force_arrow)
        elif case_type == 'neg_triangle_out':
            success_result = self.sample_gen.generate_negative_triangle_out(w, h, draw, force_arrow)
        elif case_type == 'neg_multiple_circles':
            success_result = self.sample_gen.generate_negative_multiple_circles(w, h, draw, force_arrow)
        elif case_type == 'neg_disjoint':
            success_result = self.sample_gen.generate_negative_disjoint(w, h, draw, force_arrow)
        elif case_type == 'neg_empty':
            success_result = self.sample_gen.generate_negative_empty(w, h, draw, force_arrow)

        has_arrow = False
        if isinstance(success_result, tuple):
            success, heatmap, has_arrow = success_result
        else:
            success = success_result

        if success:
            filename = f"image_{sample_id:05d}.png"
            filepath = os.path.join(self.images_dir, filename)

            # Supprimer le fichier existant s'il existe (évite le renommage automatique de PIL)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    # Probablement un problème de verrouillage (service de synchronisation).
                    pass

            try:
                img.save(filepath)
            except Exception as e:
                # Si la sauvegarde échoue, nous retournons tout de même le nom attendu, le nettoyage corrigera cela
                print(f"  Warning: Save failed for {filename}, cleanup will attempt fix. ({e})")

            img.close()  # Fermer l'image pour libérer le descripteur de fichier

            if heatmap:
                heatmap_values, heatmap_rgb = heatmap

                heatmap_png_path = os.path.join(self.heatmaps_dir, f"image_{sample_id:05d}_heatmap.png")
                if os.path.exists(heatmap_png_path):
                    os.remove(heatmap_png_path)
                heatmap_rgb.save(heatmap_png_path)
                heatmap_rgb.close()  # Fermer pour libérer le descripteur de fichier

                # Sauvegarder les valeurs brutes pour l'entraînement (tableau numpy)
                heatmap_npy_path = os.path.join(self.heatmaps_dir, f"image_{sample_id:05d}_heatmap.npy")
                if os.path.exists(heatmap_npy_path):
                    os.remove(heatmap_npy_path)
                np.save(heatmap_npy_path, heatmap_values)

            return filename, label, has_arrow

        return None, None, False

    def run(self):
        """Lance la génération du dataset complet."""
        annotations = []
        total = self.config.NUM_IMAGES

        # Utiliser la configuration pour les ratios
        positive_count = int(total * self.config.POSITIVE_RATIO / 100)
        negative_count = total - positive_count

        neg_types = self.config.NEGATIVE_CASE_TYPES

        # Positifs: séparer ceux avec bruit de ceux sans bruit
        pos_with_noise_count = int(positive_count * self.config.POSITIVE_WITH_NOISE_RATIO / 100)
        pos_without_noise_count = positive_count - pos_with_noise_count

        # Pour les positifs sans bruit
        pos_no_noise_arrow = int(pos_without_noise_count * self.config.ARROW_RATIO_POSITIVE / 100)
        pos_no_noise_no_arrow = pos_without_noise_count - pos_no_noise_arrow
        positives_clean = [('positive', True)] * pos_no_noise_arrow + [('positive', False)] * pos_no_noise_no_arrow

        # Pour les positifs avec bruit
        pos_noise_arrow = int(pos_with_noise_count * self.config.ARROW_RATIO_POSITIVE / 100)
        pos_noise_no_arrow = pos_with_noise_count - pos_noise_arrow
        positives_noisy = [('positive_noise', True)] * pos_noise_arrow + [('positive_noise', False)] * pos_noise_no_arrow

        positives = positives_clean + positives_noisy

        num_types = len(neg_types)
        base_count = negative_count // num_types
        remainder = negative_count % num_types

        type_counts = [base_count] * num_types
        for i in range(remainder):
            type_counts[i] += 1

        negatives = []

        arrow_target = int(negative_count * self.config.ARROW_RATIO_NEGATIVE / 100)
        arrows_assigned = 0

        # Construire les répartitions exactes
        for type_idx, count in enumerate(type_counts):
            # Répartir les flèches proportionnellement
            type_arrow_target = int(count * self.config.ARROW_RATIO_NEGATIVE / 100)

            # Si dernier type, ajuster pour atteindre exactement le target global
            if type_idx == num_types - 1:
                type_arrow_target = arrow_target - arrows_assigned

            arrows_assigned += type_arrow_target
            type_no_arrow_target = count - type_arrow_target

            # Ajouter aux totaux corrects
            negatives.extend([(neg_types[type_idx], True)] * type_arrow_target)
            negatives.extend([(neg_types[type_idx], False)] * type_no_arrow_target)

        tasks = positives + negatives
        random.shuffle(tasks)

        print(f"Generating {total} images...")

        # Nettoyer grossièrement le dossier
        for f in os.listdir(self.images_dir):
            os.remove(os.path.join(self.images_dir, f))
        for f in os.listdir(self.heatmaps_dir):
            if f.endswith('.png') or f.endswith('.npy'):
                os.remove(os.path.join(self.heatmaps_dir, f))

        count = 0
        for i, _ in enumerate(tasks):
            # Boucle de tentative pour garantir le succès (max 100 essais par image)
            max_attempts = 100
            for attempt in range(max_attempts):
                case_type, force_arrow = tasks[i]
                filename, label, has_arrow = self.generate_sample(count, case_type, force_arrow)
                if filename:
                    annotations.append({
                        "filename": filename,
                        "class": label,
                        "case": case_type,
                        "has_arrow": has_arrow
                    })
                    count += 1
                    break
            else:
                # Échec après max_attempts
                print(f"Warning: Failed to generate image {count} after {max_attempts} attempts")
                count += 1

        print(f"Generated {count} / {total} images successfully.")

        # Sauvegarder data.js pour la visualisation HTML
        data_js_path = os.path.join(self.config.OUTPUT_DIR, 'data.js')
        with open(data_js_path, 'w') as f:
            f.write(f"const datasetData = {json.dumps(annotations, indent=2)};")
        print(f"  → data.js")

        # Sauvegarder également le CSV pour la compatibilité avec les scripts d'entraînement
        csv_path = os.path.join(self.config.OUTPUT_DIR, 'annotations.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'class', 'case', 'has_arrow'])
            writer.writeheader()
            writer.writerows(annotations)
        print(f"  → annotations.csv")

        print(f"\n✓ Dataset generated at {self.config.OUTPUT_DIR}")
        print(f"  {len([a for a in annotations if a['class'] == 'positive'])} positive images")
        print(f"  {len([a for a in annotations if a['class'] == 'negative'])} negative images")


if __name__ == "__main__":
    gen = DatasetGenerator(Config)
    gen.run()
