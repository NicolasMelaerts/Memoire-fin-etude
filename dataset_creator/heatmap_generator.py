"""
heatmap_generator.py - Génération de heatmaps de style GradCAM

Contient la logique de génération des heatmaps pour les échantillons positifs :
- Triangle = zone la plus importante (rouge)
- Contour du triangle = zone importante (orange)
- Cercle = zone modérément importante (jaune)
- Background = non important (bleu)
"""
import numpy as np
from PIL import Image, ImageDraw


class HeatmapGenerator:
    """Génère des heatmaps de type GradCAM pour visualiser l'importance des zones."""

    def __init__(self, config):
        self.config = config

    def generate_heatmap(self, w, h, circles, triangles):
        """
        Génère une heatmap de type GradCAM avec quatre zones discrètes :
        - Intérieur du triangle : 0.98 (rouge)
        - Contour du triangle   : 0.85 (orange)
        - Intérieur du cercle   : 0.75 (jaune)
        - Arrière-plan (bg)     : 0.05 (bleu foncé)

        Retourne :
            - heatmap_values : tableau numpy (h, w) avec des valeurs dans [0, 1] (pour l'entraînement)
            - heatmap_rgb : image PIL avec la visualisation colorisée
        """
        y, x = np.ogrid[:h, :w]
        heatmap = np.zeros((h, w), dtype=float)

        if not triangles or not circles:
            # Repli : retourner des zéros pour les valeurs et un fond bleu pour le RGB
            heatmap_values = np.zeros((h, w), dtype=np.float32)
            heatmap_rgb = Image.fromarray(np.full((h, w, 3), [0, 0, 139], dtype=np.uint8))
            return heatmap_values, heatmap_rgb

        # Récupérer le premier (et généralement unique) triangle et cercle
        triangle_pts = triangles[0] if triangles else None
        cx, cy, cr = circles[0] if circles else (w//2, h//2, min(w, h)//4)

        if triangle_pts is None:
            heatmap_values = np.zeros((h, w), dtype=np.float32)
            heatmap_rgb = Image.fromarray(np.full((h, w, 3), [0, 0, 139], dtype=np.uint8))
            return heatmap_values, heatmap_rgb

        # Masque pour les pixels à l'intérieur du cercle
        dist_from_circle_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        inside_circle = dist_from_circle_center <= cr

        # Créer un masque pour les pixels à l'intérieur du triangle avec PIL
        triangle_mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(triangle_mask)
        draw.polygon(triangle_pts, fill=255)
        inside_triangle = np.array(triangle_mask) > 0

        heatmap = np.zeros((h, w), dtype=float)
        heatmap[:] = 0.05                   # Arrière-plan (background)
        heatmap[inside_circle] = 0.75       # Intérieur du cercle

        # Contour du triangle : dilater le masque du triangle de 'contour_width' pixels
        contour_width = 5
        try:
            from scipy.ndimage import binary_dilation as dilate_func
            kernel = np.ones((contour_width*2+1, contour_width*2+1))
            dilated_triangle = dilate_func(inside_triangle, structure=kernel)
        except ImportError:
            dilated_triangle = inside_triangle.copy()
            for dy in range(-contour_width, contour_width+1):
                for dx in range(-contour_width, contour_width+1):
                    if dy == 0 and dx == 0:
                        continue
                    shifted = np.roll(np.roll(inside_triangle, dy, axis=0), dx, axis=1)
                    dilated_triangle = dilated_triangle | shifted

        triangle_contour = dilated_triangle & ~inside_triangle & inside_circle
        heatmap[triangle_contour] = 0.85    # Contour du triangle
        heatmap[inside_triangle] = 0.98     # Intérieur du triangle

        # Convertir les valeurs de heatmap en couleurs GradCAM (zones bien délimitées)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                val = heatmap[i, j]
                if val < 0.15:
                    rgb[i, j] = [0, 0, 128]    # Arrière-plan : bleu foncé
                elif val < 0.80:
                    rgb[i, j] = [255, 255, 0]  # Cercle : jaune
                elif val < 0.90:
                    rgb[i, j] = [255, 165, 0]  # Contour du triangle : orange
                else:
                    rgb[i, j] = [255, 0, 0]    # Triangle : rouge

        # Retourner les valeurs brutes pour l'entraînement et l'image colorisée pour la visualisation
        heatmap_values = heatmap.astype(np.float32)
        heatmap_rgb = Image.fromarray(rgb, mode='RGB')
        return heatmap_values, heatmap_rgb
