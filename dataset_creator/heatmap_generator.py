"""
heatmap_generator.py — Génération de heatmaps de style GradCAM

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
        Generate a GradCAM-style heatmap with four discrete zones:
        - Triangle interior  : 0.98 (red)
        - Triangle contour   : 0.85 (orange)
        - Circle interior    : 0.75 (yellow)
        - Background         : 0.05 (dark blue)

        Returns:
            - heatmap_values: numpy array (h, w) with values in [0, 1] (for training)
            - heatmap_rgb: PIL Image with colorized visualization
        """
        y, x = np.ogrid[:h, :w]
        heatmap = np.zeros((h, w), dtype=float)

        if not triangles or not circles:
            # Fallback: return zeros for values and blue background for RGB
            heatmap_values = np.zeros((h, w), dtype=np.float32)
            heatmap_rgb = Image.fromarray(np.full((h, w, 3), [0, 0, 139], dtype=np.uint8))
            return heatmap_values, heatmap_rgb

        # Get the first (and usually only) triangle and circle
        triangle_pts = triangles[0] if triangles else None
        cx, cy, cr = circles[0] if circles else (w//2, h//2, min(w, h)//4)

        if triangle_pts is None:
            heatmap_values = np.zeros((h, w), dtype=np.float32)
            heatmap_rgb = Image.fromarray(np.full((h, w, 3), [0, 0, 139], dtype=np.uint8))
            return heatmap_values, heatmap_rgb

        # Mask for pixels inside the circle
        dist_from_circle_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        inside_circle = dist_from_circle_center <= cr

        # Create mask for pixels inside triangle using PIL
        triangle_mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(triangle_mask)
        draw.polygon(triangle_pts, fill=255)
        inside_triangle = np.array(triangle_mask) > 0

        heatmap = np.zeros((h, w), dtype=float)
        heatmap[:] = 0.05                   # Background
        heatmap[inside_circle] = 0.75       # Circle interior

        # Triangle contour: dilate the triangle mask by contour_width pixels
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
        heatmap[triangle_contour] = 0.85    # Triangle contour
        heatmap[inside_triangle] = 0.98     # Triangle interior

        # Convert heatmap values to GradCAM colors (zones bien délimitées)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                val = heatmap[i, j]
                if val < 0.15:
                    rgb[i, j] = [0, 0, 128]    # Background: bleu foncé
                elif val < 0.80:
                    rgb[i, j] = [255, 255, 0]  # Circle: jaune
                elif val < 0.90:
                    rgb[i, j] = [255, 165, 0]  # Triangle contour: orange
                else:
                    rgb[i, j] = [255, 0, 0]    # Triangle: rouge

        # Return raw values for training and colorized image for visualization
        heatmap_values = heatmap.astype(np.float32)
        heatmap_rgb = Image.fromarray(rgb, mode='RGB')
        return heatmap_values, heatmap_rgb
