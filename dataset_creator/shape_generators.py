"""
shape_generators.py - Génération des formes de base

Contient la logique pour générer :
- Cercles (avec détection de chevauchement)
- Triangles équilatéraux
- Flèches
"""
import random
import math
from geometry_utils import GeometryUtils


class ShapeGenerator:
    """Génère les formes de base : cercles, triangles, flèches."""

    def __init__(self, config):
        self.config = config

    def generate_random_circle(self, img_w, img_h, existing_circles=None):
        """
        Génère un cercle aléatoire dans l'image.

        Args:
            img_w: largeur de l'image
            img_h: hauteur de l'image
            existing_circles: liste de tuples (center, radius) pour éviter les chevauchements

        Returns:
            (center, radius) ou (None, None) si impossible de placer le cercle
        """
        max_attempts = 100
        for _ in range(max_attempts):
            radius = random.randint(self.config.MIN_CIRCLE_RADIUS, self.config.MAX_CIRCLE_RADIUS)
            margin = radius + 2
            x = random.randint(margin, img_w - margin)
            y = random.randint(margin, img_h - margin)

            if existing_circles:
                overlap = False
                for qc, qr in existing_circles:
                    if GeometryUtils.is_circle_overlap((x, y), radius, qc, qr):
                        overlap = True
                        break
                if overlap:
                    continue

            return (x, y), radius
        return None, None

    def generate_random_triangle(self, img_w, img_h):
        """
        Génère un triangle équilatéral aléatoire dans l'image.

        Args:
            img_w: largeur de l'image
            img_h: hauteur de l'image

        Returns:
            (points, center) ou (None, None) si le triangle sort de l'image
        """
        center_x = random.randint(0, img_w)
        center_y = random.randint(0, img_h)
        return self._create_triangle_points(center_x, center_y, img_w, img_h)

    def _create_triangle_points(self, center_x, center_y, img_w, img_h):
        """Crée les points d'un triangle équilatéral. Retourne None si le triangle sort des limites."""
        side_length = random.randint(self.config.MIN_TRIANGLE_SIDE, self.config.MAX_TRIANGLE_SIDE)
        dist = side_length / math.sqrt(3)
        angle_offset = random.uniform(0, 2 * math.pi)
        points = []

        # Générer les 3 points
        for i in range(3):
            angle = angle_offset + i * (2 * math.pi / 3)
            pt_x = int(center_x + dist * math.cos(angle))
            pt_y = int(center_y + dist * math.sin(angle))

            # Vérifier si le point est hors limites
            if pt_x < 0 or pt_x >= img_w or pt_y < 0 or pt_y >= img_h:
                return None, None  # Le triangle serait coupé - le rejeter

            points.append((pt_x, pt_y))

        return points, (center_x, center_y)

    def draw_arrow(self, draw, start, end, color=0):
        """Dessine une flèche nette avec un meilleur rendu."""
        # Convertir en entiers pour un rendu au pixel près
        start = (int(round(start[0])), int(round(start[1])))
        end = (int(round(end[0])), int(round(end[1])))

        # Dessiner la ligne principale avec l'épaisseur spécifiée
        draw.line([start, end], fill=color, width=self.config.ARROW_LINE_WIDTH)

        # Calculer la tête de la flèche
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.atan2(dy, dx)

        head_size = self.config.ARROW_HEAD_SIZE
        angle1 = angle + math.pi - math.pi / 6
        angle2 = angle + math.pi + math.pi / 6

        # Calculer les points de la tête de flèche avec des coordonnées entières
        p1 = (int(round(end[0] + head_size * math.cos(angle1))),
              int(round(end[1] + head_size * math.sin(angle1))))
        p2 = (int(round(end[0] + head_size * math.cos(angle2))),
              int(round(end[1] + head_size * math.sin(angle2))))

        # Dessiner le triangle plein pour la tête de flèche
        draw.polygon([end, p1, p2], fill=color, outline=color)

        # Ajouter des lignes supplémentaires pour rendre la tête de flèche plus épaisse et plus visible
        if self.config.ARROW_LINE_WIDTH > 1:
            draw.line([end, p1], fill=color, width=self.config.ARROW_LINE_WIDTH)
            draw.line([end, p2], fill=color, width=self.config.ARROW_LINE_WIDTH)

    def add_random_arrow(self, draw, w, h, target_center=None, target_radius=None, force_arrow=False):
        """Ajoute une flèche à l'image si force_arrow est True. Réessaye les positions jusqu'à ce que cela fonctionne."""
        if not force_arrow:
            return True, False  # Succès (exécution valide), mais pas de flèche

        for _ in range(50):
            angle = random.uniform(0, 2 * math.pi)

            if target_center and target_radius:
                end_x = target_center[0] + target_radius * math.cos(angle)
                end_y = target_center[1] + target_radius * math.sin(angle)

                arrow_len = random.randint(20, 40)
                start_x = target_center[0] + (target_radius + arrow_len) * math.cos(angle)
                start_y = target_center[1] + (target_radius + arrow_len) * math.sin(angle)
            else:
                # pointer aléatoirement
                end_x = random.randint(20, w - 20)
                end_y = random.randint(20, h - 20)
                arrow_len = random.randint(20, 40)
                start_x = end_x + arrow_len * math.cos(angle)
                start_y = end_y + arrow_len * math.sin(angle)

            if not (0 <= start_x < w and 0 <= start_y < h):
                angle = angle + math.pi
                if target_center and target_radius:
                    end_x = target_center[0] + target_radius * math.cos(angle)
                    end_y = target_center[1] + target_radius * math.sin(angle)

                    arrow_len = random.randint(20, 40)
                    start_x = target_center[0] + (target_radius + arrow_len) * math.cos(angle)
                    start_y = target_center[1] + (target_radius + arrow_len) * math.sin(angle)
                else:
                    # pointer aléatoirement
                    end_x = random.randint(20, w - 20)
                    end_y = random.randint(20, h - 20)
                    arrow_len = random.randint(20, 40)
                    start_x = end_x + arrow_len * math.cos(angle)
                    start_y = end_y + arrow_len * math.sin(angle)

                if not (0 <= start_x < w and 0 <= start_y < h):
                    angle = angle + math.pi  # Essayer la direction opposée
                    if target_center and target_radius:
                        end_x = target_center[0] + target_radius * math.cos(angle)
                        end_y = target_center[1] + target_radius * math.sin(angle)
                        start_x = target_center[0] + (target_radius + arrow_len) * math.cos(angle)
                        start_y = target_center[1] + (target_radius + arrow_len) * math.sin(angle)
                    else:
                        start_x = end_x + arrow_len * math.cos(angle)
                        start_y = end_y + arrow_len * math.sin(angle)

                if not (0 <= start_x < w and 0 <= start_y < h):
                    continue

            self.draw_arrow(draw, (start_x, start_y), (end_x, end_y), color=0)
            return True, True

        return False, False
