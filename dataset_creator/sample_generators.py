"""
sample_generators.py - Générateurs d'échantillons positifs et négatifs

Contient tous les générateurs de samples :
- Positifs : triangle dans cercle (avec ou sans bruit)
- Négatifs : cercle seul, 2 triangles, triangle dehors, etc.
"""
import random
import math
from geometry_utils import GeometryUtils


class SampleGenerator:
    """Génère les différents types d'échantillons (positifs et négatifs)."""

    def __init__(self, config, shape_generator, heatmap_generator):
        self.config = config
        self.shape_gen = shape_generator
        self.heatmap_gen = heatmap_generator

    def generate_positive_sample(self, w, h, draw, force_arrow):
        """
        Génère un échantillon positif : cercle avec triangle à l'intérieur.
        Flèche pointant vers le cercle depuis l'extérieur.
        """
        # Circle with triangle inside + arrow from OUTSIDE pointing TO circle
        circle_center, circle_radius = self.shape_gen.generate_random_circle(w, h)
        if not circle_center:
            return False

        # Try to put triangle inside
        triangle_points = None

        for _ in range(50):
            # Generate random center inside circle with margin
            margin = (self.config.MAX_TRIANGLE_SIDE / math.sqrt(3)) + 5
            if circle_radius < margin:
                return False

            r = random.uniform(0, circle_radius - margin)
            theta = random.uniform(0, 2*math.pi)
            tx = int(circle_center[0] + r * math.cos(theta))
            ty = int(circle_center[1] + r * math.sin(theta))

            pts, center = self.shape_gen._create_triangle_points(tx, ty, w, h)
            # Check if triangle is valid (not clipped) and inside circle
            if pts is not None and GeometryUtils.is_triangle_in_circle(pts, circle_center, circle_radius):
                triangle_points = pts
                break

        if not triangle_points:
            return False

        # Draw Circle
        x, y = circle_center
        r = circle_radius
        draw.ellipse([x-r, y-r, x+r, y+r], outline=None, fill=128)

        # Draw Triangle
        draw.polygon(triangle_points, outline=None, fill=0)

        # Draw random Arrow: Outside pointing to circle
        arrow_success, has_arrow = self.shape_gen.add_random_arrow(draw, w, h, circle_center, circle_radius, force_arrow=force_arrow)
        if not arrow_success:
            return False

        # Generate heatmap (both values and RGB)
        heatmap_values, heatmap_rgb = self.heatmap_gen.generate_heatmap(w, h, [(circle_center[0], circle_center[1], circle_radius)], [triangle_points])
        return True, (heatmap_values, heatmap_rgb), has_arrow

    def generate_positive_sample_with_noise(self, w, h, draw, force_arrow):
        """
        Positive sample with noise (distractors):
        - Main: circle with triangle inside (positive pattern)
        - Noise: 1-2 extra circles and/or triangles outside the main circle
        """
        # First generate the standard positive pattern
        circle_center, circle_radius = self.shape_gen.generate_random_circle(w, h)
        if not circle_center:
            return False

        # Try to put triangle inside main circle
        triangle_points = None

        for _ in range(50):
            margin = (self.config.MAX_TRIANGLE_SIDE / math.sqrt(3)) + 5
            if circle_radius < margin:
                return False

            r = random.uniform(0, circle_radius - margin)
            theta = random.uniform(0, 2*math.pi)
            tx = int(circle_center[0] + r * math.cos(theta))
            ty = int(circle_center[1] + r * math.sin(theta))

            pts, center = self.shape_gen._create_triangle_points(tx, ty, w, h)
            if pts is not None and GeometryUtils.is_triangle_in_circle(pts, circle_center, circle_radius):
                triangle_points = pts
                break

        if not triangle_points:
            return False

        # Track all shapes for overlap checking
        all_circles = [(circle_center, circle_radius)]
        all_triangles = [triangle_points]

        # Garantir au moins 1 élément de bruit (1-2 cercles ou 1-2 triangles)
        # Strategy: roll for circles first, if 0, force at least 1 triangle
        num_noise_circles = random.randint(0, 2)
        num_noise_triangles = random.randint(0, 2)

        # Si aucun bruit n'est prévu, forcer au moins 1 élément
        if num_noise_circles == 0 and num_noise_triangles == 0:
            if random.random() < 0.5:
                num_noise_circles = random.randint(1, 2)
            else:
                num_noise_triangles = random.randint(1, 2)

        # Add 1-2 noise circles (outside main circle)
        for _ in range(num_noise_circles):
            for attempt in range(30):
                nc, nr = self.shape_gen.generate_random_circle(w, h, all_circles)
                if nc and not GeometryUtils.is_circle_overlap(nc, nr, circle_center, circle_radius):
                    all_circles.append((nc, nr))
                    break

        # Add 1-2 noise triangles (outside all circles)
        for _ in range(num_noise_triangles):
            for attempt in range(30):
                pts, _ = self.shape_gen.generate_random_triangle(w, h)
                if pts is None:
                    continue

                # Check it's outside all circles
                outside_all = True
                for c, r in all_circles:
                    if not GeometryUtils.is_triangle_outside_circle(pts, c, r):
                        outside_all = False
                        break

                # Check it doesn't overlap with existing triangles
                if outside_all:
                    overlaps = False
                    for existing_tri in all_triangles:
                        if GeometryUtils.triangles_overlap(pts, existing_tri):
                            overlaps = True
                            break

                    if not overlaps:
                        all_triangles.append(pts)
                        break

        # Draw all shapes
        # Circles
        for (cx, cy), cr in all_circles:
            draw.ellipse([cx-cr, cy-cr, cx+cr, cy+cr], outline=None, fill=128)

        # Triangles
        for tri in all_triangles:
            draw.polygon(tri, outline=None, fill=0)

        # Draw arrow pointing to main circle
        arrow_success, has_arrow = self.shape_gen.add_random_arrow(draw, w, h, circle_center, circle_radius, force_arrow=force_arrow)
        if not arrow_success:
            return False

        # Generate heatmap (only for main circle+triangle, noise is ignored)
        heatmap_values, heatmap_rgb = self.heatmap_gen.generate_heatmap(w, h, [(circle_center[0], circle_center[1], circle_radius)], [triangle_points])
        return True, (heatmap_values, heatmap_rgb), has_arrow

    def generate_negative_circle_only(self, w, h, draw, force_arrow):
        """Génère un cercle seul (négatif)."""
        center, radius = self.shape_gen.generate_random_circle(w, h)
        if not center:
            return False
        x, y = center
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], outline=None, fill=128)
        arrow_success, has_arrow = self.shape_gen.add_random_arrow(draw, w, h, center, radius, force_arrow=force_arrow)
        if not arrow_success:
            return False

        return True, None, has_arrow

    def generate_negative_two_triangles_in(self, w, h, draw, force_arrow):
        """Génère un cercle avec 2 triangles à l'intérieur (négatif)."""
        # Circle with 2 triangles inside
        circle_center, circle_radius = self.shape_gen.generate_random_circle(w, h)
        if not circle_center:
            return False

        triangles = []
        for _ in range(50):
            if len(triangles) == 2:
                break
            margin = (self.config.MAX_TRIANGLE_SIDE / math.sqrt(3)) + 2
            r = random.uniform(0, circle_radius - margin)
            theta = random.uniform(0, 2*math.pi)
            tx = int(circle_center[0] + r * math.cos(theta))
            ty = int(circle_center[1] + r * math.sin(theta))
            pts, _ = self.shape_gen._create_triangle_points(tx, ty, w, h)

            # Check if triangle is valid (not clipped)
            if pts is None:
                continue

            # Check if triangle is in circle
            if not GeometryUtils.is_triangle_in_circle(pts, circle_center, circle_radius):
                continue

            # Check if it overlaps with existing triangles
            overlaps = False
            for existing_tri in triangles:
                if GeometryUtils.triangles_overlap(pts, existing_tri):
                    overlaps = True
                    break

            if not overlaps:
                triangles.append(pts)

        if len(triangles) < 2:
            return False

        x, y = circle_center
        draw.ellipse([x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius], outline=None, fill=128)
        for t in triangles:
            draw.polygon(t, outline=None, fill=0)
        arrow_success, has_arrow = self.shape_gen.add_random_arrow(draw, w, h, circle_center, circle_radius, force_arrow=force_arrow)
        if not arrow_success:
            return False

        return True, None, has_arrow

    def generate_negative_triangle_out(self, w, h, draw, force_arrow):
        """Génère un cercle avec un triangle à l'extérieur (négatif)."""
        # Circle + Triangle outside
        circle_center, circle_radius = self.shape_gen.generate_random_circle(w, h)
        if not circle_center:
            return False

        triangle_points = None
        for _ in range(50):
            pts, _ = self.shape_gen.generate_random_triangle(w, h)
            # Check if triangle is valid (not clipped) and outside circle
            if pts is not None and GeometryUtils.is_triangle_outside_circle(pts, circle_center, circle_radius):
                triangle_points = pts
                break

        if not triangle_points:
            return False

        x, y = circle_center
        draw.ellipse([x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius], outline=None, fill=128)
        draw.polygon(triangle_points, outline=None, fill=0)
        arrow_success, has_arrow = self.shape_gen.add_random_arrow(draw, w, h, circle_center, circle_radius, force_arrow=force_arrow)
        if not arrow_success:
            return False

        return True, None, has_arrow

    def generate_negative_multiple_circles(self, w, h, draw, force_arrow):
        """Génère plusieurs cercles (2 à 4) sans triangles (négatif)."""
        # Several circles (2 to 4)
        circles = []
        attempts = 0
        target_count = random.randint(2, 4)

        while len(circles) < target_count and attempts < 50:
            c, r = self.shape_gen.generate_random_circle(w, h, circles)
            if c:
                circles.append((c, r))
            attempts += 1

        if len(circles) < 2:
            return False

        for (x, y), r in circles:
            draw.ellipse([x-r, y-r, x+r, y+r], outline=None, fill=128)

        target_center, target_radius = random.choice(circles)
        arrow_success, has_arrow = self.shape_gen.add_random_arrow(draw, w, h, target_center, target_radius, force_arrow=force_arrow)
        if not arrow_success:
            return False

        return True, None, has_arrow

    def generate_negative_disjoint(self, w, h, draw, force_arrow):
        """Génère plusieurs cercles et triangles, tous séparés (négatif)."""
        # Multiple circles and multiple triangles, never inside
        # 2-3 circles, 2-3 triangles
        circles = []
        attempts = 0
        target_circles = random.randint(2, 3)
        while len(circles) < target_circles and attempts < 50:
            c, r = self.shape_gen.generate_random_circle(w, h, circles)
            if c:
                circles.append((c, r))
            attempts += 1

        if not circles:
            return False

        triangles = []
        attempts = 0
        target_triangles = random.randint(2, 3)
        while len(triangles) < target_triangles and attempts < 50:
            pts, _ = self.shape_gen.generate_random_triangle(w, h)

            # Check if triangle is valid (not clipped)
            if pts is None:
                attempts += 1
                continue

            # Check against ALL circles
            safe = True
            for c, r in circles:
                if not GeometryUtils.is_triangle_outside_circle(pts, c, r):
                    safe = False
                    break

            # Check against existing triangles (no overlap)
            if safe:
                for existing_tri in triangles:
                    if GeometryUtils.triangles_overlap(pts, existing_tri):
                        safe = False
                        break

            if safe:
                triangles.append(pts)
            attempts += 1

        if not triangles:
            return False  # Require at least one for the "and triangles" part?

        for (x, y), r in circles:
            draw.ellipse([x-r, y-r, x+r, y+r], outline=None, fill=128)
        for t in triangles:
            draw.polygon(t, outline=None, fill=0)

        target_center, target_radius = random.choice(circles)
        arrow_success, has_arrow = self.shape_gen.add_random_arrow(draw, w, h, target_center, target_radius, force_arrow=force_arrow)
        if not arrow_success:
            return False

        return True, None, has_arrow

    def generate_negative_empty(self, w, h, draw, force_arrow):
        """Génère un fond vide (négatif)."""
        # Empty background
        arrow_success, has_arrow = self.shape_gen.add_random_arrow(draw, w, h, force_arrow=force_arrow)
        if not arrow_success:
            return False
        return True, None, has_arrow
