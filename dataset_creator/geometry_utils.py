"""
geometry_utils.py - Utilitaires géométriques pour la génération de dataset

Contient toutes les fonctions de calcul géométrique :
- Détection de points dans un cercle
- Détection de triangles dans/hors d'un cercle
- Détection d'intersection cercle-segment
- Détection de chevauchement entre cercles et triangles
"""
import math


class GeometryUtils:
    @staticmethod
    def is_point_in_circle(point, circle_center, circle_radius):
        """Vérifie si un point (x, y) est à l'intérieur d'un cercle."""
        dist = math.hypot(point[0] - circle_center[0], point[1] - circle_center[1])
        return dist <= circle_radius

    @staticmethod
    def is_triangle_in_circle(triangle_points, circle_center, circle_radius):
        """Vérifie si un triangle est complètement à l'intérieur d'un cercle."""
        for point in triangle_points:
            if not GeometryUtils.is_point_in_circle(point, circle_center, circle_radius):
                return False
        return True

    @staticmethod
    def is_triangle_outside_circle(triangle_points, circle_center, circle_radius):
        """
        Vérifie si un triangle est complètement à l'extérieur d'un cercle.
        1. Tous les sommets doivent être à l'extérieur.
        2. Aucun côté ne doit intersecter le cercle.
        """
        # 1. Vérifier les sommets
        for point in triangle_points:
            if GeometryUtils.is_point_in_circle(point, circle_center, circle_radius):
                return False

        # 2. Vérifier le chevauchement avec les côtés
        for i in range(3):
            p1 = triangle_points[i]
            p2 = triangle_points[(i + 1) % 3]
            if GeometryUtils.circle_intersects_segment(circle_center, circle_radius, p1, p2):
                return False

        return True

    @staticmethod
    def is_circle_overlap(c1, r1, c2, r2):
        dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        return dist < (r1 + r2)

    @staticmethod
    def circle_intersects_segment(center, radius, p1, p2):
        """Vérifie si un cercle intersecte un segment de droite p1-p2."""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        if dx == 0 and dy == 0:
            return False

        t = ((center[0] - p1[0]) * dx + (center[1] - p1[1]) * dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t))
        closest_x = p1[0] + t * dx
        closest_y = p1[1] + t * dy

        dist = math.hypot(center[0] - closest_x, center[1] - closest_y)
        return dist <= radius

    @staticmethod
    def triangles_overlap(tri1, tri2):
        """Vérifie si deux triangles se chevauchent en utilisant le théorème de l'axe séparateur (SAT)."""
        def get_edges(triangle):
            """Récupère les côtés (sous forme de vecteurs) d'un triangle."""
            edges = []
            for i in range(3):
                p1 = triangle[i]
                p2 = triangle[(i + 1) % 3]
                edges.append((p2[0] - p1[0], p2[1] - p1[1]))
            return edges

        def get_perpendicular(edge):
            """Récupère le vecteur perpendiculaire (normale) à un côté."""
            return (-edge[1], edge[0])

        def project_triangle(triangle, axis):
            """Projette tous les points du triangle sur l'axe et retourne le min/max."""
            projections = []
            for point in triangle:
                # Produit scalaire
                proj = point[0] * axis[0] + point[1] * axis[1]
                projections.append(proj)
            return min(projections), max(projections)

        def projections_overlap(proj1, proj2):
            """Vérifie si deux projections 1D se chevauchent."""
            min1, max1 = proj1
            min2, max2 = proj2
            return not (max1 < min2 or max2 < min1)

        # Récupérer tous les côtés des deux triangles
        edges1 = get_edges(tri1)
        edges2 = get_edges(tri2)

        # Tester tous les axes (perpendiculaires aux côtés)
        all_edges = edges1 + edges2
        for edge in all_edges:
            axis = get_perpendicular(edge)

            # Normaliser l'axe (optionnel mais aide à la stabilité numérique)
            length = math.sqrt(axis[0]**2 + axis[1]**2)
            if length < 1e-10:
                continue
            axis = (axis[0] / length, axis[1] / length)

            # Projeter les deux triangles sur cet axe
            proj1 = project_triangle(tri1, axis)
            proj2 = project_triangle(tri2, axis)

            # Si les projections ne se chevauchent pas sur un axe, les triangles ne se chevauchent pas
            if not projections_overlap(proj1, proj2):
                return False

        # Si tous les axes ont des projections qui se chevauchent, les triangles se chevauchent
        return True
