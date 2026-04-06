"""
geometry_utils.py — Utilitaires géométriques pour la génération de dataset

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
        """Check if a point (x, y) is inside a circle."""
        dist = math.hypot(point[0] - circle_center[0], point[1] - circle_center[1])
        return dist <= circle_radius

    @staticmethod
    def is_triangle_in_circle(triangle_points, circle_center, circle_radius):
        """Check if a triangle is completely inside a circle."""
        for point in triangle_points:
            if not GeometryUtils.is_point_in_circle(point, circle_center, circle_radius):
                return False
        return True

    @staticmethod
    def is_triangle_outside_circle(triangle_points, circle_center, circle_radius):
        """
        Check if a triangle is completely outside a circle.
        1. All vertices must be outside.
        2. No edge should intersect the circle.
        """
        # 1. Check vertices
        for point in triangle_points:
            if GeometryUtils.is_point_in_circle(point, circle_center, circle_radius):
                return False

        # 2. Check overlap with edges
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
        """Check if a circle intersects with a line segment p1-p2."""
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
        """Check if two triangles overlap using Separating Axis Theorem (SAT)."""
        def get_edges(triangle):
            """Get edges (as vectors) from a triangle."""
            edges = []
            for i in range(3):
                p1 = triangle[i]
                p2 = triangle[(i + 1) % 3]
                edges.append((p2[0] - p1[0], p2[1] - p1[1]))
            return edges

        def get_perpendicular(edge):
            """Get perpendicular vector (normal) to an edge."""
            return (-edge[1], edge[0])

        def project_triangle(triangle, axis):
            """Project all points of triangle onto axis and return min/max."""
            projections = []
            for point in triangle:
                # Dot product
                proj = point[0] * axis[0] + point[1] * axis[1]
                projections.append(proj)
            return min(projections), max(projections)

        def projections_overlap(proj1, proj2):
            """Check if two 1D projections overlap."""
            min1, max1 = proj1
            min2, max2 = proj2
            return not (max1 < min2 or max2 < min1)

        # Get all edges from both triangles
        edges1 = get_edges(tri1)
        edges2 = get_edges(tri2)

        # Test all axes (perpendiculars to edges)
        all_edges = edges1 + edges2
        for edge in all_edges:
            axis = get_perpendicular(edge)

            # Normalize axis (optional but helps with numerical stability)
            length = math.sqrt(axis[0]**2 + axis[1]**2)
            if length < 1e-10:
                continue
            axis = (axis[0] / length, axis[1] / length)

            # Project both triangles onto this axis
            proj1 = project_triangle(tri1, axis)
            proj2 = project_triangle(tri2, axis)

            # If projections don't overlap on any axis, triangles don't overlap
            if not projections_overlap(proj1, proj2):
                return False

        # If all axes have overlapping projections, triangles overlap
        return True
