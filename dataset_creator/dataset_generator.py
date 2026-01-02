import numpy as np
from PIL import Image, ImageDraw
import os
import random
import math
import json
import csv
from config import Config

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

class DatasetGenerator:
    def __init__(self, config):
        self.config = config
        self.images_dir = os.path.join(config.OUTPUT_DIR, 'images')
        
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

    def generate_random_circle(self, img_w, img_h):
        # Radius
        radius = random.randint(self.config.MIN_CIRCLE_RADIUS, self.config.MAX_CIRCLE_RADIUS)
        # Center (ensure fully within bounds if possible, but minimal overlap is fine too)
        # To avoid clipping, keep center at least radius away from edges.
        x = random.randint(radius, img_w - radius)
        y = random.randint(radius, img_h - radius)
        return (x, y), radius

    def generate_random_triangle(self, img_w, img_h):
        center_x = random.randint(0, img_w)
        center_y = random.randint(0, img_h)
        
        side_length = random.randint(self.config.MIN_TRIANGLE_SIDE, self.config.MAX_TRIANGLE_SIDE)
        # Distance from centroid to vertex for equilateral triangle
        dist = side_length / math.sqrt(3)
        
        angle_offset = random.uniform(0, 2 * math.pi)
        points = []
        for i in range(3):
            angle = angle_offset + i * (2 * math.pi / 3)
            pt_x = int(center_x + dist * math.cos(angle))
            pt_y = int(center_y + dist * math.sin(angle))
            # Clip
            pt_x = max(0, min(img_w - 1, pt_x))
            pt_y = max(0, min(img_h - 1, pt_y))
            points.append((pt_x, pt_y))
        return points

    def generate_sample(self, sample_id, label):
        w, h = self.config.IMAGE_SIZE
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts:
            # Create white image
            img = Image.new('L', (w, h), 255) # 'L' = 8-bit pixels, black and white
            draw = ImageDraw.Draw(img)
            
            # 1. Generate Circle
            circle_center, circle_radius = self.generate_random_circle(w, h)
            
            # 2. Generate Triangle candidates
            triangle_points = self.generate_random_triangle(w, h)
            
            if label == 'inside':
                # Rejection sampling for 'inside' is hard if random.
                # Heuristic: generate triangle INSIDE circle directly.
                if not GeometryUtils.is_triangle_in_circle(triangle_points, circle_center, circle_radius):
                     # Try to force generation inside
                    # For equilateral, we need to be careful.
                    # Generate center inside circle with margin
                    margin = (self.config.MAX_TRIANGLE_SIDE / math.sqrt(3)) + 2
                    if circle_radius > margin:
                         # Generate random center within allowed radius
                        max_dist = circle_radius - margin
                        rx = random.uniform(0, max_dist)
                        theta = random.uniform(0, 2*math.pi)
                        cx = circle_center[0] + rx * math.cos(theta)
                        cy = circle_center[1] + rx * math.sin(theta)
                        
                        # Generate equilateral points around this center
                        side_length = random.randint(self.config.MIN_TRIANGLE_SIDE, self.config.MAX_TRIANGLE_SIDE)
                        # Ensure side length fits (approximate check, could be better but let's rely on rejection for strictness)
                        # We used a safe margin based on MAX side, so current side should fit if we regenerate points
                        dist = side_length / math.sqrt(3)
                        angle_offset = random.uniform(0, 2 * math.pi)
                        points = []
                        for i in range(3):
                            angle = angle_offset + i * (2 * math.pi / 3)
                            pt_x = int(cx + dist * math.cos(angle))
                            pt_y = int(cy + dist * math.sin(angle))
                            points.append((pt_x, pt_y))
                        triangle_points = points
                    
                    if not GeometryUtils.is_triangle_in_circle(triangle_points, circle_center, circle_radius):
                        attempts += 1
                        continue
            
            elif label == 'outside':
                if not GeometryUtils.is_triangle_outside_circle(triangle_points, circle_center, circle_radius):
                    attempts += 1
                    continue
            
            # Draw shapes
            # Circle: Filled Gray (128)
            x, y = circle_center
            r = circle_radius
            draw.ellipse([x-r, y-r, x+r, y+r], outline=None, fill=128)
            
            # Triangle: Filled Black (0)
            draw.polygon(triangle_points, outline=None, fill=0)

            filename = f"image_{sample_id:04d}.png"
            filepath = os.path.join(self.images_dir, filename)
            img.save(filepath)
            
            return filename, label
        
        return None, None

    def run(self):
        annotations = []
        num_inside = int(self.config.NUM_IMAGES * self.config.RATIO_INSIDE)
        num_outside = self.config.NUM_IMAGES - num_inside
        
        labels = ['inside'] * num_inside + ['outside'] * num_outside
        random.shuffle(labels)
        
        print(f"Generating {self.config.NUM_IMAGES} images...")
        
        # Clean output dir
        # ... logic if needed, but we overwrite files anyway
        
        for i, label in enumerate(labels):
            filename, lbl = self.generate_sample(i, label)
            if filename:
                annotations.append({'filename': filename, 'class': lbl})
            else:
                print(f"Failed to generate sample {i} ({label})")

        # Save annotations
        csv_path = os.path.join(self.config.OUTPUT_DIR, 'annotations.csv')
        json_path = os.path.join(self.config.OUTPUT_DIR, 'annotations.json')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'class'])
            writer.writeheader()
            writer.writerows(annotations)
            
        with open(json_path, 'w') as f:
            json.dump(annotations, f, indent=4)
            
        print(f"Dataset generated at {self.config.OUTPUT_DIR}")

if __name__ == "__main__":
    gen = DatasetGenerator(Config)
    gen.run()
