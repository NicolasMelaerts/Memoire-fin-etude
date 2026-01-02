import os

class Config:
    # General settings
    NUM_IMAGES = 1000
    IMAGE_SIZE = (128, 128)  # (width, height)
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'generated_dataset')
    
    # Shape settings
    MIN_CIRCLE_RADIUS = 10
    MAX_CIRCLE_RADIUS = 40
    MIN_TRIANGLE_SIDE = 10
    MAX_TRIANGLE_SIDE = 30
    
    # Class balance
    RATIO_INSIDE = 0.5  # 50% Inside, 50% Outside