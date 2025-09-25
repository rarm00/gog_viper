import json
import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def labelme_to_mask(json_file, output_path, class_mapping):
    """
    Convert LabelMe JSON to segmentation mask PNG
    
    Parameters:
    json_file: Path to LabelMe JSON file
    output_path: Where to save the mask PNG
    class_mapping: Dictionary mapping class names to pixel values
    """
    # Read JSON file
    with open(json_file) as f:
        data = json.load(f)
    
    # Create blank mask
    img_height = data['imageHeight']
    img_width = data['imageWidth']
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Draw each shape
    for shape in data['shapes']:
        points = shape['points']
        label = shape['label']
        if label in class_mapping:
            # Convert points to format required by PIL
            points = [tuple(point) for point in points]
            
            # Create mask for this shape
            img = Image.new('L', (img_width, img_height), 0)
            ImageDraw.Draw(img).polygon(points, outline=class_mapping[label], fill=class_mapping[label])
            shape_mask = np.array(img)
            
            # Add to main mask
            mask[shape_mask > 0] = class_mapping[label]
    
    # Save mask
    Image.fromarray(mask).save(output_path)

# Example usage:
class_mapping = {
    'navigation_button': 1,
    'resource_indicator': 2,
    'building': 3,
    'menu_icon': 4,
    'status_indicator': 5,
    'text_area': 6,
    'interactive_zone': 7
}

# Convert all JSON files in a directory
input_dir = 'train_data/gog_train/labels2'
output_dir = 'train_data/gog_train/images'
os.makedirs(output_dir, exist_ok=True)

for json_file in os.listdir(input_dir):
    if json_file.endswith('.json'):
        base_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(input_dir, json_file)
        mask_path = os.path.join(output_dir, f'{base_name}_P.png')
        labelme_to_mask(json_path, mask_path, class_mapping)

# Create codes.txt
with open('train_data/gog_train/codes.txt', 'w') as f:
    f.write('background\n')  # Class 0
    for class_name in class_mapping:
        f.write(f'{class_name}\n')