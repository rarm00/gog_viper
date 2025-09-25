import json
import os
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict

def get_unique_labels(json_dir):
    """Extract all unique labels from JSON files"""
    unique_labels = set()
    
    # Scan all JSON files for labels
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            with open(os.path.join(json_dir, json_file)) as f:
                data = json.load(f)
                for shape in data['shapes']:
                    unique_labels.add(shape['label'])
    
    # Create mapping with incremental values starting from 1
    # (0 is reserved for background)
    return {label: idx + 1 for idx, label in enumerate(sorted(unique_labels))}

def labelme_to_mask(json_file, output_path, class_mapping):
    """Convert LabelMe JSON to segmentation mask PNG"""
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
            
            # Handle different shape types
            draw = ImageDraw.Draw(img)
            if shape['shape_type'] == 'polygon':
                draw.polygon(points, outline=class_mapping[label], fill=class_mapping[label])
            elif shape['shape_type'] == 'rectangle':
                # Rectangle needs top-left and bottom-right points
                draw.rectangle([tuple(points[0]), tuple(points[1])], 
                             outline=class_mapping[label], 
                             fill=class_mapping[label])
            
            shape_mask = np.array(img)
            
            # Add to main mask
            mask[shape_mask > 0] = class_mapping[label]
    
    # Save mask
    Image.fromarray(mask).save(output_path)

def process_labelme_dataset(input_dir, output_dir):
    """Process entire dataset"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get class mapping from JSON files
    class_mapping = get_unique_labels(input_dir)
    print("Found classes:", class_mapping)
    
    # Convert all JSON files to masks
    for json_file in os.listdir(input_dir):
        if json_file.endswith('.json'):
            base_name = os.path.splitext(json_file)[0]
            json_path = os.path.join(input_dir, json_file)
            mask_path = os.path.join(output_dir, f'{base_name}_P.png')
            labelme_to_mask(json_path, mask_path, class_mapping)
            print(f"Processed {json_file}")
    
    # Create codes.txt
    with open(os.path.join(output_dir, 'codes.txt'), 'w') as f:
        f.write('background\n')  # Class 0
        for label in sorted(class_mapping.keys()):
            f.write(f'{label}\n')
    
    print(f"\nProcessed files saved to: {output_dir}")
    print(f"Created codes.txt with {len(class_mapping) + 1} classes")

# Usage
input_dir = 'train_data/gog_train/labels2'  # Directory containing your JSON files
output_dir = 'train_data/gog_train/images'  # Directory where masks and codes.txt will be saved

process_labelme_dataset(input_dir, output_dir)