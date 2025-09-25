import json
import os
import numpy as np
from PIL import Image, ImageDraw

def build_class_mapping(json_directory):
    """Scan all JSON files and build class mapping from unique labels"""
    unique_labels = set()
    
    # Scan all JSON files for labels
    for json_file in os.listdir(json_directory):
        if json_file.endswith('.json'):
            with open(os.path.join(json_directory, json_file)) as f:
                data = json.load(f)
                for shape in data['shapes']:
                    unique_labels.add(shape['label'])
    
    # Create mapping with incremental values starting from 1 (0 is reserved for background)
    class_mapping = {label: idx + 1 for idx, label in enumerate(sorted(unique_labels))}
    
    print("Found labels:", list(class_mapping.keys()))
    print("Total unique labels:", len(class_mapping))
    
    return class_mapping

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
        shape_type = shape['shape_type']
        
        if label in class_mapping:
            # Convert points to format required by PIL
            points = [tuple(point) for point in points]
            
            # Create mask for this shape
            img = Image.new('L', (img_width, img_height), 0)
            draw = ImageDraw.Draw(img)
            
            if shape_type == 'polygon':
                draw.polygon(points, outline=class_mapping[label], fill=class_mapping[label])
            elif shape_type == 'rectangle':
                # # For rectangles, we need the top-left and bottom-right points
                # if len(points) == 2:
                #     top_left = points[0]
                #     bottom_right = points[1]
                #     draw.rectangle([top_left, bottom_right], 
                #                  outline=class_mapping[label], 
                #                  fill=class_mapping[label])
                # For rectangles, we need the top-left and bottom-right points
                if len(points) == 2:
                    # Ensure top_left is the point with minimum coordinates
                    x0, y0 = points[0]
                    x1, y1 = points[1]
                    top_left = (min(x0, x1), min(y0, y1))
                    bottom_right = (max(x0, x1), max(y0, y1))
                    
                    draw.rectangle([top_left, bottom_right], 
                                outline=class_mapping[label], 
                                fill=class_mapping[label])
            
            shape_mask = np.array(img)
            # Add to main mask
            mask[shape_mask > 0] = class_mapping[label]
    
    # Save mask
    Image.fromarray(mask).save(output_path)

# Directory setup
input_dir = 'train_data/gog_train_v3/labels2'  # Directory containing your JSON files
output_dir = 'train_data/gog_train_v3/images2'  # Directory where masks and codes.txt will be saved
os.makedirs(output_dir, exist_ok=True)

# Build class mapping from your JSON files
class_mapping = build_class_mapping(input_dir)

# Save codes.txt
with open(os.path.join(output_dir, 'codes.txt'), 'w') as f:
    f.write('background\n')  # Class 0
    for label in sorted(class_mapping.keys()):
        f.write(f'{label}\n')

# Convert all JSON files to masks
for json_file in os.listdir(input_dir):
    if json_file.endswith('.json'):
        base_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(input_dir, json_file)
        mask_path = os.path.join(output_dir, f'{base_name}_P.png')
        labelme_to_mask(json_path, mask_path, class_mapping)
        print(f"Processed {json_file}")