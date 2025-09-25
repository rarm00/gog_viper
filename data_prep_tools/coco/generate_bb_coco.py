import cv2
import numpy as np
import os
import json
from pycocotools import mask as maskUtils

def create_coco_annotations(screenshot_dir, template_dir, output_json):
    # Prepare COCO format data
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    category_id = 1
    categories = {}
    
    # Get templates
    templates = [f for f in os.listdir(template_dir) if f.endswith(('.png', '.jpg'))]
    
    # Loop through each template and assign category IDs
    for template_name in templates:
        categories[template_name] = category_id
        coco_format["categories"].append({
            "id": category_id,
            "name": template_name.split('.')[0]  # Using file name as category name
        })
        category_id += 1

    annotation_id = 1
    image_id = 1

    # Process each screenshot
    for screenshot_name in os.listdir(screenshot_dir):
        if screenshot_name.endswith(('.png', '.jpg')):
            screenshot_path = os.path.join(screenshot_dir, screenshot_name)
            image = cv2.imread(screenshot_path)
            height, width, _ = image.shape
            
            # Add image info
            coco_format["images"].append({
                "id": image_id,
                "file_name": screenshot_name,
                "height": height,
                "width": width
            })

            # Loop through templates to find matches
            for template_name in templates:
                template_path = os.path.join(template_dir, template_name)
                template = cv2.imread(template_path)

                # Perform template matching
                result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8  # Adjust this threshold as needed
                locations = np.where(result >= threshold)

                # Create annotations for each found match
                for pt in zip(*locations[::-1]):  # Switch columns and rows
                    x, y = pt
                    w, h = template.shape[1], template.shape[0]
                    
                    # Add annotation
                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": categories[template_name],
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    annotation_id += 1
            
            image_id += 1

    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(coco_format, f)

# Example usage
create_coco_annotations('path_to_screenshots', 'path_to_templates', 'annotations.json')
