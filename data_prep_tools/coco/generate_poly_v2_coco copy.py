import cv2
import numpy as np
import os
import json
import shutil

def create_coco_annotations(screenshot_dir, template_dir, output_json, annotated_dir):
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
    screenshot_count = 1

    # Create the annotated images directory if it doesn't exist
    os.makedirs(annotated_dir, exist_ok=True)

    # Process each screenshot
    for screenshot_name in os.listdir(screenshot_dir):
        if screenshot_name.endswith(('.png', '.jpg')):
            screenshot_path = os.path.join(screenshot_dir, screenshot_name)
            image = cv2.imread(screenshot_path)
            height, width, _ = image.shape
            
            # Create a new filename for the annotated image
            new_filename = f"{screenshot_count:05d}.png"
            annotated_path = os.path.join(annotated_dir, new_filename)
            
            # Save the screenshot to the new directory with the new name
            shutil.copy(screenshot_path, annotated_path)

            # Add image info
            coco_format["images"].append({
                "id": image_id,
                "file_name": new_filename,
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
                    h, w = template.shape[:2]

                    # Create a mask for the template
                    mask = np.zeros_like(image, dtype=np.uint8)
                    mask[y:y+h, x:x+w] = template
                    
                    # Find contours of the mask
                    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        # Approximate the contour to reduce the number of points
                        epsilon = 0.01 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)

                        # Convert contour points to the required format
                        segmentation = approx.flatten().tolist()
                        
                        # Add annotation
                        coco_format["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": categories[template_name],
                            "segmentation": [segmentation],
                            "area": cv2.contourArea(contour),
                            "iscrowd": 0
                        })
                        annotation_id += 1
            
            image_id += 1
            screenshot_count += 1

    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(coco_format, f)

# Example usage
screenshot_dir = 'data/gog_dataset/source_images'
template_dir = 'data/gog_dataset/templates'
annotation_file = 'data/gog_dataset/annotated_coco_v2.txt'
annotated_dir = 'data/gog_dataset/annotated_images'
create_coco_annotations(screenshot_dir, template_dir, annotation_file, annotated_dir)