import cv2
import numpy as np
import os
import json
import shutil

def resize_image(image, scale_ratio):
    """ Resize image by a given scale ratio. """
    new_w = int(image.shape[1] * scale_ratio)
    new_h = int(image.shape[0] * scale_ratio)
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image

def convert_to_native(data):
    """ Recursively convert NumPy data types to native Python types. """
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy arrays to lists
    elif isinstance(data, np.int64):
        return int(data)  # Convert numpy int64 to native Python int
    elif isinstance(data, np.float64):
        return float(data)  # Convert numpy float64 to native Python float
    elif isinstance(data, dict):
        return {key: convert_to_native(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native(item) for item in data]
    return data  # For other data types, return them unchanged

def create_coco_annotations(screenshot_dir, template_dir, output_json, annotated_dir, target_width=256, target_height=144):
    # Prepare COCO format data
    coco_format = {
        "categories": [],
        "images": [],
        "annotations": []
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

    # Calculate the resize ratio for 1920x1080 -> target width/height (e.g., 256x144)
    scale_width = target_width / 1920
    scale_height = target_height / 1080
    scale_ratio = min(scale_width, scale_height)  # Use the smaller ratio to preserve aspect ratio

    # Process each screenshot
    for screenshot_name in os.listdir(screenshot_dir):
        if screenshot_name.endswith(('.png', '.jpg')):
            screenshot_path = os.path.join(screenshot_dir, screenshot_name)
            image = cv2.imread(screenshot_path)

            # Resize the screenshot using the scale ratio
            resized_image = resize_image(image, scale_ratio)

            height, width, _ = resized_image.shape
            
            # Create a new filename for the resized image
            new_filename = f"{screenshot_count:05d}.png"
            annotated_path = os.path.join(annotated_dir, new_filename)
            
            # Save the resized screenshot to the new directory with the new name
            cv2.imwrite(annotated_path, resized_image)

            # Add image info
            coco_format["images"].append({
                "id": image_id,
                "file_name": new_filename
            })

            # Loop through templates to find matches
            for template_name in templates:
                template_path = os.path.join(template_dir, template_name)
                template = cv2.imread(template_path)

                # Resize the template using the same scale ratio
                resized_template = resize_image(template, scale_ratio)

                # Perform template matching
                result = cv2.matchTemplate(resized_image, resized_template, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8  # Adjust this threshold as needed
                locations = np.where(result >= threshold)

                # Create annotations for each found match
                for pt in zip(*locations[::-1]):  # Switch columns and rows
                    x, y = pt
                    h, w = resized_template.shape[:2]

                    # Create bbox [x_min, y_min, width, height]
                    bbox = [x, y, w, h]

                    # Add annotation
                    coco_format["annotations"].append({
                        #"id": annotation_id,
                        "image_id": image_id,
                        "bbox": bbox,
                        "category_id": categories[template_name]
                    })
                    annotation_id += 1

            image_id += 1
            screenshot_count += 1

    # Convert the data to native Python types to avoid serialization issues
    coco_format = convert_to_native(coco_format)

    # Save to JSON file
    with open(output_json, 'w') as f:
        json.dump(coco_format, f)

# Example usage
screenshot_dir = 'data/gog_dataset/source_images'
template_dir = 'data/gog_dataset/templates'
annotation_file = 'data/gog_dataset/annotated_coco_v2.json'  # Correct file extension
annotated_dir = 'data/gog_dataset/annotated_images'
create_coco_annotations(screenshot_dir, template_dir, annotation_file, annotated_dir)
