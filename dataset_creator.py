from pathlib import Path
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple
import os

class DatasetCreator:
    """Helper class for creating and managing the UI element detection dataset"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "images"
        self.annotations_path = self.base_path / "annotations"
        self.categories_path = self.base_path / "categories.json"
        
        # Create necessary directories
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.annotations_path.mkdir(parents=True, exist_ok=True)
        
        # Define UI element categories
        self.categories = {
            "alliance": 1,
            "inventory": 2,
            "mail": 3,
            "kingdom": 4,
            "guards": 5,
            "develop": 6,
            "event_center": 7,
            "gifts": 8,
            "vip": 9,
            "resources": 10
            # Add more categories as needed
        }
        
        # Save categories
        with open(self.categories_path, 'w') as f:
            json.dump(self.categories, f, indent=2)

    def capture_and_save_screenshot(self, window_title: str) -> str:
        """Capture game window screenshot and save it"""
        # Implementation depends on your window capture method
        pass

    def annotate_image(self, image_path: str) -> Dict:
        """
        Interactive tool to annotate UI elements in an image.
        Returns annotation data in COCO format.
        """
        image = cv2.imread(str(image_path))
        annotations = []
        
        def draw_rectangle(event, x, y, flags, param):
            nonlocal drawing, start_x, start_y, image_copy
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_x, start_y = x, y
                image_copy = image.copy()
            
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp = image_copy.copy()
                cv2.rectangle(temp, (start_x, start_y), (x, y), (0, 255, 0), 2)
                cv2.imshow('Annotation Tool', temp)
            
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                end_x, end_y = x, y
                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                
                # Get category from user
                print("\nAvailable categories:")
                for name, id in self.categories.items():
                    print(f"{id}: {name}")
                
                category_id = int(input("Enter category ID: "))
                
                # Save annotation
                width = abs(end_x - start_x)
                height = abs(end_y - start_y)
                x = min(start_x, end_x)
                y = min(start_y, end_y)
                
                annotation = {
                    'bbox': [x, y, width, height],
                    'category_id': category_id,
                    'area': width * height,
                    'iscrowd': 0
                }
                annotations.append(annotation)
        
        # Set up annotation window
        cv2.namedWindow('Annotation Tool')
        cv2.setMouseCallback('Annotation Tool', draw_rectangle)
        
        drawing = False
        start_x = start_y = -1
        image_copy = image.copy()
        
        while True:
            cv2.imshow('Annotation Tool', image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('c'):  # Clear last annotation
                if annotations:
                    annotations.pop()
                    image = cv2.imread(str(image_path))
                    for ann in annotations:
                        x, y, w, h = ann['bbox']
                        cv2.rectangle(image, (int(x), int(y)), 
                                    (int(x + w), int(y + h)), (0, 255, 0), 2)
        
        cv2.destroyAllWindows()
        return annotations

    def create_dataset(self, num_samples: int = 10):
        """
        Create a dataset by capturing screenshots and annotating them.
        """
        for i in range(num_samples):
            # Capture screenshot
            image_path = self.capture_and_save_screenshot(f"sample_{i}.png")
            
            # Annotate image
            annotations = self.annotate_image(image_path)
            
            # Save annotations
            annotation_path = self.annotations_path / f"sample_{i}.json"
            with open(annotation_path, 'w') as f:
                json.dump(annotations, f, indent=2)
            
            print(f"Saved annotations for sample_{i}")

# Usage example:
if __name__ == "__main__":
    dataset_creator = DatasetCreator("game_ui_dataset")
    dataset_creator.create_dataset(num_samples=10)