import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_images_with_masks(images_dir, labels_dir, num_samples=6):
    """Display images with corresponding masks overlaid."""
    # Get list of image filenames (only load files with a matching label)
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    matched_files = [f for f in image_files if os.path.exists(os.path.join(labels_dir, f"{os.path.splitext(f)[0]}_P.png"))]

    if not matched_files:
        print("No matching images and labels found.")
        return
    
    # Limit the number of displayed samples to num_samples
    matched_files = matched_files[:num_samples]
    
    for file_name in matched_files:
        # Load image and mask
        img_path = os.path.join(images_dir, file_name)
        mask_path = os.path.join(labels_dir, f"{os.path.splitext(file_name)[0]}_P.png")
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Load mask as grayscale

        # Convert mask to numpy for visualization with color map
        mask_np = np.array(mask)

        # Plot the image and the mask side by side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image)
        ax[0].set_title("Image")
        ax[0].axis("off")

        ax[1].imshow(image)
        ax[1].imshow(mask_np, cmap="nipy_spectral", alpha=0.5)  # Overlay mask with transparency
        ax[1].set_title("Image with Mask Overlay")
        ax[1].axis("off")

        plt.show()

# Example usage
images_dir = 'train_data/gog_train_v2/images'
labels_dir = 'train_data/gog_train_v2/labels'
show_images_with_masks(images_dir, labels_dir, num_samples=6)

test_land = 'test'
