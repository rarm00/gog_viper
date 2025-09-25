from PIL import Image
import os

# Set paths
images_dir = 'train_data/gog_train_v3/images2'
labels_dir = 'train_data/gog_train_v3/labels2'
resized_images_dir = 'train_data/gog_train_v3/resized_images2'
resized_labels_dir = 'train_data/gog_train_v3/resized_labels2'

# Create output directories if they don't exist
os.makedirs(resized_images_dir, exist_ok=True)
os.makedirs(resized_labels_dir, exist_ok=True)

# Set the target size
target_size = (128, 96)  # Adjust this size as needed

# Function to resize images
def resize_image(input_dir, output_dir, target_size):
    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, file_name)
            img = Image.open(img_path)
            img = img.resize(target_size, Image.Resampling.LANCZOS)  # Updated resampling method
            img.save(os.path.join(output_dir, file_name))
            print(f"Resized {file_name}")

# Resize images and labels
resize_image(images_dir, resized_images_dir, target_size)
resize_image(labels_dir, resized_labels_dir, target_size)

print("Resizing complete.")

