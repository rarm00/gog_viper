import os
import shutil

# Define paths
images_dir = 'train_data/gog_train_v2/images'
labels_dir = 'train_data/gog_train_v2/labels'
output_dir = 'train_data/gog_train_v2/filtered_images'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through images and copy only if corresponding label exists
for image_file in os.listdir(images_dir):
    base_name = os.path.splitext(image_file)[0]
    label_file = f"{base_name}_P.png"
    
    if os.path.isfile(os.path.join(labels_dir, label_file)):
        shutil.copy(os.path.join(images_dir, image_file), output_dir)

print("Filtered images copied successfully.")
