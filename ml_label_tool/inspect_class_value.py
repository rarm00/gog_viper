# Check the unique values in each mask to ensure they align with codes.txt.
import numpy as np
import os
from PIL import Image

labels_dir = 'train_data/gog_train_v2/labels'

for mask_name in os.listdir(labels_dir):
    mask_path = os.path.join(labels_dir, mask_name)
    mask = np.array(Image.open(mask_path))
    unique_values = np.unique(mask)
    print(f"{mask_name} unique values: {unique_values}")
