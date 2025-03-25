import os

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Define paths (update these if needed)
base_dir = "/home/m1h1r/Documents/[2] dev/influencer-net"
ssd_dir = "/run/media/m1h1r/04E884E1E884D1FA"
image_dir = os.path.join(base_dir, "image")
mapping_file = os.path.join(base_dir, "smallInfluencers.txt")
output_base = os.path.join(ssd_dir, "preprocessedImages")

# Define the target image size for EfficientNetV2‑S (224×224)
target_size = (224, 224)


def load_mapping(mapping_path):
    """
    Reads the influencer mapping file and returns a dictionary
    mapping username to category.
    """
    mapping = {}
    with open(mapping_path, "r") as f:
        for line in f:
            # Each line is expected to be: username<TAB>category<SPACE>...
            parts = line.strip().split()
            if len(parts) >= 2:
                username = parts[0]
                category = parts[1]
                mapping[username] = category
    return mapping


def process_and_save_image(image_path, username, category):
    """
    Load the image, resize, preprocess, and save as a .npy file
    in the appropriate category folder.
    """
    try:
        # Open image and convert to RGB
        img = Image.open(image_path).convert("RGB")
        # Resize image to target size
        img = img.resize(target_size)
        # Convert image to numpy array (float32)
        img_array = np.array(img).astype(np.float32)
        # Expand dims to simulate batch of 1 (if required by preprocess_input)
        img_batch = np.expand_dims(img_array, axis=0)
        # Preprocess image using EfficientNetV2's preprocess_input (scales to [-1, 1])
        preprocessed = preprocess_input(img_batch)
        # Remove the batch dimension
        preprocessed = preprocessed[0]
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return

    # Create output directory for the category if it doesn't exist
    category_dir = os.path.join(output_base, category)
    os.makedirs(category_dir, exist_ok=True)

    # Save the preprocessed image as a .npy file using the original filename
    filename = os.path.basename(image_path)
    output_path = os.path.join(category_dir, os.path.splitext(filename)[0] + ".npy")
    with open(output_path, "wb") as f:
        np.save(f, preprocessed)
    print(f"Saved preprocessed image to: {output_path}")


def main():
    # Load influencer to category mapping
    mapping = load_mapping(mapping_file)

    # Process each image in the image directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            # The influencer username is the part before the dash '-'
            if "-" not in filename:
                print(f"Filename {filename} does not have the expected '-' separator.")
                continue
            username = filename.split("-")[0]
            # Get the corresponding category from the mapping file
            if username not in mapping:
                print(f"Username {username} not found in mapping. Skipping {filename}.")
                continue
            category = mapping[username]
            image_path = os.path.join(image_dir, filename)
            process_and_save_image(image_path, username, category)


if __name__ == "__main__":
    main()
