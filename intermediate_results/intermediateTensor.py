import os

import numpy as np
import tensorflow as tf
from PIL import Image

# Paths defined
base_dir = "/home/m1h1r/Documents/[2] dev/influencer-net"
ssd_dir = "/run/media/m1h1r/04E884E1E884D1FA"
image_dir = os.path.join(base_dir, "testImage")
mapping_file = os.path.join(base_dir, "data_sampling/smallInfluencers.txt")
output_base = os.path.join(ssd_dir, "debugTensor")

# Define the target image size for EfficientNetV2‑S (224×224)
target_size = (224, 224)


def load_mapping(mapping_path):
    """Loads a mapping of usernames to categories."""
    mapping = {}
    with open(mapping_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                username = parts[0]
                category = parts[1]
                mapping[username] = category
    return mapping


def preprocess_for_train(image, target_size, debug_dir, filename):
    """Applies training augmentations and saves intermediate steps."""

    # Ensure tensor format
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Save the original image
    np.savez_compressed(
        os.path.join(debug_dir, f"{filename}_1_original.npz"), image.numpy()
    )

    # Step 1: Random Crop
    shape = tf.shape(image)
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(0.05, 1.0),
        min_object_covered=0,
        use_image_if_no_bounding_boxes=True,
    )
    cropped = tf.slice(image, begin, size)
    np.savez_compressed(
        os.path.join(debug_dir, f"{filename}_2_cropped.npz"), cropped.numpy()
    )

    # Step 2: Resize to Target Size
    resized = tf.image.resize(cropped, target_size)
    np.savez_compressed(
        os.path.join(debug_dir, f"{filename}_3_resized.npz"), resized.numpy()
    )

    # Step 3: Random Horizontal Flip
    flipped = tf.image.random_flip_left_right(resized)
    np.savez_compressed(
        os.path.join(debug_dir, f"{filename}_4_flipped.npz"), flipped.numpy()
    )

    # Step 4: Normalize (Convert to [-1, 1] range)
    normalized = (flipped - 128.0) / 128.0
    np.savez_compressed(
        os.path.join(debug_dir, f"{filename}_5_normalized.npz"), normalized.numpy()
    )

    return normalized


def process_and_save_image(image_path, username, category):
    """Processes an image using training augmentation and saves intermediate steps."""
    try:
        # Load image using PIL and convert to RGB
        img = Image.open(image_path).convert("RGB")

        # Resize to a larger size (to allow cropping to 224x224 later)
        img = img.resize((256, 256))
        img_array = np.array(img).astype(np.float32)

        # Debug folder for intermediate results
        debug_dir = os.path.join(output_base, "debug", category)
        os.makedirs(debug_dir, exist_ok=True)

        # Apply training preprocessing & save intermediate steps
        preprocessed_tensor = preprocess_for_train(
            img_array, target_size, debug_dir, os.path.basename(image_path)
        )

        # Convert back to numpy array for saving
        preprocessed = preprocessed_tensor.numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return

    # Create output directory for the category if it doesn't exist
    category_dir = os.path.join(output_base, category)
    os.makedirs(category_dir, exist_ok=True)

    # Save the final preprocessed image
    filename = os.path.basename(image_path)
    output_path = os.path.join(category_dir, os.path.splitext(filename)[0] + ".npz")
    try:
        np.savez_compressed(output_path, preprocessed)
        print(f"Saved compressed preprocessed image to: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")


def main():
    # load mapping
    mapping = load_mapping(mapping_file)

    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            if "-" not in filename:
                print(f"Filename {filename} does not have the expected '-' separator.")
                continue

            username = filename.split("-")[0]
            if username not in mapping:
                print(f"Username {username} not found in mapping. Skipping {filename}.")
                continue

            category = mapping[username]
            image_path = os.path.join(image_dir, filename)
            process_and_save_image(image_path, username, category)


if __name__ == "__main__":
    main()
