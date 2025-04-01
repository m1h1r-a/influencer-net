import os

import numpy as np
import tensorflow as tf
from PIL import Image

# Paths defined
base_dir = "/home/m1h1r/Documents/[2] dev/influencer-net"
ssd_dir = "/run/media/m1h1r/04E884E1E884D1FA"
image_dir = os.path.join(base_dir, "image")
mapping_file = os.path.join(base_dir, "data_sampling/smallInfluencers.txt")
output_base = os.path.join(ssd_dir, "tfCompressedPreprocessedImages")

# Define the target image size for EfficientNetV2‑S (224×224)
target_size = (224, 224)


def load_mapping(mapping_path):
    # dictionary for username -> category
    mapping = {}
    with open(mapping_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                username = parts[0]
                category = parts[1]
                mapping[username] = category
    return mapping


def preprocess_for_train(image, target_size):
    """Applies training augmentations: random crop, random flip, resize, and normalization.

    Args:
      image: a tf.Tensor of shape [height, width, 3] with pixel values in [0, 255].
      target_size: tuple (height, width) for the output image.

    Returns:
      A tf.Tensor with shape target_size and pixel values normalized to [-1, 1].
    """
    # conver to tensor, random crop it, resize, to 224x224, random flipping, normalization to [-1,1]

    image = tf.convert_to_tensor(image, dtype=tf.float32)

    # image shape
    shape = tf.shape(image)
    # random crop
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(0.05, 1.0),
        min_object_covered=0,
        use_image_if_no_bounding_boxes=True,
    )
    cropped = tf.slice(image, begin, size)

    # resize the cropped image to the target size using bilinear interpolation.
    resized = tf.image.resize(cropped, target_size)

    # Random horizontal flip.
    flipped = tf.image.random_flip_left_right(resized)

    # Normalize image to [-1, 1]
    normalized = (flipped - 128.0) / 128.0

    # shape is target_size normalized
    return normalized


def process_and_save_image(image_path, username, category):
    """Process an image using training augmentation and save it as a compressed numpy file."""
    try:
        # load image using PIL and convert to RGB
        img = Image.open(image_path).convert("RGB")
        # resize image (256x256 original images become 224x224 after processing)
        img = img.resize((256, 256))
        img_array = np.array(img).astype(np.float32)

        # apply training preprocessing
        preprocessed_tensor = preprocess_for_train(img_array, target_size)
        # convert back to numpy array for saving.
        preprocessed = preprocessed_tensor.numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return

    # create output directory for the category if it doesn't exist
    category_dir = os.path.join(output_base, category)
    os.makedirs(category_dir, exist_ok=True)

    # save the compressed image
    filename = os.path.basename(image_path)
    output_path = os.path.join(category_dir, os.path.splitext(filename)[0] + ".npz")
    try:
        np.savez_compressed(output_path, preprocessed)
        print(f"Saved compressed preprocessed image to: {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")


def main():
    # return dictionary to map username to category
    mapping = load_mapping(mapping_file)

    # process each image in the image directory
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
