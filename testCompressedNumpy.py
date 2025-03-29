import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import (
    EfficientNetV2S,
    decode_predictions,
    preprocess_input,
)

sample_image_path = "/run/media/m1h1r/04E884E1E884D1FA/compressedPreprocessedImages/food/simplywhisked-1660001007436771654.npz"

# Load the compressed .npz file
with np.load(sample_image_path) as data:
    img = data["arr_0"]  # Extract the stored array

# Print out the original shape and pixel range to verify integrity
print("Original image shape:", img.shape)
print("Original pixel range: [{:.3f}, {:.3f}]".format(img.min(), img.max()))

# Resize image to the expected input size for EfficientNetV2‑S (384×384)
img_resized = tf.image.resize(img, (384, 384)).numpy()
print("Resized image shape:", img_resized.shape)

# Expand dimensions to create a batch of one
img_batch = np.expand_dims(img_resized, axis=0)

# Reapply preprocessing if necessary (this scales pixel values to [-1, 1])
img_batch = preprocess_input(img_batch)

# Load the EfficientNetV2-S model with ImageNet weights
model = EfficientNetV2S(weights="imagenet")

# Run inference
predictions = model.predict(img_batch)
decoded_preds = decode_predictions(predictions, top=3)[0]

# Print the top 3 predictions
print("Top 3 Predictions:")
for pred in decoded_preds:
    print(f"Label: {pred[1]}, Probability: {pred[2]:.4f}")
