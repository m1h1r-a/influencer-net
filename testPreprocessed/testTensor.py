import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# Enable GPU memory growth (if GPUs are available)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define the base directory where preprocessed npz images are stored.
output_base = "/run/media/m1h1r/04E884E1E884D1FA/compressedPreprocessedImages"

# Get list of category folders.
categories = sorted(
    [
        cat
        for cat in os.listdir(output_base)
        if os.path.isdir(os.path.join(output_base, cat))
    ]
)
print("Categories found:", categories)

# Create a mapping from category name to numerical label.
label_map = {cat: i for i, cat in enumerate(categories)}

# Gather all npz file paths and corresponding labels.
file_paths = []
labels = []
for cat in categories:
    cat_dir = os.path.join(output_base, cat)
    for fname in os.listdir(cat_dir):
        if fname.endswith(".npz"):
            file_paths.append(os.path.join(cat_dir, fname))
            labels.append(label_map[cat])

file_paths = np.array(file_paths)
labels = np.array(labels)

print(f"Found {len(file_paths)} images.")


# Define a function to load a single npz file.
def load_npz(npz_path, label):
    def _load(path):
        # Convert the tensor to a Python string.
        path_str = path.numpy().decode("utf-8")
        npz_data = np.load(path_str)
        img = npz_data["arr_0"]
        return img.astype(np.float32)

    image = tf.py_function(func=_load, inp=[npz_path], Tout=tf.float32)
    # Set the shape explicitly for downstream usage.
    image.set_shape([224, 224, 3])
    return image, label


# Create a tf.data.Dataset from the file paths and labels.
dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
# Shuffle, map the loader, batch, and prefetch.
batch_size = 32  # Adjust batch size if necessary
dataset = dataset.shuffle(buffer_size=len(file_paths))
dataset = dataset.map(load_npz, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Build the classification model using EfficientNetV2S.
input_tensor = Input(shape=(224, 224, 3))
base_model = EfficientNetV2S(
    include_top=False, weights="imagenet", input_tensor=input_tensor, pooling="avg"
)
output_tensor = Dense(len(categories), activation="softmax")(base_model.output)
model = Model(inputs=input_tensor, outputs=output_tensor)

# Compile the model.
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Evaluate the model using the tf.data pipeline.
loss, accuracy = model.evaluate(dataset, verbose=2)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Optionally, test on a single sample.
for sample_batch, sample_labels in dataset.take(1):
    predictions = model.predict(sample_batch)
    predicted_label = tf.argmax(predictions, axis=1).numpy()[0]
    print(f"Predicted category for a sample: {categories[predicted_label]}")
    break
