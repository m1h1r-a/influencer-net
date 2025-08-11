# %%
# Import required libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime

# Enable mixed precision for memory efficiency
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')  # Use mixed precision

# Configuration
SEED = 42
BATCH_SIZE = 32
NUM_CLASSES = 9
LEARNING_RATE = 1e-4
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

# Set random seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Class names (matching your folder structure)
class_names = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'other', 'pet', 'travel']

# Create output directory for saved models
os.makedirs('models', exist_ok=True)

# Path to your preprocessed data
data_dir = '/kaggle/input/image-classification/tfCompressedPreprocessedImages'

print("Libraries and configuration loaded!")

# %%
# Enhanced data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
])

# Custom learning rate scheduler with warmup
class WarmupCosineDecayScheduler(tf.keras.callbacks.Callback):
    def __init__(self, learning_rate_base, total_steps, warmup_steps, hold_base_rate_steps=0):
        super(WarmupCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.learning_rates = []

    def on_batch_begin(self, batch, logs=None):
        lr = self.get_lr(batch)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.learning_rates.append(lr)

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.learning_rate_base * (step / self.warmup_steps)
        
        if step < self.warmup_steps + self.hold_base_rate_steps:
            return self.learning_rate_base
        
        step = step - self.warmup_steps - self.hold_base_rate_steps
        total_steps = self.total_steps - self.warmup_steps - self.hold_base_rate_steps
        
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / total_steps))
        return self.learning_rate_base * cosine_decay

print("Augmentation and callbacks defined!")

# %%
# Function to get file count by class for display
def get_file_counts(data_dir):
    counts = {}
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            counts[class_name] = len([f for f in os.listdir(class_dir) if f.endswith('.npz')])
        else:
            counts[class_name] = 0
    return counts

# Create efficient tf.data pipeline from npz files
def create_dataset_from_npz_files(data_dir, class_names, batch_size=32, is_training=False):
    """Create a tf.data.Dataset from npz files on disk"""
    # Create lists of files and labels
    file_paths = []
    labels = []
    
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist")
            continue
            
        files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.npz')]
        print(f"Found {len(files)} files in {class_name}")
        file_paths.extend(files)
        labels.extend([idx] * len(files))
    
    # Function to load an image from an npz file
    def load_npz_file(file_path, label):
        try:
            data = np.load(file_path.numpy().decode())
            image = data[data.files[0]]
            # Ensure proper data type and range
            image = tf.cast(image, tf.float32)
            return image, tf.one_hot(label, depth=NUM_CLASSES)
        except Exception as e:
            print(f"Error loading file: {e}")
            # Return a placeholder image in case of error
            return tf.zeros([224, 224, 3], dtype=tf.float32), tf.one_hot(label, depth=NUM_CLASSES)
    
    # Create dataset from file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    # Shuffle dataset if training
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), 10000))
    
    # Map the loading function
    dataset = dataset.map(
        lambda file, label: tf.py_function(
            load_npz_file, [file, label], [tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Set shapes explicitly after loading
    dataset = dataset.map(
        lambda x, y: (tf.ensure_shape(x, [224, 224, 3]), tf.ensure_shape(y, [NUM_CLASSES])),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply data augmentation if training - now with known shapes
    if is_training:
        # Apply individual augmentations
        def apply_augmentation(image, label):
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            # Random brightness
            image = tf.image.random_brightness(image, 0.1)
            # Random contrast
            image = tf.image.random_contrast(image, 0.9, 1.1)
            # Random saturation
            image = tf.image.random_saturation(image, 0.9, 1.1)
            return image, label
        
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset, len(file_paths)

# %%
# Display dataset information
file_counts = get_file_counts(data_dir)
print("Dataset file counts by class:")
for class_name, count in file_counts.items():
    print(f"  {class_name}: {count} files")

total_files = sum(file_counts.values())
print(f"\nTotal files: {total_files}")

# Calculate approximate sizes for train/val/test split (80/10/10)
train_size = int(0.8 * total_files)
val_size = int(0.1 * total_files)
test_size = total_files - train_size - val_size

# Estimate steps per epoch
steps_per_epoch = train_size // BATCH_SIZE
validation_steps = val_size // BATCH_SIZE

print(f"\nEstimated steps per epoch: {steps_per_epoch}")
print(f"Estimated validation steps: {validation_steps}")

# %%
# Create datasets using efficient loading
print("\nCreating datasets...")

train_dataset, train_count = create_dataset_from_npz_files(
    data_dir, class_names, batch_size=BATCH_SIZE, is_training=True
)

val_dataset, val_count = create_dataset_from_npz_files(
    data_dir, class_names, batch_size=BATCH_SIZE, is_training=False
)

test_dataset, test_count = create_dataset_from_npz_files(
    data_dir, class_names, batch_size=BATCH_SIZE, is_training=False
)

# Update steps based on actual counts
steps_per_epoch = train_count // BATCH_SIZE
validation_steps = val_count // BATCH_SIZE

print(f"\nActual dataset sizes:")
print(f"  Training: {train_count} images")
print(f"  Validation: {val_count} images")
print(f"  Test: {test_count} images")
print(f"  Total: {train_count + val_count + test_count} images")

# %%
# Function to build enhanced model
def build_advanced_model():
    """Build an enhanced EfficientNetV2 model with better architecture"""
    # Create base model with pretrained weights
    base_model = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_preprocessing=False
    )
    
    # Freeze the base model initially
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create model with enhanced classification head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    
    # Pass through base model
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    
    # Add more capacity with deeper classification head
    x = Dense(512, activation=None)(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, activation=None)(x)
    x = BatchNormalization()(x)
    x = tf.keras.activations.relu(x)
    x = Dropout(0.3)(x)
    
    # Final classification layer with float32 for stability
    outputs = Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
    
    # Assemble the model
    model = Model(inputs, outputs)
    
    # Compile with appropriate optimizer and loss
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

print("Model building function defined!")

# %%
# Phase 1: Train only top layers
print("Phase 1: Training only top layers...")
model, base_model = build_advanced_model()
model.summary()

# Setup callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'models/efficient_net_phase1.weights.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# Log directory for TensorBoard
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_phase1"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train with frozen base model
history_phase1 = model.fit(
    train_dataset,
    epochs=10,  # Adjust as needed
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr, checkpoint, tensorboard_callback]
)

# Plot phase 1 results
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history_phase1.history['accuracy'])
plt.plot(history_phase1.history['val_accuracy'])
plt.title('Phase 1 - Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history_phase1.history['loss'])
plt.plot(history_phase1.history['val_loss'])
plt.title('Phase 1 - Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()

# %%
# Phase 2: Unfreeze and fine-tune the top 50 layers
print("Phase 2: Fine-tuning top 50 layers...")

# Unfreeze top 50 layers
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Update callbacks for phase 2
checkpoint.filepath = 'models/efficient_net_phase2.weights.h5'
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_phase2"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train with partially unfrozen model
history_phase2 = model.fit(
    train_dataset,
    epochs=30,  # Adjust as needed
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr, checkpoint, tensorboard_callback]
)

# Plot phase 2 results
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history_phase2.history['accuracy'])
plt.plot(history_phase2.history['val_accuracy'])
plt.title('Phase 2 - Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history_phase2.history['loss'])
plt.plot(history_phase2.history['val_loss'])
plt.title('Phase 2 - Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()

# %%
# Phase 3: Unfreeze and fine-tune more layers
print("Phase 3: Fine-tuning more layers...")

# Unfreeze more layers
for layer in base_model.layers[-100:]:
    layer.trainable = True

# Recompile with even lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Update callbacks for phase 3
checkpoint.filepath = 'models/efficient_net_phase3.weights.h5'
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "_phase3"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Instead of using the WarmupCosineDecayScheduler, use a simpler approach
# Create a learning rate schedule function
def cosine_decay_with_warmup(epoch, lr):
    # Total epochs and warmup epochs
    total_epochs = epochs_phase3
    warmup_epochs = int(0.1 * total_epochs)  # 10% of epochs for warmup
    
    # Warmup phase
    if epoch < warmup_epochs:
        return 1e-5 * ((epoch + 1) / warmup_epochs)
    
    # Cosine decay phase
    decay_epochs = total_epochs - warmup_epochs
    epoch_in_decay = epoch - warmup_epochs
    cosine = 0.5 * (1 + np.cos(np.pi * epoch_in_decay / decay_epochs))
    return 1e-5 * cosine

# Create a learning rate scheduler callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup, verbose=1)

# Train with more unfrozen layers - removed the warmup_lr callback
epochs_phase3 = 10  # Adjust as needed
history_phase3 = model.fit(
    train_dataset,
    epochs=epochs_phase3,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr, checkpoint, tensorboard_callback, lr_scheduler]
)

# Plot phase 3 results
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history_phase3.history['accuracy'])
plt.plot(history_phase3.history['val_accuracy'])
plt.title('Phase 3 - Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history_phase3.history['loss'])
plt.plot(history_phase3.history['val_loss'])
plt.title('Phase 3 - Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()

# Save final model
model.save('models/efficient_net_final.keras')

# %%
# Combine histories
combined_history = {
    'accuracy': (
        history_phase1.history['accuracy'] + 
        history_phase2.history['accuracy'] + 
        history_phase3.history['accuracy']
    ),
    'val_accuracy': (
        history_phase1.history['val_accuracy'] + 
        history_phase2.history['val_accuracy'] + 
        history_phase3.history['val_accuracy']
    ),
    'loss': (
        history_phase1.history['loss'] + 
        history_phase2.history['loss'] + 
        history_phase3.history['loss']
    ),
    'val_loss': (
        history_phase1.history['val_loss'] + 
        history_phase2.history['val_loss'] + 
        history_phase3.history['val_loss']
    )
}

# Plot overall training history
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(combined_history['accuracy'])
plt.plot(combined_history['val_accuracy'])
plt.title('Overall Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.axvline(x=len(history_phase1.history['accuracy']), color='r', linestyle='--')
plt.axvline(x=len(history_phase1.history['accuracy']) + len(history_phase2.history['accuracy']), 
            color='r', linestyle='--')
plt.legend(['Train', 'Validation', 'Phase Change'])

plt.subplot(1, 2, 2)
plt.plot(combined_history['loss'])
plt.plot(combined_history['val_loss'])
plt.title('Overall Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.axvline(x=len(history_phase1.history['loss']), color='r', linestyle='--')
plt.axvline(x=len(history_phase1.history['loss']) + len(history_phase2.history['loss']), 
            color='r', linestyle='--')
plt.legend(['Train', 'Validation', 'Phase Change'])
plt.tight_layout()
plt.savefig('overall_training_history.png')
plt.show()

# %%
# Function to evaluate model on test set
def evaluate_model(model, test_dataset):
    """Evaluate model on test set and visualize results"""
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred = []
    y_true = []
    
    for images, labels in test_dataset:
        batch_predictions = model.predict(images)
        y_pred.extend(np.argmax(batch_predictions, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    # Convert to arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return test_accuracy, y_true, y_pred

# Evaluate the trained model
test_accuracy, y_true, y_pred = evaluate_model(model, test_dataset)
print(f"Final test accuracy: {test_accuracy:.4f}")

# %%
# Create a feature extractor for multimodal model
def create_feature_extractor(model):
    """Create and save a feature extractor for multimodal integration"""
    # Create a new model that outputs features
    feature_extractor = tf.keras.models.Model(
        inputs=model.input,
        outputs=model.get_layer('global_average_pooling2d').output
    )
    
    feature_extractor.save('models/image_feature_extractor.keras')
    print("Feature extractor saved for multimodal model integration!")
    
    return feature_extractor

# Create and save the feature extractor
feature_extractor = create_feature_extractor(model)

# %%
# Visualize some predictions
def visualize_predictions(model, test_dataset, class_names, num_samples=10):
    """Visualize model predictions on test images"""
    # Get a batch of test images
    for images, labels in test_dataset:
        # Only take the first num_samples
        if images.shape[0] >= num_samples:
            batch_images = images[:num_samples]
            batch_labels = labels[:num_samples]
            break
    
    # Make predictions
    predictions = model.predict(batch_images)
    
    # Plot the results
    plt.figure(figsize=(20, 10))
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)
        
        # Convert from [-1,1] to [0,1] range for display
        img = (batch_images[i] + 1) / 2.0
        plt.imshow(img)
        
        true_label = np.argmax(batch_labels[i])
        pred_label = np.argmax(predictions[i])
        
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", 
                  color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.show()

# Visualize some test predictions
visualize_predictions(model, test_dataset, class_names)


