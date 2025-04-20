# %%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# %%

SEED = 42
BATCH_SIZE = 32
NUM_CLASSES = 9  
LEARNING_RATE = 1e-4
FEATURE_DIM = 2048  
HIDDEN_DIM = 512 

np.random.seed(SEED)
tf.random.set_seed(SEED)

class_names = ['beauty', 'family', 'fashion', 'fitness', 'food', 'interior', 'other', 'pet', 'travel']

data_dir = '/kaggle/input/continuous-representation/combined_features/'

# %%
def load_feature_dataset():
    feature_files = glob(data_dir + '**/*.npz', recursive=True)
    print(f"Found {len(feature_files)} feature files")
    
    features = []
    labels = []
    post_ids = []
    
    for file_path in feature_files:
        category = file_path.split('/')[-2]
        label_idx = class_names.index(category)
        
        try:
            with np.load(file_path, allow_pickle=True) as data:
                feature_vector = data['features']
                post_id = str(data['post_id'])
                
                feature_vector = feature_vector.astype(np.float32)
                
                features.append(feature_vector)
                labels.append(label_idx)
                post_ids.append(post_id)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels, post_ids

features, labels, post_ids = load_feature_dataset()

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Class distribution:")
for i, class_name in enumerate(class_names):
    count = np.sum(labels == i)
    print(f"  {class_name}: {count} ({count/len(labels)*100:.2f}%)")

# %%
X_train_val, X_test, y_train_val, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=SEED, stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=SEED, stratify=y_train_val
)

y_train_onehot = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_val_onehot = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
y_test_onehot = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")

# %%
def create_tf_dataset(features, labels, batch_size, is_training=False):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(features))
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_tf_dataset(X_train, y_train_onehot, BATCH_SIZE, is_training=True)
val_dataset = create_tf_dataset(X_val, y_val_onehot, BATCH_SIZE)
test_dataset = create_tf_dataset(X_test, y_test_onehot, BATCH_SIZE)

# %%
class PostAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(PostAttentionLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.fc = Dense(hidden_dim)
        self.context_vector = self.add_weight(
            name="context_vector",
            shape=(hidden_dim,),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True
        )
    
    def call(self, inputs):
        # inputs shape: [batch_size, feature_dim]
        # Project the features into a hidden space
        hidden = self.fc(inputs)  # [batch_size, hidden_dim]
        hidden = tf.nn.tanh(hidden)
        
        # Calculate attention scores
        # We need to preserve the batch dimension
        attention_logits = tf.matmul(
            hidden, 
            tf.reshape(self.context_vector, [-1, 1])
        )  # [batch_size, 1]
        
        attention_weights = tf.nn.softmax(attention_logits, axis=0)
        
        # Apply attention weights to the inputs
        # Need to keep the batch dimension for the next layer
        weighted_inputs = inputs * attention_weights
        
        return weighted_inputs, attention_weights
    
    def get_config(self):
        config = super(PostAttentionLayer, self).get_config()
        config.update({
            "hidden_dim": self.hidden_dim
        })
        return config

# %%
def build_influencer_profiler_model():
    inputs = Input(shape=(FEATURE_DIM,), name="feature_input")
    
    # initial embedding
    x = Dense(HIDDEN_DIM, name="feature_embedding")(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # post attention layer to focus on relevant features
    attention_layer = PostAttentionLayer(HIDDEN_DIM, name="post_attention")
    attention_output, attention_weights = attention_layer(x)
    
    # final classification layers
    x = Dense(256, activation='relu', name="fc1")(attention_output)
    x = Dropout(0.5)(x)
    
    # INFLUENCER CLASSIFICATION
    main_output = Dense(NUM_CLASSES, activation='softmax', name="influencer_classifier")(x)
    
    # create the model
    model = Model(inputs=inputs, outputs=main_output)
    
    # compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# build the model
model = build_influencer_profiler_model()
model.summary()

# %%
# Set up callbacks for training
early_stopping = EarlyStopping(
    monitor='val_accuracy', 
    patience=10, 
    restore_best_weights=True, 
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'influencer_profiler_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# %%
# Train the model
history = model.fit(
    train_dataset,
    epochs=100, 
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.show()

# %%
# evaluate on test set
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.4f}")

# make predictions on test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test

# generate classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# generate confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
# model to visualize attention weights
attention_model = Model(inputs=model.input, outputs=model.get_layer('post_attention').output[1])

# sample of test data to visualize
sample_indices = np.random.choice(len(X_test), size=9, replace=False)
sample_features = X_test[sample_indices]
sample_labels = y_test[sample_indices]

# attention weights for the samples
attention_weights = attention_model.predict(sample_features)

# visualize attention weights
plt.figure(figsize=(15, 10))
for i in range(len(sample_indices)):
    plt.subplot(len(sample_indices), 1, i+1)
    plt.bar(range(attention_weights[i].shape[0]), attention_weights[i].flatten())
    plt.title(f"Sample {i+1}: True Class = {class_names[sample_labels[i]]}")
    plt.xlabel('Feature Index')
    plt.ylabel('Attention Weight')
plt.tight_layout()
plt.show()

# %%
# Save the model
model.save('influencer_profiler_final.keras')

# laoding model
# loaded_model = tf.keras.models.load_model('influencer_profiler_final.keras', 
#                                          custom_objects={'PostAttentionLayer': PostAttentionLayer})


