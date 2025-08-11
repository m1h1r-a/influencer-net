# Influencer-Net: Deep Learning Classification Project

## Project Overview

Influencer-Net is a comprehensive machine learning project designed to classify social media influencer content into 9 distinct categories using deep learning techniques. The project implements a complete pipeline from data preprocessing to model deployment, utilizing EfficientNetV2 architecture for image classification and feature extraction.

### Classification Categories
- Beauty
- Family
- Fashion
- Fitness
- Food
- Interior
- Other
- Pet
- Travel

## Project Architecture

```
influencer-net/
├── dataset/                          # Raw data storage
├── data_sampling/                    # Data sampling and filtering tools
├── preprocess_img/                   # Image preprocessing modules
├── intermediate_results/             # Testing and debugging outputs
├── classify_images/                  # Main classification models
├── final_models/                     # Production-ready models
└── classification_output/            # Training results and visualizations
```

## Core Components and Usage

### 1. Data Management (`dataset/`)

#### Key Files:
- **`influencers.txt`**: Master dataset containing influencer usernames, categories, and statistics
  - Format: `username	category	followers	following	posts`
  - Contains comprehensive influencer metadata

- **`JSON-Image_files_mapping.txt`**: Maps image files to influencer usernames
  - Large file containing filename-to-username mappings
  - Used for organizing raw image data

- **`image/`**: Directory containing raw Instagram images
  - Images named as: `username-postid.jpg`
  - Organized by influencer username prefix

#### When to Use:
- Use `influencers.txt` for understanding dataset composition
- Reference `JSON-Image_files_mapping.txt` for data organization
- Raw images serve as input for preprocessing pipeline

### 2. Data Sampling (`data_sampling/`)

#### `minimize_data.py` - Dataset Size Reduction
**Purpose**: Creates smaller, manageable datasets for development and testing

**Key Parameters**:
- `INFLUENCERS_PER_CATEGORY = 50`: Maximum influencers per category
- `POSTS_PER_INFLUENCER = 300`: Maximum posts per influencer

**Usage**:
```bash
cd data_sampling/
python minimize_data.py
```

**Input Files**:
- `influencers.txt`
- `JSON-Image_files_mapping.txt`

**Output Files**:
- `smallInfluencers.txt`: Filtered influencer list
- `smallMappings.txt`: Filtered image mappings

**When to Use**:
- Development phase with limited computational resources
- Creating test datasets
- Prototyping before full-scale training

### 3. Image Preprocessing (`preprocess_img/`)

#### `imgPreprocess.py` - Basic Preprocessing
**Purpose**: Preprocesses images for EfficientNetV2 training using standard format

**Key Features**:
- Resizes images to 224×224 pixels
- Applies EfficientNetV2's `preprocess_input()`
- Saves as `.npy` files for efficient loading

**Usage**:
```bash
cd preprocess_img/
python imgPreprocess.py
```

**Configuration**:
```python
target_size = (224, 224)
base_dir = "/path/to/project"
ssd_dir = "/path/to/storage"
```

#### `compressPreprocess.py` - Compressed Preprocessing
**Purpose**: Similar to basic preprocessing but with compressed storage

**Key Differences**:
- Uses `np.savez_compressed()` for storage efficiency
- Reduces file sizes by ~50-70%
- Maintains same image quality

**Usage**:
```bash
cd preprocess_img/
python compressPreprocess.py
```

**When to Use**:
- Limited storage space
- Faster I/O operations
- Production deployments

### 4. Debugging and Testing (`intermediate_results/`)

#### `intermediateTensor.py` - Debug Preprocessing
**Purpose**: Detailed preprocessing with step-by-step visualization

**Key Features**:
- Saves intermediate preprocessing steps
- Applies training augmentations (crop, flip, normalize)
- Debug output for each transformation step

**Augmentation Pipeline**:
1. Random crop (area_range: 0.05-1.0)
2. Resize to 224×224
3. Random horizontal flip
4. Normalize to [-1, 1] range

**Usage**:
```bash
cd intermediate_results/
python intermediateTensor.py
```

**When to Use**:
- Debugging preprocessing issues
- Understanding augmentation effects
- Validating preprocessing pipeline

#### Other Debug Tools:
- `intermediatePreprocess.py`: Alternative preprocessing implementation
- `visualize.py`: Visualization utilities
- `notesImgPreprocessing.md`: Preprocessing documentation and guidelines

### 5. Classification Models (`classify_images/`)

#### `classificationV3.py` - Advanced EfficientNet Model
**Purpose**: Comprehensive image classification with sophisticated architecture

**Key Features**:
- **Multi-phase Training**:
  - Phase 1: Frozen base model (10 epochs)
  - Phase 2: Fine-tune top 50 layers (30 epochs)  
  - Phase 3: Fine-tune top 100 layers (10 epochs)

- **Advanced Architecture**:
  - EfficientNetV2-S base model
  - Deep classification head (512 → 256 → 9 neurons)
  - Batch normalization and dropout layers

- **Training Features**:
  - Mixed precision training (`mixed_float16`)
  - Comprehensive data augmentation
  - Cosine decay learning rate scheduling
  - Early stopping and model checkpointing

**Usage**:
```bash
cd classify_images/
python classificationV3.py
```

**Configuration**:
```python
BATCH_SIZE = 32
NUM_CLASSES = 9
LEARNING_RATE = 1e-4
IMAGE_SIZE = (224, 224)
```

**Output**:
- Model weights for each phase
- Training history plots
- Confusion matrix
- Classification reports
- Feature extractor model

#### `extract_image_features.py` - Feature Extraction
**Purpose**: Extracts deep features from preprocessed images for downstream tasks

**Key Features**:
- Batch processing for efficiency
- Preserves original filenames
- Progress tracking with tqdm
- Error handling and logging

**Usage**:
```bash
cd classify_images/
python extract_image_features.py
```

**Configuration**:
```python
BATCH_SIZE = 32
MODEL_PATH = 'path/to/feature_extractor.keras'
INPUT_DIR = 'path/to/preprocessed_images'
OUTPUT_DIR = 'path/to/features'
```

**When to Use**:
- Building multimodal models
- Feature-based similarity search
- Transfer learning applications

### 6. Production Models (`final_models/`)

#### `final_classification.py` - Attention-Based Classifier
**Purpose**: Advanced classification using attention mechanisms on extracted features

**Key Architecture**:
- **PostAttentionLayer**: Custom attention mechanism
- **Feature Embedding**: 2048 → 512 dimensions
- **Attention Mechanism**: Context-aware feature weighting
- **Classification Head**: 256 → 9 categories

**Usage**:
```bash
cd final_models/
python final_classification.py
```

**Key Components**:
```python
class PostAttentionLayer(tf.keras.layers.Layer):
    # Implements attention mechanism for feature importance
    # Calculates attention weights for feature vectors
    # Returns weighted features and attention scores
```

#### `run_classification.py` - Model Inference
**Purpose**: Production inference script for classifying new images

**Usage**:
```python
from run_classification import predict_influencer_category

result = predict_influencer_category(feature_vector, model, class_names)
print(f"Predicted: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

**Output Format**:
```json
{
    "predicted_category": "fashion",
    "confidence": 0.87,
    "top_predictions": [
        ("fashion", 0.87),
        ("beauty", 0.08),
        ("lifestyle", 0.03)
    ]
}
```

## Complete Workflow

### Training Pipeline

1. **Data Preparation**:
   ```bash
   # Reduce dataset size (optional)
   cd data_sampling/
   python minimize_data.py
   ```

2. **Image Preprocessing**:
   ```bash
   # Choose preprocessing method
   cd preprocess_img/
   python compressPreprocess.py  # Recommended for production
   ```

3. **Model Training**:
   ```bash
   # Train classification model
   cd classify_images/
   python classificationV3.py
   ```

4. **Feature Extraction**:
   ```bash
   # Extract features for attention model
   python extract_image_features.py
   ```

5. **Attention Model Training**:
   ```bash
   # Train final attention-based classifier
   cd ../final_models/
   python final_classification.py
   ```

### Inference Pipeline

1. **Preprocess New Image**:
   - Resize to 224×224
   - Apply EfficientNetV2 preprocessing
   - Save as compressed numpy array

2. **Extract Features**:
   - Load image feature extractor
   - Generate 2048-dimensional feature vector

3. **Classify**:
   ```python
   cd final_models/
   python run_classification.py
   ```

## Model Performance Metrics

### Expected Outputs:
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Test Accuracy**: 75-85%

### Generated Artifacts:
- `confusion_matrix.png`: Visual performance analysis
- `overall_training_history.png`: Training progress plots
- `prediction_samples.png`: Sample predictions visualization
- `classification-v3.log`: Detailed training logs

## Hardware Requirements

### Minimum:
- GPU: 8GB VRAM
- RAM: 16GB
- Storage: 100GB

### Recommended:
- GPU: RTX 3080/4080 or better
- RAM: 32GB
- Storage: 500GB SSD

## Configuration and Customization

### Key Parameters to Adjust:

1. **Dataset Size**:
   - `INFLUENCERS_PER_CATEGORY` in `minimize_data.py`
   - `POSTS_PER_INFLUENCER` in `minimize_data.py`

2. **Training Parameters**:
   - `BATCH_SIZE`: Adjust based on GPU memory
   - `LEARNING_RATE`: Fine-tune for convergence
   - `NUM_EPOCHS`: Extend for better performance

3. **Model Architecture**:
   - `HIDDEN_DIM`: Attention layer complexity
   - `FEATURE_DIM`: Feature vector dimensions
   - Dropout rates: Regularization strength

### Adding New Categories:

1. Update `class_names` in all relevant files
2. Modify `NUM_CLASSES` parameter
3. Ensure training data includes new categories
4. Retrain models with updated configuration

## Best Practices

### Development:
- Start with `minimize_data.py` for smaller datasets
- Use `intermediateTensor.py` for debugging
- Monitor training with TensorBoard logs

### Production:
- Use `compressPreprocess.py` for efficiency
- Implement proper error handling
- Set up model versioning
- Monitor inference performance

### Performance Optimization:
- Use mixed precision training
- Implement efficient data pipelines
- Cache preprocessed data
- Use appropriate batch sizes

This comprehensive project provides a complete solution for influencer content classification, from raw data processing to production deployment. Each component is designed to be modular and configurable for different use cases and requirements.