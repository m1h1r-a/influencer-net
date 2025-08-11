# Influencer-Net: Social Media Content Classification

A deep learning project that classifies social media influencer images into 9 categories using EfficientNetV2 architecture and attention mechanisms.

## Classification Categories
- Beauty
- Family  
- Fashion
- Fitness
- Food
- Interior
- Other
- Pet
- Travel

## Project Structure

```
influencer-net/
├── dataset/                          # Raw data storage
│   ├── influencers.txt              # Influencer metadata (username, category, stats)
│   ├── JSON-Image_files_mapping.txt # Image-to-username mappings
│   └── image/                       # Raw Instagram images
├── data_sampling/                    # Dataset filtering
│   └── minimize_data.py             # Reduce dataset size (50 influencers/category, 300 posts/influencer)
├── preprocess_img/                   # Image preprocessing
│   ├── imgPreprocess.py            # Basic preprocessing to .npy
│   └── compressPreprocess.py       # Compressed preprocessing to .npz
├── intermediate_results/             # Debug and testing
│   └── intermediateTensor.py       # Debug preprocessing with step visualization
├── classify_images/                  # Main models
│   ├── classificationV3.py        # EfficientNetV2 training (3-phase)
│   └── extract_image_features.py  # Feature extraction
├── final_models/                     # Production models
│   ├── final_classification.py     # Attention-based classifier
│   └── run_classification.py       # Inference script
└── classification_output/            # Training results
    ├── models/                      # Trained model files
    ├── confusion_matrix.png
    ├── overall_training_history.png
    └── prediction_samples.png
```

## High-Level Workflow

```
Raw Images → Data Sampling → Preprocessing → EfficientNetV2 Training → Feature Extraction → Attention Classification → Predictions
```

## Usage

### 1. Data Preparation (Optional)
Reduce dataset size for development:
```bash
cd data_sampling/
python minimize_data.py
```

### 2. Image Preprocessing
Choose preprocessing method:
```bash
# Basic preprocessing
cd preprocess_img/
python imgPreprocess.py

# Compressed preprocessing (recommended)
python compressPreprocess.py
```

### 3. Model Training
Train the main classification model:
```bash
cd classify_images/
python classificationV3.py
```

### 4. Feature Extraction
Extract features for attention model:
```bash
python extract_image_features.py
```

### 5. Final Model Training
Train attention-based classifier:
```bash
cd final_models/
python final_classification.py
```

### 6. Inference
Classify new images:
```bash
python run_classification.py
```

## Model Architecture

### EfficientNetV2 Training (`classificationV3.py`)
- **Base**: EfficientNetV2-S pre-trained on ImageNet
- **Input**: 224×224×3 images
- **Multi-phase training**:
  - Phase 1: Frozen base, train head (10 epochs)
  - Phase 2: Fine-tune top 50 layers (30 epochs)  
  - Phase 3: Fine-tune top 100 layers (10 epochs)
- **Features**: Mixed precision, data augmentation, cosine decay LR

### Attention Model (`final_classification.py`)
- **Input**: 2048-dimensional feature vectors
- **Architecture**: Feature embedding → PostAttentionLayer → Classification
- **Custom PostAttentionLayer**: Context-aware feature weighting
- **Output**: 9-class predictions with attention weights

## Configuration

Key parameters in the scripts:
```python
# Training
BATCH_SIZE = 32
NUM_CLASSES = 9
LEARNING_RATE = 1e-4
IMAGE_SIZE = (224, 224)

# Data sampling
INFLUENCERS_PER_CATEGORY = 50
POSTS_PER_INFLUENCER = 300

# Model architecture  
FEATURE_DIM = 2048
HIDDEN_DIM = 512
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- PIL, NumPy, Matplotlib, Seaborn, scikit-learn
- GPU with 8GB+ VRAM (recommended)

## Debugging

Use `intermediate_results/intermediateTensor.py` for:
- Step-by-step preprocessing visualization
- Understanding data augmentation effects
- Validating the preprocessing pipeline

## Output Files

Training generates:
- Model weights for each phase
- `confusion_matrix.png`: Performance visualization
- `overall_training_history.png`: Training progress
- `prediction_samples.png`: Sample predictions
- `classification-v3.log`: Training logs

## Inference Format

```python
result = predict_influencer_category(feature_vector, model, class_names)
# Returns:
{
    "predicted_category": "fashion",
    "confidence": 0.87,
    "top_predictions": [("fashion", 0.87), ("beauty", 0.08), ("other", 0.03)]
}
```

This project implements a complete pipeline from raw social media images to production-ready classification models using modern deep learning techniques.