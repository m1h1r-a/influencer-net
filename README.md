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

## Architecture & Flow Diagrams

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Dataset   │    │  Data Sampling  │    │  Preprocessed   │
│                 │───▶│                 │───▶│    Images       │
│ • Images        │    │ minimize_data.py│    │   (.npz files)  │
│ • Metadata      │    │ (50/category)   │    │   224x224x3     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Predictions   │    │  Feature Vector │    │ EfficientNetV2  │
│                 │◀───│                 │◀───│   Training      │
│ • Category      │    │ 2048 dimensions │    │                 │
│ • Confidence    │    │   (.npz files)  │    │ • 3-Phase       │
│ • Top-3         │    │                 │    │ • Mixed Prec.   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       │                       │
         │                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Attention Model │    │Feature Extractor│    │   Checkpoints   │
│                 │    │                 │    │                 │
│ • PostAttention │    │ extract_image_  │    │ • Phase models  │
│ • Classification│    │ features.py     │    │ • Best weights  │
│ • 9 categories  │    │                 │    │ • History plots │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow (Low-Level)
```
📁 dataset/
├── influencers.txt ────┐
├── JSON-mapping.txt ───┼─── minimize_data.py ───▶ 📁 data/samples/
└── image/*.jpg ────────┘                           ├── smallInfluencers.txt
                                                    └── smallMappings.txt
                                                             │
                                                             ▼
📁 preprocess_img/                                  📁 data/processed/
├── imgPreprocess.py ────────────────────────────▶ ├── *.npz (standard)
└── compressPreprocess.py ───────────────────────▶ └── *.npz (compressed)
                                                             │
                                                             ▼
📁 classify_images/                                 📁 models/checkpoints/
├── classificationV3.py ─────────────────────────▶ ├── phase_1_model.keras
│   ├── Phase 1: Frozen base (10 epochs)            ├── phase_2_model.keras
│   ├── Phase 2: Fine-tune 50 layers (30 epochs)   ├── phase_3_model.keras
│   └── Phase 3: Fine-tune 100 layers (10 epochs)  └── feature_extractor.keras
└── extract_image_features.py ──────────────────▶ 📁 data/features/
                                                    └── *.npz (2048-dim vectors)
                                                             │
                                                             ▼
📁 final_models/                                    📁 models/final/
├── final_classification.py ─────────────────────▶ ├── attention_model.keras
│   ├── Feature Embedding (2048→512)               └── best_attention.keras
│   ├── PostAttentionLayer                                  │
│   └── Classification Head (512→256→9)                     ▼
└── run_classification.py ───────────────────────▶ 📁 outputs/results/
                                                    ├── predictions.json
                                                    └── confidence_scores.csv
```

### Training Pipeline (Execution Order)
```
[1] Data Prep     [2] Preprocessing    [3] Base Training      [4] Feature Extract
    │                 │                   │                      │
    ▼                 ▼                   ▼                      ▼
minimize_data.py   imgPreprocess.py   classificationV3.py   extract_image_
    │                 │                   │                features.py
    ▼                 ▼                   ▼                      │
samples/*.txt ──▶ processed/*.npz ──▶ checkpoints/*.keras ──▶   ▼
                                        │                 features/*.npz
                                        ▼                      │
[6] Inference   [5] Attention Training  │                      ▼
    │               │ ◀──────────────────┘           [5] final_classification.py
    ▼               ▼                                        │
run_classification  final_classification.py                 ▼
    │               │                                  final/*.keras
    ▼               ▼
results/*.json  final_models/*.keras
```

### Model Architecture Details
```
Input Image (224x224x3)
         │
         ▼
┌─────────────────────────┐
│    EfficientNetV2-S     │ ◀── Pre-trained ImageNet
│     (Base Network)      │
└─────────────────────────┘
         │
         ▼ (2048 features)
┌─────────────────────────┐
│   Feature Extractor     │ ◀── Saved after training
│    (Frozen Inference)   │
└─────────────────────────┘
         │
         ▼ (2048-dim vector)
┌─────────────────────────┐
│  Feature Embedding      │ ◀── 2048 → 512 Dense
│    + Batch Norm         │
└─────────────────────────┘
         │
         ▼ (512-dim)
┌─────────────────────────┐
│  PostAttentionLayer     │ ◀── Custom attention mechanism
│  • Query/Key/Value      │     - Self-attention weights
│  • Attention Weights    │     - Context-aware features  
│  • Weighted Features    │
└─────────────────────────┘
         │
         ▼ (512-dim weighted)
┌─────────────────────────┐
│  Classification Head    │ ◀── 512 → 256 → 9
│  • Dense(256) + ReLU    │     - Dropout(0.5)
│  • Dense(9) + Softmax   │     - 9 category output
└─────────────────────────┘
         │
         ▼
    Predictions (9 classes)
    + Attention Visualization
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