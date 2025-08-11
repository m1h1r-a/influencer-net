# Influencer-Net: Project Execution Flow

## 🚀 New Organized Structure Overview

The project has been reorganized into a more professional and maintainable structure:

```
influencer-net/
├── 📂 src/                          # Source code (organized by function)
│   ├── data_processing/             # Data sampling and filtering
│   ├── preprocessing/               # Image preprocessing modules  
│   ├── training/                    # Model training scripts
│   └── inference/                   # Production inference code
├── 📂 data/                         # All data files
│   ├── raw/                        # Original datasets
│   ├── processed/                  # Preprocessed data
│   ├── features/                   # Extracted features
│   └── samples/                    # Small test datasets
├── 📂 models/                      # Model storage
│   ├── checkpoints/               # Training checkpoints
│   └── final/                     # Production models
├── 📂 experiments/                 # Research and debugging
├── 📂 outputs/                     # Generated outputs
│   ├── logs/                      # Training logs
│   ├── plots/                     # Visualizations
│   └── results/                   # Final results
├── 📂 config/                      # Configuration files
├── 📂 docs/                        # Documentation
│   └── technical/                 # Technical docs
├── 📂 notebooks/                   # Jupyter notebooks
├── 📂 scripts/                     # Utility scripts
├── 📂 tests/                       # Unit tests
└── 📂 utils/                       # Helper utilities
```

## 📋 Project Execution Flow

### Phase 1: Data Preparation
```bash
# Step 1: Reduce dataset size for development (optional)
cd src/data_processing/
python minimize_data.py
# Creates: data/samples/smallInfluencers.txt, data/samples/smallMappings.txt
```

### Phase 2: Image Preprocessing  
```bash
# Step 2: Preprocess images for training
cd src/preprocessing/
python compressPreprocess.py  # Recommended for production (compressed storage)
# OR
python imgPreprocess.py       # Standard preprocessing
# Creates: Preprocessed .npz files in data/processed/
```

### Phase 3: Model Training
```bash
# Step 3: Train the main classification model
cd src/training/
python classificationV3.py
# Creates: Model checkpoints in models/checkpoints/
#         Training plots in outputs/plots/
#         Logs in outputs/logs/
```

### Phase 4: Feature Extraction
```bash
# Step 4: Extract features for attention model
cd src/training/
python extract_image_features.py
# Creates: Feature vectors in data/features/
```

### Phase 5: Final Classification with Attention
```bash
# Step 5: Train attention-based classifier
cd src/training/
python final_classification.py
# Creates: Final model in models/final/
#         Results in outputs/results/
```

### Phase 6: Inference
```bash
# Step 6: Run inference on new images
cd src/inference/
python run_classification.py
# Classifies new images using trained models
```

## 📁 File Organization by Purpose

### Core Training Files (src/training/)
- **`classificationV3.py`**: Main EfficientNet training with 3-phase approach
- **`extract_image_features.py`**: Feature extraction for attention model  
- **`final_classification.py`**: Attention-based final classifier

### Preprocessing Files (src/preprocessing/)
- **`compressPreprocess.py`**: Compressed image preprocessing (recommended)
- **`imgPreprocess.py`**: Standard image preprocessing
- **`tfPreprocess.py`**: TensorFlow-specific preprocessing

### Data Management (src/data_processing/)
- **`minimize_data.py`**: Dataset size reduction for development
- **`cp_sample_data.sh`**: Data copying utilities

### Inference (src/inference/)
- **`run_classification.py`**: Production inference script

### Experiments & Debugging (experiments/)
- **`intermediateTensor.py`**: Debug preprocessing with visualizations
- **`intermediatePreprocess.py`**: Alternative preprocessing implementation
- **`visualize.py`**: Data visualization utilities
- **`testPreprocessed/`**: Testing utilities for preprocessing

## 🗑️ Files You Can Consider Deleting

### Duplicate Files (kept for safety, can remove after verification)
- `intermediate_results/` directory (duplicated in `experiments/`)
- `data_sampling/` directory (moved to `src/data_processing/`)
- `preprocess_img/` directory (moved to `src/preprocessing/`)  
- `classify_images/` directory (moved to `src/training/`)
- `final_models/` directory (moved to `src/training/` and `src/inference/`)

### Legacy/Outdated Files
- **`experiments/testImage/`**: Single test image (use `data/samples/` instead)
- **Duplicate small dataset files**: Remove duplicates from original directories after verifying new structure works

### Development Files (Optional to keep)
- **`experiments/testPreprocessed/test*.py`**: Testing scripts (keep for debugging)
- **`experiments/notesImgPreprocessing.md`**: Development notes (moved to `docs/technical/`)

## 🔧 Configuration Updates Needed

### Update Python Import Paths
After restructuring, update imports in Python files:
```python
# Old imports
from classify_images.classificationV3 import model

# New imports  
from src.training.classificationV3 import model
```

### Update File Paths in Scripts
Update hardcoded paths in:
- `src/training/classificationV3.py`: Model save paths → `models/checkpoints/`
- `src/training/extract_image_features.py`: Input/output paths
- `src/inference/run_classification.py`: Model loading paths

### Environment Setup
```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/influencer-net/src"

# Or create setup.py for proper package installation
pip install -e .
```

## 🚀 Quick Start (New Structure)
```bash
# 1. Development with small dataset
cd src/data_processing && python minimize_data.py

# 2. Preprocess images  
cd ../preprocessing && python compressPreprocess.py

# 3. Train model
cd ../training && python classificationV3.py

# 4. Extract features and train attention model
python extract_image_features.py
python final_classification.py

# 5. Run inference
cd ../inference && python run_classification.py
```

## 📊 Expected Outputs
- **Models**: Saved in `models/checkpoints/` and `models/final/`
- **Plots**: Training history, confusion matrices in `outputs/plots/`
- **Logs**: Detailed training logs in `outputs/logs/`
- **Results**: Classification results in `outputs/results/`

---

**Note**: All original files are preserved in their original locations. You can safely delete the legacy directories after verifying the new structure works correctly with your workflows.