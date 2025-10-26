# CIFAR-10 Noise Level Estimation

A deep learning project for estimating noise levels in CIFAR-10 images using ResNet architecture. The model learns to predict the standard deviation (σ) of Gaussian noise added to clean images.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Configuration](#configuration)
- [Results](#results)
- [Methodology](#methodology)
- [Citation](#citation)

## 🎯 Overview

This project implements a noise estimation framework where:
- Clean images from CIFAR-10 are corrupted with Gaussian noise: `x_noise = x + eps * σ`
- `eps ~ N(0, I)` is standard Gaussian noise
- `σ` is uniformly sampled from `[σ_min, σ_max]`
- A ResNet-18 model is trained to predict the noise level `σ` from noisy images

This is a fundamental task in image denoising and is related to diffusion models and score-based generative models.

## ✨ Features

- **Modular Architecture**: Clean, research-oriented codebase with clear separation of concerns
- **Flexible Configuration**: YAML-based configuration with CLI override support
- **Experiment Tracking**: Automatic experiment directory creation with timestamped runs
- **Comprehensive Logging**: Detailed logging of training progress and metrics
- **Rich Visualizations**: Loss curves, prediction scatter plots, noise distributions
- **Checkpoint Management**: Automatic saving of best models and resumable training
- **Unified CLI Interface**: Single main script with train/test subcommands using Click
- **Reproducibility**: Seed control for reproducible experiments
- **Persian/Farsi Support**: Documentation and CLI help in Persian language

## 📁 Project Structure

```
cifar-noise-estimation/
├── config/
│   └── default.yaml          # Default configuration file
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py        # Dataset loading and noise addition
│   ├── models/
│   │   ├── __init__.py
│   │   └── resnet.py         # ResNet architecture
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training loop
│   │   └── evaluator.py      # Evaluation logic
│   └── utils/
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       ├── logger.py         # Logging utilities
│       └── visualization.py  # Plotting functions
├── runs/                     # Experiment outputs (auto-generated)
│   └── exp_YYYYMMDD_HHMMSS/
│       ├── config.yaml       # Experiment configuration
│       ├── checkpoints/      # Model checkpoints
│       ├── logs/             # Training logs
│       └── plots/            # Visualization plots
├── main.py                   # Main CLI script (train & test)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd cifar-noise-estimation
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

The CIFAR-10 dataset will be automatically downloaded on first run.

## 🔧 New Unified CLI Interface

Starting from this version, the project uses a single `main.py` script with subcommands instead of separate `train.py` and `test.py` files. This provides:

- **Cleaner interface**: Single entry point for all operations
- **Shared functionality**: Common code is reused between train and test
- **Better organization**: Related commands grouped together
- **Multilingual support**: Help text available in Persian/Farsi

### CLI Structure

```
main.py
├── train    # Training subcommand
└── test     # Testing subcommand
```

### Migration from Previous Version

If you were using the old interface:

```bash
# Old way
python train.py --epochs 100
python test.py --checkpoint model.pth

# New way
python main.py train --epochs 100
python main.py test --checkpoint model.pth
```

## 🎬 Quick Start

### Show Available Commands

```bash
# Show main help
python main.py --help

# Show training options
python main.py train --help

# Show testing options  
python main.py test --help
```

### Train a Model

```bash
# Train with default configuration
python main.py train

# Train with custom parameters
python main.py train --epochs 200 --lr 0.001 --batch-size 128
```

### Evaluate a Model

```bash
# Evaluate the best model from training
python main.py test --checkpoint runs/exp_YYYYMMDD_HHMMSS/checkpoints/best_model.pth
```

## 📖 Usage

### Training

#### Basic Training

```bash
python main.py train --config config/default.yaml
```

#### Training with Custom Parameters

```bash
python main.py train \
    --config config/default.yaml \
    --exp-name my_experiment \
    --epochs 200 \
    --batch-size 128 \
    --lr 0.001 \
    --sigma-min 0.0 \
    --sigma-max 1.0 \
    --device cuda
```

#### Resume Training

```bash
python main.py train \
    --resume \
    --resume-path runs/exp_YYYYMMDD_HHMMSS/checkpoints/latest.pth
```

#### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to config file | `config/default.yaml` |
| `--exp-name` | Experiment name prefix | `None` |
| `--epochs` | Number of training epochs | `100` |
| `--batch-size` | Training batch size | `128` |
| `--batch-size-test` | Test batch size | `100` |
| `--lr` | Learning rate | `0.001` |
| `--device` | Device (cuda/cpu) | `cuda` |
| `--seed` | Random seed | `42` |
| `--sigma-min` | Min noise std | `0.0` |
| `--sigma-max` | Max noise std | `1.0` |
| `--resume` | Resume from checkpoint | `False` |
| `--resume-path` | Path to checkpoint | `None` |

### Evaluation

#### Basic Evaluation

```bash
python main.py test --checkpoint path/to/checkpoint.pth
```

#### Evaluation with Options

```bash
python main.py test \
    --checkpoint runs/exp_YYYYMMDD_HHMMSS/checkpoints/best_model.pth \
    --config config/default.yaml \
    --exp-name test_experiment \
    --batch-size 100 \
    --evaluate-by-noise-level
```

#### Evaluation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--checkpoint` | Path to model checkpoint | **Required** |
| `--config` | Path to config file | `config/default.yaml` |
| `--exp-name` | Experiment name for results | `test` |
| `--batch-size` | Test batch size | `100` |
| `--device` | Device (cuda/cpu) | `cuda` |
| `--sigma-min` | Min noise std for testing | `0.0` |
| `--sigma-max` | Max noise std for testing | `1.0` |
| `--save-visualizations` | Save plots | `True` |
| `--evaluate-by-noise-level` | Per-level metrics | `False` |

### Configuration

#### YAML Configuration File

Create custom configuration files in the `config/` directory:

```yaml
# config/custom.yaml
model:
  name: "ResNet18"
  num_classes: 1

data:
  batch_size_train: 256
  batch_size_test: 100
  num_workers: 4

noise:
  sigma_min: 0.0
  sigma_max: 2.0

training:
  epochs: 200
  learning_rate: 0.0005
  optimizer: "SGD"
  scheduler: "CosineAnnealingLR"
```

#### CLI Override

CLI arguments take precedence over config file values:

```bash
# Config file has lr=0.001, but CLI overrides it to 0.0005
python main.py train --config config/default.yaml --lr 0.0005
```

The final configuration (after CLI override) is saved in the experiment directory.

## 📊 Results

### Training Output

Each training run creates an experiment directory:

```
runs/exp_20241026_143022/
├── config.yaml              # Final configuration used
├── checkpoints/
│   ├── best_model.pth       # Best model (lowest val loss)
│   ├── latest.pth           # Latest checkpoint (for resuming)
│   └── checkpoint_epoch_*.pth  # Periodic checkpoints
├── logs/
│   ├── train.log            # Detailed training logs
│   └── metrics.json         # Training metrics (JSON)
└── plots/
    ├── loss_curve.png       # Training/validation loss
    └── samples_epoch_*.png  # Sample images at different epochs
```

### Evaluation Output

```
runs/test_20241026_150000/
├── config.yaml              # Test configuration
├── logs/
│   ├── test.log             # Evaluation logs
│   └── test_metrics.json    # Detailed metrics
└── plots/
    ├── test_scatter_predictions.png    # True vs Predicted σ
    ├── test_noise_distribution.png     # Noise distributions
    └── test_sample_images.png          # Sample clean/noisy images
```

### Metrics

The evaluation provides comprehensive metrics:

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination
- **Pearson Correlation**: Linear correlation coefficient
- **Error Statistics**: Mean, median, std, max absolute error

## 🔬 Methodology

### Problem Formulation

Given a clean image `x`, we add Gaussian noise:

```
x_noisy = x + ε * σ
```

where:
- `ε ~ N(0, I)`: Standard Gaussian noise
- `σ ~ Uniform(σ_min, σ_max)`: Noise standard deviation

The model `f_θ` learns to predict `σ`:

```
σ_pred = f_θ(x_noisy)
```

### Training Process

1. **Data Loading**: CIFAR-10 images are loaded and normalized
2. **Noise Addition**: For each batch, random noise levels are sampled and noise is added
3. **Forward Pass**: Noisy images pass through ResNet-18
4. **Loss Computation**: MSE loss between predicted and true σ
5. **Backpropagation**: Gradient descent to update model weights

**Key Design Choice**: Noise is added **dynamically during training**, not pre-computed. This ensures:
- Different noise patterns in each epoch
- Better generalization
- More robust noise estimation

### Model Architecture

- **Base Model**: ResNet-18
- **Input**: Noisy RGB images (3×32×32)
- **Output**: Single scalar value (predicted σ)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: SGD with momentum
- **Scheduler**: Cosine Annealing LR

## 📝 Example Workflow

### Complete Training and Evaluation

```bash
# 1. Train a model
python main.py train \
    --exp-name noise_estimation_v1 \
    --epochs 100 \
    --lr 0.001 \
    --sigma-min 0.0 \
    --sigma-max 1.0

# 2. Evaluate the trained model
python main.py test \
    --checkpoint runs/noise_estimation_v1_20241026_143022/checkpoints/best_model.pth \
    --exp-name evaluation_v1 \
    --evaluate-by-noise-level

# 3. Check results
ls runs/evaluation_v1_*/plots/
```

### Experiment with Different Noise Ranges

```bash
# Train on low noise
python main.py train --exp-name low_noise --sigma-min 0.0 --sigma-max 0.5

# Train on high noise
python main.py train --exp-name high_noise --sigma-min 0.5 --sigma-max 2.0

# Train on full range
python main.py train --exp-name full_range --sigma-min 0.0 --sigma-max 2.0
```

## 🛠️ Tips and Best Practices

1. **GPU Memory**: If you encounter OOM errors, reduce batch size:
   ```bash
   python main.py train --batch-size 64
   ```

2. **Reproducibility**: Always set the same seed for reproducible results:
   ```bash
   python main.py train --seed 42
   ```

3. **Hyperparameter Tuning**: Modify config file for extensive experiments:
   ```yaml
   training:
     learning_rate: 0.0005
     scheduler: "CosineAnnealingLR"
     scheduler_params:
       T_max: 200
   ```

4. **Monitoring**: Check training progress in real-time:
   ```bash
   tail -f runs/exp_*/logs/train.log
   ```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size: `--batch-size 64`
   - Use CPU: `--device cpu`

2. **Dataset download fails**
   - Check internet connection
   - Manually download CIFAR-10 and place in `./data/`

3. **Import errors**
   - Ensure you're in the project root directory
   - Reinstall dependencies: `pip install -r requirements.txt`

## 📚 References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Technical Report.

## 📄 License

This project is provided as-is for research and educational purposes.

## 👤 Author

Developed as a research project for noise estimation in images.

## 🙏 Acknowledgments

- CIFAR-10 dataset by Alex Krizhevsky
- ResNet architecture by Kaiming He et al.
- PyTorch team for the deep learning framework

---

**Note**: This is a research-oriented project. For production use, additional optimization and validation may be required.

