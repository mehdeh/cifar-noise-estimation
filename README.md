# CIFAR-10 Noise Level Estimation

A deep learning project for estimating noise levels in CIFAR-10 images using ResNet architecture. The model learns to predict the standard deviation (σ) of Gaussian noise added to clean images.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [New Unified CLI Interface](#new-unified-cli-interface)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Configuration](#configuration)
- [Results](#results)
- [Methodology](#methodology)
- [Example Workflow](#example-workflow)
- [Experimental Scenarios](#experimental-scenarios)
- [Monitoring Training](#monitoring-training)
- [Tips and Best Practices](#tips-and-best-practices)
- [Troubleshooting](#troubleshooting)
- [Useful Commands](#useful-commands)
- [References](#references)

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
python main.py train --exp-name low_noise --sigma-min 0.0 --sigma-max 0.5 --epochs 100

# Train on high noise
python main.py train --exp-name high_noise --sigma-min 0.5 --sigma-max 2.0 --epochs 100

# Train on full range
python main.py train --exp-name full_range --sigma-min 0.0 --sigma-max 2.0 --epochs 150

# Evaluate all three models
python main.py test --checkpoint runs/low_noise_*/checkpoints/best_model.pth --exp-name eval_low
python main.py test --checkpoint runs/high_noise_*/checkpoints/best_model.pth --exp-name eval_high
python main.py test --checkpoint runs/full_range_*/checkpoints/best_model.pth --exp-name eval_full
```

## 🧪 Experimental Scenarios

### Experiment 1: Hyperparameter Tuning - Learning Rate

```bash
# Test different learning rates
python main.py train --exp-name lr_0001 --lr 0.001 --epochs 50
python main.py train --exp-name lr_0005 --lr 0.005 --epochs 50
python main.py train --exp-name lr_00001 --lr 0.0001 --epochs 50
```

### Experiment 2: Batch Size Comparison

```bash
# Test different batch sizes
python main.py train --exp-name bs_64 --batch-size 64 --epochs 50
python main.py train --exp-name bs_128 --batch-size 128 --epochs 50
python main.py train --exp-name bs_256 --batch-size 256 --epochs 50
```

### Experiment 3: Noise Range Comparison

```bash
# Small range
python main.py train --exp-name sigma_0_05 --sigma-min 0.0 --sigma-max 0.5 --epochs 100

# Medium range
python main.py train --exp-name sigma_0_10 --sigma-min 0.0 --sigma-max 1.0 --epochs 100

# Large range
python main.py train --exp-name sigma_0_20 --sigma-min 0.0 --sigma-max 2.0 --epochs 100

# Evaluate all three models with detailed analysis
python main.py test --checkpoint runs/sigma_0_05_*/checkpoints/best_model.pth --exp-name eval_0_05 --evaluate-by-noise-level
python main.py test --checkpoint runs/sigma_0_10_*/checkpoints/best_model.pth --exp-name eval_0_10 --evaluate-by-noise-level
python main.py test --checkpoint runs/sigma_0_20_*/checkpoints/best_model.pth --exp-name eval_0_20 --evaluate-by-noise-level
```

## 📈 Monitoring Training

### Using tmux for Background Training

For long training sessions, use tmux to keep training running in the background:

```bash
# Start a new tmux session
tmux new -s training

# Run your training
python main.py train --epochs 200 --exp-name long_training

# Detach from session: Press Ctrl+B, then D

# Reattach to session later
tmux attach -t training

# List all sessions
tmux ls

# Kill a session
tmux kill-session -t training
```

### Save Training Output to File

```bash
# Save both stdout and stderr to a log file
python main.py train --epochs 100 2>&1 | tee training_output.log

# This allows you to:
# - See output in real-time
# - Have a complete log file saved
```

### Monitor Training Progress

```bash
# Watch training log in real-time
tail -f runs/exp_*/logs/train.log

# View latest metrics
cat runs/exp_*/logs/metrics.json

# Check GPU usage
watch -n 1 nvidia-smi
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

### Common Issues and Solutions

#### 1. CUDA out of memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Option 1: Reduce batch size
python main.py train --batch-size 64  # or even smaller: 32

# Option 2: Use CPU instead
python main.py train --device cpu --batch-size 32

# Option 3: Reduce test batch size as well
python main.py train --batch-size 64 --batch-size-test 50
```

#### 2. ModuleNotFoundError

**Error**: `ModuleNotFoundError: No module named 'torch'` (or other dependencies)

**Solution**:
```bash
# Reinstall all dependencies
pip install -r requirements.txt

# Or install specific missing package
pip install torch torchvision
```

#### 3. Dataset download fails

**Error**: Dataset download timeout or connection issues

**Solutions**:
```bash
# Check internet connection first
ping www.cs.toronto.edu

# Manually download CIFAR-10
# Download from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# Extract to: ./datasets/cifar-10-batches-py/

# Or retry with better connection
python main.py train
```

#### 4. Import errors / Module not found

**Error**: `ImportError: attempted relative import with no known parent package`

**Solution**:
```bash
# Make sure you're in the project root directory
cd /home/ubuntu/cifar-noise-estimation

# Verify structure
ls -la  # Should see main.py, src/, config/, etc.

# Run from project root
python main.py train
```

#### 5. Permission denied errors

**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**:
```bash
# Check and fix directory permissions
chmod -R u+w runs/
chmod -R u+w datasets/

# Or run with appropriate permissions
sudo python main.py train  # Not recommended, fix permissions instead
```

#### 6. Checkpoint not found

**Error**: `FileNotFoundError: checkpoint not found`

**Solution**:
```bash
# List available checkpoints
find runs/ -name "*.pth"

# Use the full path to existing checkpoint
python main.py test --checkpoint runs/exp_20241026_143022/checkpoints/best_model.pth

# Check if checkpoint file exists
ls -lh runs/*/checkpoints/
```

## 🔍 Useful Commands

### Get Help

```bash
# Show main help
python main.py --help

# Show training options
python main.py train --help

# Show testing options
python main.py test --help
```

### Project Navigation

```bash
# Find all Python files (excluding cache)
find . -name "*.py" | grep -v __pycache__

# Show project structure
tree -I '__pycache__|*.pyc|*.pth'

# Count lines of code
find . -name "*.py" -not -path "*/__pycache__/*" | xargs wc -l
```

### Results Management

```bash
# List all experiments
ls -lh runs/

# Find the latest experiment
ls -lt runs/ | head -n 5

# Check experiment size
du -sh runs/*

# View specific experiment results
cat runs/exp_*/logs/metrics.json | jq '.'

# Find best models
find runs/ -name "best_model.pth"
```

### Cleanup

```bash
# Remove all experiment results (be careful!)
rm -rf runs/*

# Remove downloaded datasets
rm -rf datasets/

# Remove specific experiment
rm -rf runs/exp_20241026_143022/

# Remove all checkpoints except best models
find runs/ -name "checkpoint_epoch_*.pth" -delete
find runs/ -name "latest.pth" -delete

# Clean Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### System Information

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check PyTorch version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check GPU information
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Check disk space
df -h

# Check memory usage
free -h
```

## 📚 References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Technical Report.

## 📄 License

This project is free and open source, available for anyone to use, modify, and distribute. It is provided as-is for research, educational, and commercial purposes without any restrictions.

**MIT License** - Feel free to use this code in your own projects!

## 👤 Author

**Mehdi Dehghani**
- GitHub: [@mehdeh](https://github.com/mehdeh)
- Project Link: [cifar-noise-estimation](https://github.com/mehdeh/cifar-noise-estimation)

This project was developed as part of research in noise estimation for images, with the assistance of AI-powered development tools (Cursor AI).

## 🙏 Acknowledgments

- CIFAR-10 dataset by Alex Krizhevsky
- ResNet architecture by Kaiming He et al.
- PyTorch team for the deep learning framework
- Cursor AI for development assistance and code optimization
- The open-source community for inspiration and best practices

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/mehdeh/cifar-noise-estimation/issues) or submit a pull request.

## ⭐ Support

If you find this project helpful for your research or work, please consider giving it a star on GitHub!

---

**Note**: This is a research-oriented project developed for educational and research purposes. For production use, additional optimization and validation may be required.

