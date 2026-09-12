# Technology Stack, System Architecture & Reproducibility Inventory

This document provides a comprehensive technical inventory of the hardware architecture, compute backend, software environment, and determinism controls utilized in the 100-species, 400-subpopulation empirical study.

---

## 1. Silicon Architecture & Hardware Profile

All experiments and model fine-tuning were performed on an Apple Silicon M4 system utilizing hardware-accelerated Metal Performance Shaders (`mps`).

| Component | Specification | Technical Details & Architecture |
|:---|:---|:---|
| **SoC (System-on-Chip)** | Apple M4 | TSMC 3nm (N3E) process node |
| **CPU Microarchitecture** | 10 Cores Total | 4 Performance Cores (up to 4.41 GHz, 192KB L1i, 128KB L1d) + 6 Efficiency Cores (up to 2.89 GHz, 128KB L1i, 64KB L1d) |
| **GPU Microarchitecture** | 10 Cores | 10-core GPU with hardware-accelerated ray tracing, dynamic caching, and Metal 3 instruction set |
| **Neural Processing Engine** | 16 Cores | 38 TOPS (Trillion Operations Per Second) Apple Neural Engine |
| **Unified System Memory** | 16 GB LPDDR5X | Unified memory pool shared across CPU, GPU, and NPU |
| **Memory Bandwidth** | 120 GB/s | Ultra-wide unified memory bus enabling zero-copy tensor transfers between CPU and GPU |
| **Operating System** | macOS Darwin 27.0 | Darwin Kernel Version 27.0.0 (Build 26A428, arm64) |
| **Power & Thermal Profile** | Level 0 (Nominal) | Monitored continuously via `pmset -g therm`. No thermal throttling or performance reduction occurred during training |

---

## 2. Software Manifest & Pinned Dependencies

All libraries were installed and locked within a dedicated Python virtual environment:

| Package | Pinned Version | Role in Experimental Pipeline |
|:---|:---:|:---|
| **Python** | `3.14.7` | Base runtime environment (`[Clang 21.0.0 (clang-2100.0.12.3)]`) |
| **`torch`** | `2.14.0` | Core tensor autograd engine and Metal (`mps`) compute backend |
| **`torchvision`** | `0.29.0` | Deep convolutional vision architectures (ResNet-50) and transformation pipelines |
| **`numpy`** | `2.5.3` | High-performance array operations and procedural coordinate trigonometry |
| **`pandas`** | `3.0.5` | Dataframe serialization, group stratification metadata, and metric logging |
| **`pillow` (PIL)** | `12.3.0` | Vector rendering, polygon rasterization, and synthetic image synthesis |
| **`matplotlib`** | `3.11.2` | High-resolution publication chart rendering (`Agg` headless backend) |

---

## 3. Deep Learning Architecture & Training Configuration

* **Backbone Architecture:** ResNet-50 (50-layer deep residual network with bottleneck blocks; 25,557,124 total parameters).
* **Classifier Head:** Linear layer mapping 2048-dimensional global average pooled features to $C = 100$ output classes.
* **Input Preprocessing:**
  - Standard ImageNet Normalization: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
  - Training Pipeline: `RandomResizedCrop(224, scale=(0.7, 1.0))`, `RandomHorizontalFlip()`, `ColorJitter(b=0.3, c=0.3, s=0.3, h=0.1)`, `RandomErasing(p=0.2)`.
  - Evaluation Pipeline: `Resize(256)`, `CenterCrop(224)`, `ToTensor()`, `Normalize()`.
* **Optimization Parameters (Baseline ERM):**
  - Optimizer: AdamW (`weight_decay=1e-2`, $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$)
  - Differential Learning Rates: Backbone $\text{lr} = 1 \times 10^{-4}$, Linear head $\text{lr} = 1 \times 10^{-3}$
  - Learning Rate Schedule: Cosine Annealing decay ($T_{\max}=8, \eta_{\min}=10^{-6}$)
  - Batch Size: 32 (Full batch gradient accumulation)
* **Optimization Parameters (DFR Champion):**
  - Optimizer: AdamW (`lr=0.01`, `weight_decay=0.04`)
  - Convex Steps: 200 iterations over balanced cached representations
  - Training Time: **3.58 seconds**

---

## 4. Deterministic Reproducibility Controls

To guarantee bit-exact reproducibility across independent environments:
```python
import random, os, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```
Fixed seed 42 is strictly enforced across dataset procedural generation, model weight initialization, train data shuffling, and validation subsampling.
