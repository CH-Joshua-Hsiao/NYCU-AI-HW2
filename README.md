# Artificial Intelligence Project 2: Self-Supervised Learning Foundation Model

**Author**: 蕭至恆 (Chih-Heng Hsiao)  
**Student ID**: 413551034  
**Course**: Artificial Intelligence (NYCU)  
**GitHub Repository**: [CH-Joshua-Hsiao/NYCU-AI-HW2](https://github.com/CH-Joshua-Hsiao/NYCU-AI-HW2)

## 📌 Project Overview
This repository contains the implementation of **SimCLR** (Simple Framework for Contrastive Learning of Visual Representations). The project explores the capacity of Self-Supervised Learning (SSL) to extract highly discriminative, general-purpose visual features from the CIFAR-10 dataset without the necessity of human-annotated labels. 

The project includes training pipelines for contrastive pre-training, supervised baseline training, downstream linear probing evaluation, and extensive ablation studies (Transfer Learning on CIFAR-100, Temperature scaling, Batch Size constraints, and Projector head variations).

## 📂 Repository Structure

### Core Implementation
* `dataset.py`: Handles CIFAR-10/CIFAR-100 dataset loading and implements the critical `SimCLRTransform` data augmentation pipeline (Random Crop, Horizontal Flip, Color Jitter, Grayscale).
* `models.py`: Defines the modified ResNet-18 backbone (adjusted for $32\times32$ CIFAR images), the non-linear MLP Projector Head, and the Linear Classifier used for downstream probing.
* `loss.py`: Implements the Normalized Temperature-scaled Cross Entropy (NT-Xent) contrastive loss function.
* `utils.py`: Contains utility functions, notably the `knn_monitor` which evaluates representation quality on-the-fly during contrastive training.

### Training & Evaluation Scripts
* `train_simclr.py`: The main script for executing SimCLR contrastive pre-training. Supports ablations like `--temperature`, `--batch_size`, and `--no_projector`.
* `train_linprobe.py`: Evaluates the quality of frozen backbones via linear probing. Supports switching between CIFAR-10 and CIFAR-100 (`--dataset`), and loading different backbones (Random, Supervised, SimCLR).

### Automation & Visualization
* `run_all.py`: An orchestration script that automatically sequentially executes the full suite of required baseline experiments and ablation studies.
* `plot_results.py`: Reads the generated `*_logs.json` metric files and generates publication-quality `matplotlib` charts for the final report.

### Report
* `report.tex`: The comprehensive academic LaTeX report detailing the methodology, mathematical formulations, and deep analysis of the experimental results.

---

## 🚀 Usage Instructions

### 1. Environment Setup
Ensure you have a standard PyTorch environment installed with `torchvision` and `matplotlib`.
```bash
pip install torch torchvision matplotlib
```

### 2. Running the Full Automated Pipeline
To reproduce the entire suite of experiments (Baselines, Supervised Learning, Ablation Studies, and Transfer Learning), simply execute:
```bash
python run_all.py
```
*Note: This process trains multiple ResNet-18 models and will take a significant amount of time depending on your GPU hardware.*

### 3. Running Individual Experiments Manually
If you wish to run specific configurations, you can use the command-line arguments provided in the training scripts.

**Train the SimCLR Baseline:**
```bash
python train_simclr.py --batch_size 256 --epochs 200 --temperature 0.5
```

**Evaluate the SimCLR Backbone via Linear Probing (CIFAR-10):**
```bash
python train_linprobe.py --pretrained_model simclr_baseline.pth --epochs 100 --dataset cifar10
```

**Evaluate Transfer Learning on CIFAR-100:**
```bash
python train_linprobe.py --pretrained_model simclr_baseline.pth --epochs 100 --dataset cifar100
```

### 4. Generating Plots
After the experiments finish and the `*_logs.json` files are populated, generate the charts for the report:
```bash
python plot_results.py
```
This will output several `.png` files (e.g., `simclr_learning_curves.png`, `transfer_learning_cifar100.png`) into the root directory.

---

## ✨ Implemented Features
- [x] **Modified ResNet-18 Backbone**: Optimized spatial resolution for small CIFAR images.
- [x] **NT-Xent Loss**: Efficient vectorized implementation of contrastive cosine similarity optimization.
- [x] **On-the-fly K-NN Monitoring**: Accurately tracks topological feature representation quality during unsupervised training.
- [x] **Supervised vs SSL Benchmarking**: Compares linear probing against upper-bound supervised baselines and lower-bound random initializations.
- [x] **Ablation Support**: Fully parameterized arguments for dynamically testing Temperature, Batch Sizes, and Projector head architectures.
- [x] **Cross-Domain Transfer Learning**: Automated pipeline for evaluating representations on the unseen CIFAR-100 dataset.
- [x] **JSON Metrics Logging**: Decouples training from visualization, ensuring experiments don't need to be re-run to adjust plot aesthetics.
