# Big Picture: HRNet for Cephalometric Landmark Detection

## Overview

This repository provides an adaptation of **High-Resolution Networks (HRNets)** for **Cephalometric Landmark Detection**, extending the original HRNet model which was designed for facial landmark detection. The core idea of HRNet is to maintain high-resolution representations throughout the network by connecting high-to-low resolution multi-scale subnetworks in parallel, leading to an exceptionally precise localization capability, which is critical in accurate cephalometric analyses.

The official implementation for general facial landmark detection has been extended in this repository to process Ceph (Cephalometric) X-Ray images, allowing models to identify specific anatomical and dental landmarks. 

## Directory Structure & Component Architecture

Below is an overview of the most crucial parts of the source code architecture:

```text
HRNet-Ceph-Landmark-Detection/
├── README.md               # Original HRNet Facial Landmark Detection README 
├── requirements.txt        # Python package dependencies (e.g., torch, torchvision, yacs)
├── inference.ipynb         # A Jupyter notebook likely for interactive model inference and visualizations
│
├── tools/                  # Top-level executable scripts
│   ├── train.py            # Main script to train the HRNet model on landmark datasets
│   └── test.py             # Main script to evaluate the trained model
│
├── experiments/            # Configuration directory
│   └── ...                 # Contains YAML files defining hyperparameters and model settings for various datasets
│
└── lib/                    # The core library of the repository, containing all source code for the model
    ├── config/             # Configuration management (built using yacs)
    │   ├── defaults.py     # Default configuration values for training, evaluation, and model hyperparams
    │   └── __init__.py     
    │
    ├── core/               # Core training and evaluation logic
    │   ├── evaluation.py   # Metrics calculations (e.g., NME - Normalized Mean Error)
    │   └── function.py     # Contains the `train` and `validate` functions (loops for epochs and batches)
    │
    ├── datasets/           # Dataset loading and preprocessing pipelines
    │   ├── ceph.py         # Custom dataset class for Cephalometric Landmark Detection
    │   ├── aflw.py         # AFLW dataset loader (legacy from original facial landmarks)
    │   ├── cofw.py         # COFW dataset loader
    │   ├── wflw.py         # WFLW dataset loader
    │   ├── face300w.py     # 300W dataset loader
    │   └── sideprofile.py  # Side profile dataset loader
    │
    ├── models/             # PyTorch model definitions
    │   └── hrnet.py        # The actual PyTorch implementation of the High-Resolution Network
    │
    └── utils/              # Helper functions
        ├── transforms.py   # Image augmentations and spatial transformations (affine, cropping, scaling)
        └── utils.py        # General utilities (logging, saving/loading models, etc.)
```

## System Workflow

1.  **Configuration (`lib/config` & `experiments/`)**:
    Everything is controlled via configurations. `lib/config/defaults.py` details the default hyperparameters. Specific dataset tweaks (e.g., for Ceph) are defined into `.yaml` files placed in the `experiments/` directory.

2.  **Data Ingestion (`lib/datasets/`)**:
    The dataset pipelines, primarily `ceph.py`, load the images and annotations. The pipeline applies necessary data augmentations using `lib/utils/transforms.py` to create robust models. 

3.  **Model Forward Pass (`lib/models/hrnet.py`)**:
    The HRNet architecture maintains parallel high-resolution to low-resolution convolutions. As images pass through the network, multi-scale features are continually fused. The output is a high-resolution heatmap for each discrete landmark.

4.  **Training & Validation Loops (`tools/train.py` -> `lib/core/function.py`)**:
    `train.py` initializes the model, dataset, optimizer, and loss function, then calls the iterative loops from `lib/core/function.py`. During training, heatmaps are compared using MSE (Mean Squared Error) to ground truth Gaussian heatmaps.
    
5.  **Evaluation (`lib/core/evaluation.py`)**:
    During validation and testing, the predictions are evaluated using standard metrics, most commonly Normalized Mean Error (NME), which calculates the distance from the predicted landmark coordinates to the ground truth coordinates.

6.  **Inference**:
    The provided `inference.ipynb` orchestrates the model loading and inference stages allowing users to input novel Cephalometric images, run the forward pass, and extract/visualize x, y landmark predictions.

## Big Picture Summary

The source code relies on the powerful, purely convolutional PyTorch `hrnet.py` model to tackle the exact positioning of key points. It cleanly separates concerns into Configuration (`lib/config`), Data processing (`lib/datasets`), Network Modeling (`lib/models`), and Training/Testing Execution (`tools/`, `lib/core`).

*   **Primary Application:** Identifying skeletal and soft-tissue landmarks on lateral cephalograms.
*   **Methodology:** Heatmap regression based on High-Resolution deep representation fusion.
*   **Key Advantage:** It does not throw away high-resolution spatial information via typical pooling-heavy bottlenecks, meaning it can achieve the sub-millimeter precision necessary for medical and orthodontic landmarking.
