# Generative modeling reveals the connection between cellular morphology and gene expression

This repository contains the official code for **COSMIC**, a bidirectional generative model that links single-cell transcriptomics and nuclear morphology images. It includes data preprocessing pipelines, training scripts for both directions (seq2img and img2seq), and evaluation utilities.

Data is available [here](https://drive.google.com/drive/folders/13mFoxXPIhlVMvGOJ0VN06TR3jAsoDs_S?usp=drive_link).

This is currently only a template and will be finished soon.

---

## Overview

COSMIC is designed to:

- Learn a shared representation between **gene expression** and **nuclear morphology**.
- Generate realistic nuclear images from single-cell gene expression profiles (seq2img).
- Infer gene expression from nuclear images (img2seq).

This repository provides:

- Data loading and preprocessing utilities for single-cell RNA-seq and image data.
- Model definitions for encoders, decoders, and conditional diffusion components.
- Training scripts for COSMIC in both directions.
- Evaluation scripts and example notebooks for reproducing key results.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create and activate a Python environment** (conda, mamba, or venv)
   ```bash
   conda create -n cosmic python=3.10
   conda activate cosmic
   ```

3. **Install dependencies**

  ```bash
  pip install -r requirements.txt
  ```

---

## Quick Start

Below is a high-level workflow. You can adapt paths, configs, and script names to your local setup.

### 1. Data Preparation

- Download the datasets from the Google Drive link:
  - Processed single-cell RNA-seq (`mouse_IRIS.h5ad` and `human_IRIS.h5ad`).
  - Single-cell nuclear images (`mouse_IRIS_images.zip`, `human_IRIS_images.zip`).
- Place them under a common root:
  ```text
  ./data/
  ```
- Unzip the zip file of the nuclear images. You will get images named with `cell_id.png`.

### 2. Training

You can train COSMIC in one or both directions:

- **seq2img**: generate nuclear images from gene expression.
- **img2seq**: predict gene expression from nuclear images.

1. **Train seq2img**

   First, we need to get the features from gene expression using scVI:

   ```bash
   python ./seq2img/train_mouse.py
   ```

   ```bash
   python ./seq2img/train_mouse.py
   ```

3. **Train img2seq**

   ```bash
   python ./img2seq/mouse.py
   ```

### 3. Evaluation

Once models are trained, you can evaluate fidelity, diversity, and cross-modal consistency.

1. **Generate samples**

   - For seq2img: generate nuclear images from held-out gene expression profiles.
   - For img2seq: predict gene expression for held-out images.

   ```bash
   python inference_human.py
   ```

2. **Compute quantitative metrics**

   - **Per-gene Pearson correlation** between predicted and ground-truth expression.
   - **Sliced Wasserstein Distance (SWD)** between real and generated embeddings.
   - **k-NNA** to assess overlap between real and generated distributions.

   ```bash
   python eval_swd.py
   ```
