# Generative modeling reveals the connection between cellular morphology and gene expression

This repository contains the official code for **COSMIC**, a bidirectional generative model that links single-cell transcriptomics and nuclear morphology images. It includes data preprocessing pipelines, training scripts for both directions (seq2img and img2seq), and evaluation utilities.

Data is available [here](https://drive.google.com/drive/folders/13mFoxXPIhlVMvGOJ0VN06TR3jAsoDs_S?usp=drive_link).

This is currently only a template and will be finished soon.

---

## Overview
COSMIC is a bidirectional generative framework that links single-cell morphology with gene expression. Built on a foundation model trained on over 21 million segmented nuclei and coupled to transcriptomic embeddings, COSMIC quantitatively decomposes transcriptional variance reflected in morphology and morphological variance explained by gene expression. Using a new IRIS-based multimodal dataset that captures high-resolution images and transcriptomes from the same cells, COSMIC accurately models cell type identity, continuous dynamics such as cell cycle progression, and treatment response in prostate cancer. This framework establishes a quantitative bridge between cellular form and gene expression, enabling mechanistic discovery and predictive modeling in basic and translational cell biology.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mlbio-epfl/COSMIC.git
   cd COSMIC
   ```

2. **Create and activate a Python environment** (conda, mamba, or venv)
   ```bash
   conda create -n cosmic python=3.9
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
- Unzip the zip file of the nuclear images. You will get images named with `[cell_id].jpg` in the directory `./data/images_mouse` and `./data/images_human`.

### 2. Training + Inference

You can train COSMIC and run inference in one or both directions:

- **seq2img**: generate nuclear images from gene expression.
- **img2seq**: predict gene expression from nuclear images.

**2.1. Seq2img**

   2.1.1. Here, we use the mouse data as an example. First, you need to get the features from gene expression using scVI:

   ```bash
   python ./seq2img/scVI_mouse.py
   ```

   After running this, you will get `feature_mouse_scvi.pt` in `./seq2img/feature`. If you want to skip this step, you can download the features directly from [here](https://drive.google.com/drive/folders/13mFoxXPIhlVMvGOJ0VN06TR3jAsoDs_S?usp=drive_link).
   
   2.1.2. Then, you can train the diffusion model conditioned on the gene expression features:
   
   ```bash
   python ./seq2img/train_mouse.py
   ```

   During training, you can keep checking the intermediate results in `./seq2img/result/mouse`. After training, you will have a checkpoint `seq2img_mouse.pt` in `./seq2img/ckpt`. If you want to skip this step, you can download the checkpoints directly from [here](https://drive.google.com/drive/folders/13mFoxXPIhlVMvGOJ0VN06TR3jAsoDs_S?usp=drive_link).

   2.1.3. Finally, you can run the inference to generate nuclear images:
   ```bash
   python ./seq2img/inference_mouse.py
   ```
   The generated images will be saved in the folder `./seq2img/inference/mouse`.

**2.2. Img2seq**

   First, download the checkpoint of morphology FM `ckpt_morphFM.pt` [here](https://drive.google.com/drive/folders/13mFoxXPIhlVMvGOJ0VN06TR3jAsoDs_S?usp=drive_link) and put it into `./ckpt`. Then, run
   ```bash
   python ./img2seq/mouse.py
   ```
   After running, you will get both the model checkpoint and the predicted genes in the file `img2seq_mouse.pt` in `./img2seq/inference`. The keys to the checkpoint and the inference are "model_ckpt" and "pred_genes".

### 3. Evaluation

Once models are trained, you can evaluate fidelity, diversity, and cross-modal consistency.

1. **Evaluate generated nuclear images**
   
   First, we evaluate the generated images in our morphology FM embedding space. Thus, let's first get the embedding:
   ```bash
   python ./seq2img/get_embedding.py
   ```
   
   Then, we can start evaluating:
   - **Coverage (COV)** to measure how well the generated embeddings cover the real embeddings:
   ```bash
   python ./seq2img/eval_cov.py
   ```
   - **Sliced Wasserstein Distance (SWD)** between real and generated embeddings:
   ```bash
   python ./seq2img/eval_swd.py
   ```
   - **k-Nearest-Neighbour Accuracy (k-NNA)** to assess overlap between real and generated distributions:
   ```bash
   python ./seq2img/eval_knna.py
   ```

   Besides, we can directly evaluate the generated images in the original image space by checking whether they preserve the correct cell type:
   - **Cell type classification** on generated nuclear images:
   ```bash
   python ./seq2img/eval_ct_cls.py
   ```

3. **Evaluate generated gene expression**

   - **Per-gene Pearson correlation** between predicted and ground-truth expression across different cell types:
   ```bash
   python ./img2seq/eval_pcorr_acrossCT.py
   ```
   - **Per-gene Pearson correlation** between predicted and ground-truth expression within each cell type:
   ```bash
   python ./img2seq/eval_pcorr_withinCT.py
   ```
   - **Cell type classification** on generated gene expression:
   ```bash
   python ./img2seq/eval_ct_cls.py
   ```

   
