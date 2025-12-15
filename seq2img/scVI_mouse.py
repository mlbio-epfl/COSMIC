### 0. Initialization

import warnings

# Suppress all warnings (optional, for cleaner logs)
warnings.filterwarnings("ignore")

import scanpy as sc
import numpy as np
import torch
import scvi  # requires scvi-tools installed in the environment
import pandas as pd
import os

### 0.1. Set SCVI random seed
scvi.settings.seed = 42

### 1. Load data

# Load single-cell RNA-seq data (AnnData)
adata_seq = sc.read('./data/IRIS_mouse.h5ad')

### 2. Preprocessing

# Library-size normalize counts per cell
sc.pp.normalize_total(adata_seq)

# Log-transform the normalized counts
sc.pp.log1p(adata_seq)

# Select top highly variable genes (HVGs) per batch (exp_id)
# subset=True replaces adata_seq with only these HVGs
sc.pp.highly_variable_genes(
    adata_seq,
    n_top_genes=1024,
    batch_key='exp_id',
    subset=True
)

# Invert the log1p transform so that SCVI sees (approximate) raw counts again:
# X_log = log1p(X_raw) => X_raw ≈ exp(X_log) - 1
adata_seq.X = np.exp(adata_seq.X) - 1

### 3. Set up and train SCVI model

# Register AnnData with SCVI, using exp_id as batch covariate
scvi.model.SCVI.setup_anndata(
    adata_seq,
    batch_key='exp_id'
)

# Initialize SCVI model:
# - n_hidden: size of hidden layers
# - n_latent: dimensionality of latent embedding
vae = scvi.model.SCVI(
    adata_seq,
    n_hidden=128,
    n_latent=64
)

# Train VAE on the preprocessed data
vae.train(max_epochs=50)

### 4. Extract latent representation and save

# Get latent representation for each cell (NumPy array of shape [n_cells, n_latent])
feature_seq = torch.from_numpy(vae.get_latent_representation())

# Save latent features to disk for downstream models (e.g., Imagen conditioning)
os.makedirs("./seq2img/feature", exist_ok=True)
torch.save(feature_seq, './seq2img/feature/feature_mouse_scvi.pt')
