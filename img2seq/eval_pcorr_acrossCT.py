import warnings

# Suppress all warnings (optional, for cleaner logs)
warnings.filterwarnings("ignore")

import scanpy as sc
from scipy.stats import pearsonr, spearmanr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

### 0. Initialization
### 0.1. Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


### 1. Utilities
### 1.1. Gene-wise correlation evaluation
def calculate_results(adata_subset, preds, gts):
    """
    Compute per-gene Pearson correlation between predicted and ground-truth expression.

    For each gene:
      - Compute Pearson correlation across all cells.
      - Compute Pearson correlation restricted to cells with non-zero ground-truth expression.
      - If a gene has no non-zero counts, correlation is set to 0.

    Args:
        adata_subset: AnnData object with gene metadata (used for gene names).
        preds: array-like of shape [n_cells, n_genes], predicted expression.
        gts: array-like of shape [n_cells, n_genes], ground-truth expression.

    Returns:
        results_df: pandas DataFrame with columns:
            - 'genes'
            - 'Pearson'
            - 'pvals'
            - 'Pearson_nonzero'
            - 'pvals_nonzero'
    """
    # Ensure numpy arrays (pearsonr works with array-like, but this keeps it explicit)
    preds = np.asarray(preds)
    gts = np.asarray(gts)

    corrs = []
    pvals = []
    corrs_nonzero = []
    pvals_nonzero = []
    gene_names = []

    # Boolean mask of non-zero ground-truth entries
    masks = gts > 0
    gene_counts = masks.sum(axis=0)  # number of non-zero cells per gene

    # Iterate over genes
    for idx in range(adata_subset.shape[-1]):
        if gene_counts[idx] > 0:
            # Pearson correlation using all cells
            corr, pval = pearsonr(gts[:, idx], preds[:, idx])

            # Pearson correlation using only non-zero ground-truth cells
            m = masks[:, idx]
            if m.sum() > 1:
                corr_nonzero, pval_nonzero = pearsonr(
                    gts[m, idx],
                    preds[m, idx]
                )
                gene_names.append(adata_subset.var.index[idx])
                corrs.append(corr)
                pvals.append(pval)
                corrs_nonzero.append(corr_nonzero)
                pvals_nonzero.append(pval_nonzero)

                print(idx, corr, pval_nonzero)
            else:
                # Not enough non-zero cells for a reliable correlation
                gene_names.append(adata_subset.var.index[idx])
                corrs.append(0)
                pvals.append(0)
                corrs_nonzero.append(0)
                pvals_nonzero.append(0)

                print(idx, 0, 0)
        else:
            # Gene never expressed in ground truth
            gene_names.append(adata_subset.var.index[idx])
            corrs.append(0)
            pvals.append(0)
            corrs_nonzero.append(0)
            pvals_nonzero.append(0)

            print(idx, 0, 0)

    results_df = pd.DataFrame(
        {
            "genes": gene_names,
            "Pearson": corrs,
            "pvals": pvals,
            "Pearson_nonzero": corrs_nonzero,
            "pvals_nonzero": pvals_nonzero,
        }
    )

    return results_df


### 2. Load and preprocess data
### 2.1. Load AnnData
adata_seq = sc.read(
    "/mlbio_scratch/wen2/cross-model-gen/img_dec/imagen/human_300725.h5ad"
)

# Library-size normalize counts per cell
sc.pp.normalize_total(adata_seq)

# Log-transform normalized counts
sc.pp.log1p(adata_seq)

# Subset to every second cell (matching how predictions were produced)
adata = adata_seq[0::2, :]

# Cell-type labels (not used downstream here, but kept for reference)
labels = adata.obs["cell_type"]
unique_labels = labels.unique()

### 2.2. Load predictions and ground truth
# Predicted gene expression from hybrid diffusion model
output_all = torch.load(
    "/mlbio_scratch/wen2/cross-model-gen/seq_dec_diff_latest/genes/human_0.5/"
    "predictions/hybrid_diffusion_residual_predictions_test_human.pt"
)["pred_genes"][0::2, :]

# Ground-truth expression matrix (AnnData.X)
target_tensor_all = adata.X

### 3. Compute per-gene correlations and save results
results = calculate_results(
    adata,
    output_all.detach().cpu(),
    target_tensor_all
)

# Sort genes by Pearson correlation (descending)
results_order = results.sort_values(by=["Pearson"], ascending=False)

# Save summary to CSV
results_order.to_csv("human.csv", index=True)

print("Evaluation complete!")
