import warnings

# Suppress all warnings (optional, for cleaner logs)
warnings.filterwarnings("ignore")

import scanpy as sc
from scipy.stats import pearsonr
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
device = "cuda:1" if torch.cuda.is_available() else "cpu"


### 1. Utilities
### 1.1. Gene-wise correlation evaluation (single dataset)
def calculate_results(adata_subset, preds, gts):
    """
    Compute per-gene Pearson correlation between predicted and ground-truth expression.

    For each gene:
      - Compute Pearson correlation across all cells.
      - Compute Pearson correlation restricted to cells with non-zero ground-truth expression.
      - If a gene has no non-zero counts, or fewer than 2 samples, correlation is set to 0.

    Args:
        adata_subset: AnnData object with gene metadata (used for gene names).
        preds: array-like of shape [n_cells, n_genes], predicted expression.
        gts: array-like of shape [n_cells, n_genes], ground-truth expression.

    Returns:
        results_df: pandas DataFrame with columns:
            - 'genes'
            - 'Pearson'
            - 'pvals'
            - 'nonzero_Pearson'
            - 'nonzero_pvals'
    """
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

        # Need at least 2 samples to compute Pearson correlation
        if len(gts[:, idx]) < 2:
            continue

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
            else:
                # Not enough non-zero cells for a reliable correlation
                gene_names.append(adata_subset.var.index[idx])
                corrs.append(0)
                pvals.append(0)
                corrs_nonzero.append(0)
                pvals_nonzero.append(0)
        else:
            # Gene never expressed in ground truth
            gene_names.append(adata_subset.var.index[idx])
            corrs.append(0)
            pvals.append(0)
            corrs_nonzero.append(0)
            pvals_nonzero.append(0)

    results_df = pd.DataFrame(
        {
            "genes": gene_names,
            "Pearson": corrs,
            "pvals": pvals,
            "nonzero_Pearson": corrs_nonzero,
            "nonzero_pvals": pvals_nonzero,
        }
    )

    # Drop any rows with NaNs
    results_df = results_df.dropna()

    return results_df


### 1.2. Gene-wise correlation evaluation per batch (exp_id)
def calculate_results_per_batch(adata_subset, preds, gts):
    """
    Compute per-gene correlation statistics separately for each batch (exp_id),
    then merge and average the per-batch Pearson scores.

    Args:
        adata_subset: AnnData object with obs['exp_id'] and gene metadata.
        preds: array-like [n_cells, n_genes], predicted expression.
        gts: array-like [n_cells, n_genes], ground-truth expression.

    Returns:
        df_final: pandas DataFrame with:
            - per-batch correlation columns (Pearson_expX, ...)
            - 'avg_Pearson' as the mean Pearson across batches
    """
    labels = adata_subset.obs["exp_id"]
    unique_labels = labels.unique()
    df_final = None

    # Compute correlations per batch
    for label in unique_labels:
        print(label)
        m = labels == label

        results_batch_df = calculate_results(
            adata_subset[m],
            preds[m],
            gts[m]
        )

        # Rename columns to encode batch in column name, except 'genes'
        columns = {
            k: f"{k}_{label}"
            for k in results_batch_df.columns
            if k != "genes"
        }
        results_batch_df = results_batch_df.rename(columns=columns)

        # Merge per-batch results on gene name
        if df_final is None:
            df_final = results_batch_df
        else:
            df_final = pd.merge(
                df_final,
                results_batch_df,
                on="genes",
                suffixes=("", "")
            )

    # Average Pearson scores across batches
    pearson_cols = [c for c in df_final.columns if c.startswith("Pearson")]
    df_final["avg_Pearson"] = df_final[pearson_cols].mean(axis=1)

    return df_final


### 2. Load and preprocess data
### 2.1. Load AnnData
adata_seq = sc.read(
    "/mlbio_scratch/wen2/cross-model-gen/img_dec/imagen/mouse_300725.h5ad"
)

# Library-size normalize counts per cell
sc.pp.normalize_total(adata_seq)

# Log-transform normalized counts
sc.pp.log1p(adata_seq)

# Subset to every second cell (matching how predictions were produced)
adata = adata_seq[0::2, :]

# Cell-type labels (used for per-cell-type analysis)
# labels = adata.obs['cell_type_subset']  # for human
labels = adata.obs["cell_type"]
unique_labels = labels.unique()

# Batch labels (exp_id) for per-batch averaging
labels_exp = adata.obs["exp_id"]
unique_labels_exp = labels_exp.unique()

### 2.2. Load predictions and ground truth
output_all = torch.load(
    "/mlbio_scratch/wen2/cross-model-gen/seq_dec/genes/"
    "mouse_permute_within_celltype_0.5_withinbatch/"
    "ep3_0.16849383380092917_0.16552310871581236.pt"
)[0::2, :]

target_tensor_all = adata.X


### 3. Per-cell-type evaluation
for cell_type in unique_labels:
    print(cell_type)

    # Mask cells belonging to the current cell type
    mask = [label == cell_type for label in labels]
    print(cell_type, sum(mask))

    output_tmp = output_all[mask]
    target_tensor_tmp = target_tensor_all[mask]
    labels_exp_tmp = labels_exp[mask]

    # Compute per-batch correlations and average across batches
    results = calculate_results_per_batch(
        adata[mask],
        output_tmp.detach().cpu(),
        target_tensor_tmp
    )

    # Rank genes by average Pearson correlation
    results_order = results.sort_values(
        by=["avg_Pearson"],
        ascending=False
    )

    # Note: filename is 'human_withinCT.csv' even though this is mouse data,
    # kept for consistency with the original script.
    results_order.to_csv("human_withinCT.csv", index=True)
