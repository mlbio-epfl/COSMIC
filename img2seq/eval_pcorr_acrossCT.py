import warnings

# Suppress all warnings (optional, for cleaner logs)
warnings.filterwarnings("ignore")

import argparse

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
### 0.1. Gene-wise correlation evaluation
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
    # Ensure numpy arrays
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


if __name__ == "__main__":
    ### 0.2. Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Per-gene correlation evaluation between predicted and ground-truth expression."
    )
    parser.add_argument(
        "--species",
        type=str,
        choices=["mouse", "human"],
        default="human",
        help="Species to evaluate (mouse or human).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional output CSV path. If not set, defaults to '<species>.csv'.",
    )

    args = parser.parse_args()
    species = args.species

    # 0.3. Set device (not strictly needed here, but kept for consistency)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running gene-wise correlation evaluation for species = {species} on device = {device}")

    ### 2. Load and preprocess data
    ### 2.1. Load AnnData
    adata_path = f"./data/IRIS_{species}.h5ad"
    print(f"Loading AnnData from: {adata_path}")
    adata_seq = sc.read(adata_path)

    # Subset to every second cell (matching how predictions were produced)
    adata = adata_seq[0::2, :]

    # Cell-type labels (not used downstream here, but kept for reference)
    labels = adata.obs["cell_type"]
    unique_labels = labels.unique()
    print("Unique cell types:", list(unique_labels))

    ### 2.2. Load predictions and ground truth
    # Predicted gene expression from hybrid diffusion model
    pred_path = (
        f"./img2seq/inference/img2seq_{species}.pt"
    )
    print(f"Loading predictions from: {pred_path}")
    pred_file = torch.load(pred_path, map_location="cpu")

    output_all = pred_file["pred_genes"][0::2, :]
    target_tensor_all = pred_file["gt_genes"][0::2, :]

    print("Pred shape:", output_all.shape)
    print("GT   shape:", target_tensor_all.shape)

    ### 3. Compute per-gene correlations and save results
    results = calculate_results(
        adata,
        output_all.detach().cpu(),
        target_tensor_all,
    )

    # Sort genes by Pearson correlation (descending)
    results_order = results.sort_values(by=["Pearson"], ascending=False)

    # Output CSV
    out_csv = args.out_csv or f"{species}.csv"
    results_order.to_csv(out_csv, index=True)

    print(f"Evaluation complete! Results saved to: {out_csv}")
