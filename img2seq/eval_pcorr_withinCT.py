import warnings

# Suppress all warnings (optional, for cleaner logs)
warnings.filterwarnings("ignore")

import argparse

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
                    preds[m, idx],
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
            gts[m],
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
                suffixes=("", ""),
            )

    # Average Pearson scores across batches
    pearson_cols = [c for c in df_final.columns if c.startswith("Pearson")]
    df_final["avg_Pearson"] = df_final[pearson_cols].mean(axis=1)

    return df_final


if __name__ == "__main__":
    ### 0. CLI + Initialization
    parser = argparse.ArgumentParser(
        description=(
            "Per-cell-type, per-batch gene-wise correlation evaluation for "
            "permuted-within-cell-type baseline."
        )
    )
    parser.add_argument(
        "--species",
        type=str,
        choices=["mouse", "human"],
        default="mouse",
        help="Species to evaluate (mouse or human).",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        default=None,
        help=(
            "Path to .pt file with baseline predictions. "
            "If not provided and species=='mouse', uses the original mouse path. "
            "For human, this argument is strongly recommended."
        ),
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default=None,
        help=(
            "Prefix for output CSV files. "
            "If not set, defaults to '<species>_withinCT'. "
            "Files are written as '<prefix>_<celltype>.csv'."
        ),
    )

    args = parser.parse_args()
    species = args.species

    # Device (not heavily used here, but kept for consistency)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Running within-cell-type correlation evaluation for species = {species} on device = {device}")

    base_dir = "/mlbio_scratch/wen2/cross-model-gen"

    ### 2. Load and preprocess data
    ### 2.1. Load AnnData
    adata_path = f"{base_dir}/img_dec/imagen/{species}_300725.h5ad"
    print(f"Loading AnnData from: {adata_path}")
    adata_seq = sc.read(adata_path)

    # Library-size normalize counts per cell
    sc.pp.normalize_total(adata_seq)

    # Log-transform normalized counts
    sc.pp.log1p(adata_seq)

    # Subset to every second cell (matching how predictions were produced)
    adata = adata_seq[0::2, :]

    # Cell-type labels (may differ between species)
    if species == "human":
        # For human, you mentioned using 'cell_type_subset' before
        # (change back to 'cell_type' if needed)
        label_col = "cell_type_subset"
        if label_col not in adata.obs.columns:
            # Fallback if subset label is not available
            label_col = "cell_type"
    else:
        label_col = "cell_type"

    labels = adata.obs[label_col]
    unique_labels = labels.unique()
    print(f"Using cell-type column: {label_col}")
    print("Unique cell types:", list(unique_labels))

    # Batch labels (exp_id) for per-batch averaging
    labels_exp = adata.obs["exp_id"]
    unique_labels_exp = labels_exp.unique()
    print("Unique exp_id:", list(unique_labels_exp))

    ### 2.2. Load predictions and ground truth
    if args.pred_path is not None:
        pred_path = args.pred_path
    else:
        # Original hard-coded mouse path; only safe default for mouse.
        if species == "mouse":
            pred_path = (
                f"{base_dir}/seq_dec/genes/"
                "mouse_permute_within_celltype_0.5_withinbatch/"
                "ep3_0.16849383380092917_0.16552310871581236.pt"
            )
        else:
            raise ValueError(
                "No --pred_path provided and default path is mouse-specific. "
                "Please specify --pred_path for human."
            )

    print(f"Loading predictions from: {pred_path}")
    output_all = torch.load(pred_path, map_location="cpu")[0::2, :]
    target_tensor_all = adata.X

    print("Pred shape:", output_all.shape)
    print("GT   shape:", target_tensor_all.shape)

    ### 3. Per-cell-type evaluation
    out_prefix = args.out_prefix or f"{species}_withinCT"

    for cell_type in unique_labels:
        print(f"\n=== Cell type: {cell_type} ===")

        # Mask cells belonging to the current cell type
        mask = [label == cell_type for label in labels]
        print(cell_type, "n =", sum(mask))

        output_tmp = output_all[mask]
        target_tensor_tmp = target_tensor_all[mask]
        labels_exp_tmp = labels_exp[mask]

        # Compute per-batch correlations and average across batches
        results = calculate_results_per_batch(
            adata[mask],
            output_tmp.detach().cpu(),
            target_tensor_tmp,
        )

        # Rank genes by average Pearson correlation
        results_order = results.sort_values(
            by=["avg_Pearson"],
            ascending=False,
        )

        # Save one CSV per cell type
        safe_ct = str(cell_type).replace(" ", "_").replace("/", "_")
        out_csv = f"{out_prefix}_{safe_ct}.csv"
        results_order.to_csv(out_csv, index=True)
        print(f"Saved results for {cell_type} to: {out_csv}")
