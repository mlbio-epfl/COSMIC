import warnings
warnings.filterwarnings("ignore")

import argparse

import torch


@torch.no_grad()
def compute_neighbor_purity(
    real_feats: torch.Tensor,
    gen_feats: torch.Tensor,
    k_max: int = 100,
):
    """
    Compute neighborhood purity / mixing between real and generated sets.

    For each point:
      - Find top-k nearest neighbors (excluding itself), for all k<=k_max.
      - Compute fraction of neighbors that come from the *same* distribution.

    Args:
        real_feats: (N_real, D) tensor of real features.
        gen_feats:  (N_gen,  D) tensor of generated features.
        k_max:      maximum neighborhood size to consider.

    Returns:
        purity_per_k: (k_max,) tensor where purity_per_k[k-1] =
                      mean over points of (# same-label neighbors in top-k) / k
        purity_per_point_at_kmax: (N_total,) tensor of per-point fractions
                                  at k = k_max.
    """
    assert real_feats.dim() == 2 and gen_feats.dim() == 2
    assert real_feats.size(1) == gen_feats.size(1)

    # Stack all features
    X = torch.cat([real_feats, gen_feats], dim=0)  # (N_total, D)
    N_real = real_feats.size(0)
    N_gen = gen_feats.size(0)
    N_total = N_real + N_gen

    # Labels: 0 = real, 1 = generated
    labels = torch.cat([
        torch.zeros(N_real, dtype=torch.long, device=X.device),
        torch.ones(N_gen, dtype=torch.long, device=X.device),
    ])  # (N_total,)

    # Pairwise distances
    dist = torch.cdist(X, X, p=2)  # (N_total, N_total)

    # Exclude self from neighbors
    inf = torch.tensor(float("inf"), device=X.device)
    dist.fill_diagonal_(inf)

    # Get indices of k_max nearest neighbors for each point
    k_max = min(k_max, N_total - 1)  # can't have more neighbors than N-1
    knn_idx = dist.topk(k=k_max, largest=False).indices  # (N_total, k_max)

    # Labels of neighbors
    knn_labels = labels[knn_idx]  # (N_total, k_max)

    # Is neighbor label same as query label?
    same = (knn_labels == labels.unsqueeze(1)).float()  # (N_total, k_max)

    # Cumulative counts: for each k, sum same-label counts in top-k
    cum_same = same.cumsum(dim=1)  # (N_total, k_max)

    # For each k, fraction of same-label neighbors in top-k per point
    ks = torch.arange(1, k_max + 1, device=X.device).view(1, -1)  # (1, k_max)
    frac_same = cum_same / ks  # (N_total, k_max)

    # Mean purity across points for each k
    purity_per_k = frac_same.mean(dim=0)  # (k_max,)

    # Per-point purity at k_max (you can look at the distribution of this)
    purity_per_point_at_kmax = frac_same[:, k_max - 1]  # (N_total,)

    return purity_per_k, purity_per_point_at_kmax


# ==========================
# CLI entry point
# ==========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Compute k-NNA / neighborhood purity between real and generated "
            "feature sets for a given species."
        )
    )
    parser.add_argument(
        "--species",
        type=str,
        choices=["mouse", "human"],
        default="mouse",
        help="Species to process (mouse or human).",
    )
    parser.add_argument(
        "--real_path",
        type=str,
        default=None,
        help=(
            "Optional path to real feature .pt file. "
            "If not provided, defaults to 'feature_<species>_gt_morphFM.pt'."
        ),
    )
    parser.add_argument(
        "--gen_path",
        type=str,
        default=None,
        help=(
            "Optional path to generated feature .pt file. "
            "If not provided, defaults to 'feature_<species>_gen_morphFM.pt'."
        ),
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=20,
        help="Subsample step for features (default: 20, i.e. [::20]).",
    )
    parser.add_argument(
        "--k_max",
        type=int,
        default=100,
        help="Maximum neighborhood size k_max.",
    )

    args = parser.parse_args()
    species = args.species

    # Default paths follow the same pattern as the COV script
    real_path = args.real_path or f"feature_{species}_gt_morphFM.pt"
    gen_path = args.gen_path or f"feature_{species}_gen_morphFM.pt"

    print(f"[{species}] Loading real features from: {real_path}")
    print(f"[{species}] Loading generated features from: {gen_path}")

    real = torch.load(real_path).cpu()[:: args.subsample]
    gen = torch.load(gen_path).cpu()[:: args.subsample]

    print("Real feats shape:", real.shape)
    print("Gen  feats shape:", gen.shape)

    k_max = args.k_max
    purity_per_k, purity_kmax_pointwise = compute_neighbor_purity(real, gen, k_max=k_max)

    print(f"[{species}] K-NNA (k={k_max}): {purity_per_k[k_max - 1].item():.3f}")
