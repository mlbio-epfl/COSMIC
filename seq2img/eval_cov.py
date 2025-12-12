import warnings
warnings.filterwarnings("ignore")

import argparse

import torch


@torch.no_grad()
def compute_cov(real_feats: torch.Tensor,
                gen_feats: torch.Tensor) -> float:
    """
    Coverage (COV) of real distribution by generated samples.

    Args:
        real_feats: (N_real, D) tensor of real features.
        gen_feats:  (N_gen,  D) tensor of generated features.

    Definition used here:
      1. For each generated sample, find its nearest real sample.
      2. Count how many *distinct* real samples are selected.
      3. COV = (# distinct real samples that are nearest to some gen) / N_real.
    """
    # Ensure 2D
    assert real_feats.dim() == 2 and gen_feats.dim() == 2
    assert real_feats.size(1) == gen_feats.size(1)

    # Pairwise distances: (N_gen, N_real)
    dist = torch.cdist(gen_feats, real_feats, p=2)

    # For each generated sample, nearest real index
    nn_real_idx = dist.argmin(dim=1)          # (N_gen,)

    # Distinct real samples that are "covered" by at least one generated sample
    unique_real = torch.unique(nn_real_idx)   # (N_cov,)
    cov = unique_real.numel() / real_feats.size(0)

    return float(cov)


# ==========================
# CLI entry point
# ==========================
if __name__ == "__main__":
    ### 0. Parse arguments
    parser = argparse.ArgumentParser(
        description=(
            "Compute coverage (COV) between real and generated feature sets "
            "for a given species."
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
        default=100,
        help="Subsample step for features (default: 100, i.e. [::100]).",
    )

    args = parser.parse_args()
    species = args.species

    # Default file names follow the pattern used across the repo
    real_path = args.real_path or f"./seq2img/feature/feature_{species}_gt_morphFM.pt"
    gen_path = args.gen_path or f"./seq2img/feature/feature_{species}_gen_morphFM.pt"

    print(f"Loading real features from: {real_path}")
    print(f"Loading generated features from: {gen_path}")

    real = torch.load(real_path).cpu()[:: args.subsample]
    gen = torch.load(gen_path).cpu()[:: args.subsample]

    print("Real feats shape:", real.shape)
    print("Gen  feats shape:", gen.shape)

    cov = compute_cov(real, gen)

    print(f"[{species}] COV:", cov)
