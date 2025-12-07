import torch

@torch.no_grad()
def compute_cov(real_feats: torch.Tensor,
                gen_feats: torch.Tensor) -> float:
    """
    Coverage (COV) of real distribution by generated samples.

    real_feats: (N_real, D) tensor
    gen_feats:  (N_gen,  D) tensor

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
# Example usage
# ==========================
if __name__ == "__main__":
    # Fake example: 100 real, 100 generated, 128-dim features
    real = torch.load('feature_gt.pt').cpu()[::100]
    gen  = torch.load('feature_gen.pt').cpu()[::100]

    cov = compute_cov(real, gen)

    print("COV:", cov)
