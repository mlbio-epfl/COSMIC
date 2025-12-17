import warnings
warnings.filterwarnings("ignore")

import argparse

import torch
import ot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute Sliced Wasserstein Distance (SWD) between real and "
            "generated feature sets for a given species."
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
            "If not provided, defaults to 'feature_<species>_gt.pt'."
        ),
    )
    parser.add_argument(
        "--gen_path",
        type=str,
        default=None,
        help=(
            "Optional path to generated feature .pt file. "
            "If not provided, defaults to 'feature_<species>_gen.pt'."
        ),
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=2,
        help="Subsample step for features (default: 2, i.e. [1::2]).",
    )

    args = parser.parse_args()
    species = args.species

    # Default paths follow the same convention as other metrics
    real_path = args.real_path or f"./seq2img/feature/feature_{species}_gt_morphFM.pt"
    gen_path = args.gen_path or f"./seq2img/feature/feature_{species}_gen_morphFM.pt"

    print(f"[{species}] Loading real features from: {real_path}")
    print(f"[{species}] Loading generated features from: {gen_path}")

    real = torch.load(real_path).cpu()[1:: args.subsample]
    gen = torch.load(gen_path).cpu()[1:: args.subsample]

    print("Real feats shape:", real.shape)
    print("Gen  feats shape:", gen.shape)

    normal_distribution  = torch.normal(real.mean(), real.std(unbiased=False), size=real.shape)

    swd = ot.sliced_wasserstein_distance(real, gen) / ot.sliced_wasserstein_distance(real, normal_distribution)
    print(f"[{species}] Sliced Wasserstein Distance (normalized):", swd.item())
