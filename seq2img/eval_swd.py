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
        default=100,
        help="Subsample step for features (default: 100, i.e. [::100]).",
    )

    args = parser.parse_args()
    species = args.species

    # Default paths follow the same convention as other metrics
    real_path = args.real_path or f"./seq2img/feature/feature_{species}_gt.pt"
    gen_path = args.gen_path or f"./seq2img/feature/feature_{species}_gen.pt"

    print(f"[{species}] Loading real features from: {real_path}")
    print(f"[{species}] Loading generated features from: {gen_path}")

    real = torch.load(real_path).cpu()[:: args.subsample]
    gen = torch.load(gen_path).cpu()[:: args.subsample]

    print("Real feats shape:", real.shape)
    print("Gen  feats shape:", gen.shape)

    swd = ot.sliced_wasserstein_distance(real, gen)
    print(f"[{species}] Sliced Wasserstein Distance:", swd)
