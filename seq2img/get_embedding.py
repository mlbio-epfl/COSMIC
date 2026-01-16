import warnings
warnings.filterwarnings("ignore")

import argparse
import os

import torch
import torchvision
from timm.models.vision_transformer import vit_large_patch16_224
from torch import nn
from torch.utils.data import Dataset, DataLoader

from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform

import scanpy as sc
import torchvision.transforms as T

### 0. CLI + Initialization
### 0.1. Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Extract MAE image embeddings for single-cell nuclei."
)
parser.add_argument(
    "--species",
    type=str,
    choices=["mouse", "human"],
    default="mouse",
    help="Species to process (mouse or human).",
)
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Torch device string, e.g. 'cuda:0' or 'cpu'. "
         "If not set, will use 'cuda:2' if available, else 'cpu'.",
)

args = parser.parse_args()
species = args.species

### 0.2. Set device
if args.device is not None:
    device_str = args.device
else:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

### 0.3. Paths and filenames
DATA_DIR = "./data"
IMG_DIR = "./seq2img/inference"
FEATURE_ROOT = "./seq2img/feature"
os.makedirs(FEATURE_ROOT, exist_ok=True)

adata_paths = {
    "mouse": os.path.join(DATA_DIR, "IRIS_mouse.h5ad"),
    "human": os.path.join(DATA_DIR, "IRIS_human.h5ad"),
}

image_dirs = {
    "mouse": os.path.join(DATA_DIR, "images_mouse"),
    "human": os.path.join(DATA_DIR, "images_human"),
}

adata_path = adata_paths[species]
image_dir = image_dirs[species]
ckpt_path = "./ckpt/ckpt_morphFM.pt"

# Final output file, e.g. './seq2img/feature/feature_mouse_morphFM.pt'
feature_out_path = os.path.join(FEATURE_ROOT, f"feature_{species}_gen_morphFM.pt")


### 1. MAE model definition
class MAE(nn.Module):
    """
    Masked Autoencoder (MAE) wrapper around a timm Vision Transformer.

    - Uses a MaskedVisionTransformerTIMM backbone for encoding.
    - Uses a MAEDecoderTIMM for reconstructing masked image patches.
    - Here, we mainly use the encoder to extract per-cell image features.

    Notes:
    - mask_ratio is set to 0.0 (no masking) so we effectively only use the encoder.
    """

    def __init__(self, vit):
        super().__init__()

        decoder_dim = 512
        self.mask_ratio = 0.0  # no masking, full image is encoded
        self.patch_size = vit.patch_embed.patch_size[0]

        # MAE backbone (ViT encoder with masking support)
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length

        # Decoder used for reconstruction (kept for completeness)
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=1,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

    def forward_encoder(self, images, idx_keep=None):
        """
        Encode images with the masked ViT backbone.

        Args:
            images: Tensor of shape [B, 3, H, W]
            idx_keep: indices of tokens to keep (for masking)

        Returns:
            x_encoded: sequence of encoded tokens, shape [B, L, C]
        """
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        """
        Decode encoded tokens back to image patches for masked tokens.

        Args:
            x_encoded: encoded tokens
            idx_keep: indices of kept tokens
            idx_mask: indices of masked tokens

        Returns:
            x_pred: predicted pixel values for masked patches
        """
        batch_size = x_encoded.shape[0]

        # Embed encoded tokens
        x_decode = self.decoder.embed(x_encoded)

        # Create a full token sequence filled with mask tokens
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )

        # Insert encoded tokens back at kept positions
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # Decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # Predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images):
        """
        Forward pass for feature extraction.

        Args:
            images: Tensor [B, 3, H, W]

        Returns:
            features: Tensor [B, C] global image embeddings.
        """
        batch_size = images.shape[0]

        # Random token mask; mask_ratio=0.0 → effectively keep all tokens
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        # Encode images
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)

        # x_encoded: [B, L, C]; average over tokens → [B, C]
        features = x_encoded.mean(dim=1)

        # Optionally, we could decode here, but for embeddings it is not needed
        # x_pred = self.forward_decoder(
        #     x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        # )

        return features


### 1.1. Build and load pretrained MAE model
vit = vit_large_patch16_224()
model = MAE(vit)

# Load pretrained MAE checkpoint (species-specific path)
model.load_state_dict(torch.load(ckpt_path, map_location=device))

# Freeze all weights (we only use it as a feature extractor)
for param in model.parameters():
    param.requires_grad = False

model = model.to(device)

print(f"Finished building the model for species = {species} on device = {device}")


### 2. Dataset and dataloader for image feature extraction
transform = MAETransform()  # defined but not used explicitly below


class CustomImageDataset(Dataset):
    """
    Dataset that returns (image, cell_id) pairs for single-cell nuclei.

    - Loads metadata from '<DATA_DIR>/IRIS_{species}.h5ad'
    - Uses 'cell_id' to locate the corresponding image on disk:
        '<image_dir>/{cell_id}.jpg'
    """

    def __init__(self):
        adata_seq = sc.read(adata_path)
        self.cell_id = adata_seq.obs["cell_id"]
        self.len = len(self.cell_id)

        # Resize to 224 x 224 to match ViT input
        self.transform = T.Compose([T.Resize(224)])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Returns:
            image:    Tensor [3, 224, 224] normalized to [0, 1]
            cell_id:  identifier string for the cell
        """
        cell_id_tmp = self.cell_id[idx]

        # Load image and normalize to [0, 1]
        image = (
            torchvision.io.read_image(
                os.path.join(image_dir, f"{cell_id_tmp}.jpg")
            ).float()
            / 255.0
        )

        # Resize and force 3 channels (if grayscale)
        image = self.transform(image)
        image = image.expand(3, 224, 224)

        return image, cell_id_tmp


# Create dataset and dataloader for all images
dataset_all = CustomImageDataset()
dataloader = DataLoader(
    dataset_all,
    batch_size=32,
    shuffle=False,
    num_workers=8,
)

### 3. Feature extraction loop (no per-cell saves, just concatenate)
print("Starting feature extraction")

all_features = []

model.eval()
with torch.no_grad():
    for images, cell_ids in dataloader:
        images = images.to(device)
        features_batch = model(images)          # [B, C]
        all_features.append(features_batch.cpu())

# Concatenate all features [n_cells, C]
feature = torch.cat(all_features, dim=0)

torch.save(feature, feature_out_path)

print(f"Saved aggregated features to {feature_out_path}")
