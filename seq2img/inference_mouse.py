import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T

from imagen_pytorch import Unet, Imagen, ImagenTrainer
from PIL import Image
import random
import scanpy as sc

import os

### 0. Initialization
### 0.1. Set random seed
def set_seed(seed=42):
    """
    Set random seeds for reproducibility across CPU, single-GPU and multi-GPU.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True  # deterministic convs (slower)
    torch.backends.cudnn.benchmark = False     # disable auto-tuner
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)

### 0.2. Set device and batch size
# Use GPU if available (here specifically cuda:5)
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# Batch size used during sampling
bs = 16


### 0.3. Dataset for latent features only (no images)
class CustomImageDataset(Dataset):
    """
    Dataset that returns (cell_id, feature) pairs for single cells (mouse).

    - Loads precomputed latent features from 'feature_mouse_scvi.pt'
    - Loads cell metadata (including cell_id) from 'IRIS_mouse.h5ad'
    - Used for conditional image generation (seq2img), not for training
    """

    def __init__(self, ):
        # Precomputed feature tensor of shape [n_cells, feature_dim]
        self.feature = torch.load('./seq2img/feature/feature_mouse_scvi.pt')

        # AnnData with per-cell metadata, including cell_id
        adata_seq = sc.read('./data/IRIS_mouse.h5ad')

        # Store cell IDs (used for naming output images)
        self.cell_id = adata_seq.obs['cell_id']

        # Number of samples in the dataset
        self.len = self.feature.shape[0]

        # Defined for consistency with other datasets (not used here)
        self.transform = T.Resize(256)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Returns:
            cell_id: str or categorical label for the cell
            feature: Tensor latent embedding corresponding to this cell
        """
        idx_tmp = idx
        cell_id_tmp = self.cell_id[idx_tmp]

        return cell_id_tmp, self.feature[idx_tmp]


### 1. Set up Imagen model and load checkpoint
### 1.1. Define Unet backbone used inside the Imagen model
unet1 = Unet(
    dim=32,                  # base channel dimension
    cond_dim=64,             # dimension of conditioning vector (here: feature embedding)
    dim_mults=(1, 2, 4, 8),  # channel multipliers per resolution level
    num_resnet_blocks=1,     # number of ResNet blocks per level
    layer_attns=(False, False, False, True),        # enable self-attention only at the highest resolution
    layer_cross_attns=(False, False, False, True)   # enable cross-attention with condition at the highest resolution
)

### 1.2. Define Imagen model (single-Unet configuration)
imagen = Imagen(
    unets=(unet1),           # a single Unet used as base generator
    image_sizes=(256),       # target image size (H = W = 256)
    timesteps=1000,          # number of diffusion time steps
    cond_drop_prob=0.1,      # probability of dropping condition during training (classifier-free guidance)
    pred_objectives='x_start',  # predict clean image x_0 instead of noise
    text_embed_dim=64        # dimension of conditioning embeddings (must match cond_dim)
)

### 1.3. Load pretrained weights from checkpoint
checkpoint = torch.load(
    '/mlbio_scratch/wen2/cross-model-gen/img_dec/imagen/ckpt/mouse_config0/267000.pt',
    map_location=device
)
imagen.load_state_dict(checkpoint, strict=True)
imagen = imagen.to(device)


### 2. Dataloader (no shuffling to preserve index ↔ cell_id mapping)
dataset = CustomImageDataset()
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)


### 3. Generate images from sequence features (seq2img)
# Output images are saved to: /mlbio_scratch/wen2/cross-model-gen/img_dec/imagen/test/mouse_config0/
os.makedirs(
    './seq2img/inference/mouse',
    exist_ok=True
)

with torch.no_grad():
    for idx, z in dataloader:
        # idx: batch of cell_ids (pandas index / categorical labels)
        # z:   batch of latent features (Tensor)

        # Move latent features to device
        z = z.to(device)

        # Add a singleton sequence dimension for conditioning
        # Expected shape: [batch_size, cond_seq_len, cond_dim]
        z = z.unsqueeze(1)

        # Sample images conditioned on latent features using the trained Imagen model
        images_fromseq = imagen.sample(text_embeds=z, stop_at_unet_number=1)

        # Save one image per cell in the batch
        # Use actual batch size instead of fixed 'bs' to handle final partial batch
        for j in range(len(idx)):
            # Convert cell index or ID to Python native type for file naming
            idx_tmp = idx[j]

            # Convert image tensor from [C, H, W] in [0,1] to uint8 [H, W, C] in [0,255]
            tensor = (
                images_fromseq[j]
                .permute(1, 2, 0) * 255
            ).clamp(0, 255).to(torch.uint8).cpu().numpy()

            image = Image.fromarray(tensor)

            # Use cell_id as filename to align generated images with cells
            image.save(
                f'./seq2img/inference/mouse/{idx_tmp}.png'
            )
