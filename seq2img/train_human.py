import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T

from imagen_pytorch import Unet, Imagen, ImagenTrainer
from PIL import Image
import os
import scanpy as sc

import os.path

### 0. Initilization
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

### 0.2. Set Dataset
class CustomImageDataset(Dataset):
    """
    Dataset that returns (image, feature) pairs for single cells.

    - Loads precomputed features from 'feature_human.pt'
    - Loads cell metadata (including cell_id) from 'IRIS_human.h5ad'
    - Uses cell_id to locate the corresponding image on disk
    """

    def __init__(self, ):
        # Precomputed feature tensor of shape [n_cells, feature_dim]
        self.feature = torch.load('./seq2img/feature/feature_human_scvi.pt')

        # AnnData with per-cell metadata, including cell_id
        adata_seq = sc.read('./data/IRIS_human.h5ad')

        # Store cell IDs (used to construct image file paths)
        self.cell_id = adata_seq.obs['cell_id']

        # Number of samples in the dataset
        self.len = self.feature.shape[0]

        # Resize all images to 256 x 256 (keeps aspect ratio by default)
        self.transform = T.Resize(256)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape [3, 256, 256], normalized to [0, 1]
            feature: Tensor corresponding to the same cell
        """

        # Use even index to ensure image-feature pairing consistency
        # (assumes features/images are duplicated or ordered in pairs)
        idx_tmp = idx // 2 * 2
        cell_id_tmp = self.cell_id[idx_tmp]

        # Load image from disk, convert to float and normalize to [0, 1]
        image = torchvision.io.read_image(
            f'./data/images_human/{cell_id_tmp}.jpg'
        ).float() / 255.0

        # Resize and force 3 channels (expand grayscale to RGB)
        image = self.transform(image)
        image = image.expand(3, 256, 256)

        # Return image and its corresponding feature vector
        return image, self.feature[idx_tmp]

### 0.3. Select device (GPU if available)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

### 1. Set up models, optimizers, and dataloaders
### 1.1. Define Unet backbone used inside the Imagen model
unet1 = Unet(
    dim=32,                 # base channel dimension
    cond_dim=64,            # dimension of conditioning vector (here: feature embedding)
    dim_mults=(1, 2, 4, 8), # channel multipliers per resolution level
    num_resnet_blocks=1,    # number of ResNet blocks per level
    layer_attns=(False, False, False, True),        # enable self-attention only at the highest resolution
    layer_cross_attns=(False, False, False, True)   # enable cross-attention with condition at the highest resolution
)

### 1.2. Define Imagen model (single-Unet configuration)
imagen = Imagen(
    unets=(unet1),          # a single Unet used as base generator
    image_sizes=(256),      # target image size (H = W = 256)
    timesteps=1000,         # number of diffusion time steps
    cond_drop_prob=0.1,     # probability of dropping condition during training (classifier-free guidance)
    pred_objectives='x_start',  # predict clean image x_0 instead of noise
    text_embed_dim=64       # dimension of conditioning embeddings (must match cond_dim)
)

imagen = imagen.to(device)

### 1.3. AdamW optimizer for the Imagen parameters
optimizer = torch.optim.AdamW(imagen.parameters(), lr=1.5e-4)

### 1.4. Dataloaders
dataset = CustomImageDataset()
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

### 2. Training loop
os.makedirs("./seq2img/result/human", exist_ok=True)
os.makedirs("./ckpt", exist_ok=True)

iters = -1
for epoch in range(100):
    for img, z in dataloader:

        iters += 1
        print('iters:', iters)

        img = img.to(device)
        z = z.to(device)

        # Add a singleton sequence dimension for conditioning
        # Expected shape: [batch_size, cond_seq_len, cond_dim]
        z = z.unsqueeze(1)

        # Forward pass through Imagen; returns diffusion loss
        loss = imagen(img, text_embeds=z, unet_number=1)

        # Standard backward + optimization step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Periodically sample and save generated / ground truth images
        if iters % 1000 == 0:
            # Sample images conditioned on the current batch of features
            images = imagen.sample(text_embeds=z, stop_at_unet_number=1)

            # Save first generated image in batch
            tensor = (images[0].permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(tensor)
            image.save(f'./seq2img/result/human/{iters}_test.png')

            # Save corresponding ground truth image for visual comparison
            tensor = (img[0].permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(tensor)
            image.save(f'./seq2img/result/human/{iters}_gt.png')

        # Periodically checkpoint model weights
        if iters % 1000 == 0:
            torch.save(imagen.state_dict(), f'./ckpt/seq2img_human.pt')
