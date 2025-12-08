import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import random

import scanpy as sc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

import torchvision
import torchvision.transforms as T


### 0. Initialization
### 0.1. Fix random seeds
def set_seed(seed=42):
    """
    Set random seeds for reproducibility across:
    - Python's `random`
    - NumPy
    - PyTorch (CPU, single-GPU, multi-GPU)
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    torch.backends.cudnn.deterministic = True   # deterministic convs (slower)
    torch.backends.cudnn.benchmark = False      # disable auto-tuner


set_seed(42)


### 0.2. Parse command-line arguments
parser = argparse.ArgumentParser(
    description=(
        "Evaluate how well image-based classifiers preserve cell-type identity "
        "on real vs generated images using a simple CNN backbone."
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
    "--device",
    type=str,
    default=None,
    help=(
        "Torch device string, e.g. 'cuda:0' or 'cpu'. "
        "If not set, will use 'cuda:1' if available, else 'cpu'."
    ),
)
parser.add_argument(
    "--ckpt_dir",
    type=str,
    default="./ckpt/image_classifier",
    help="Directory to save model checkpoints.",
)

args = parser.parse_args()
species = args.species
ckpt_dir = args.ckpt_dir

# Make sure checkpoint directory exists
os.makedirs(ckpt_dir, exist_ok=True)


### 0.3. Set device
if args.device is not None:
    device_str = args.device
else:
    device_str = "cuda:3" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

print(f"Running IMAGE classifier evaluation (SimpleCNN) for species = {species} on device = {device}")


### 0.4. Paths
# Ground-truth images: f'/mlbio_scratch/.../images_{species}/{cell_id}.jpg'
gt_image_dir = f"/mlbio_scratch/wen2/cross-model-gen/github/seq2img/data/images_{species}"

# Generated images: cell_id.png in config-specific folders
if species == "human":
    gen_image_dir = "/mlbio_scratch/wen2/cross-model-gen/img_dec/imagen/test/human_config1"
elif species == "mouse":
    gen_image_dir = "/mlbio_scratch/wen2/cross-model-gen/img_dec/imagen/test/mouse_config0"
else:
    raise ValueError(f"Unsupported species: {species}")

# AnnData with labels
if species == "human":
    adata_path = "/mlbio_scratch/wen2/cross-model-gen/img_dec/imagen/human_300725.h5ad"
else:
    adata_path = "/mlbio_scratch/wen2/cross-model-gen/img_dec/imagen/mouse_300725.h5ad"


### 0.5. Simple CNN backbone
class SimpleCNN(nn.Module):
    """
    Simple convolutional neural network for image classification.

    Assumes input images are resized to 256 x 256:
      - 6 pooling layers (2x2, stride 2) → 256 / 2^6 = 4
      - Flattened feature size: 256 * 4 * 4

    The final fc layer is configured with `num_classes`.
    """

    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()
        # 1st Convolutional Block
        self.conv0_0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv0_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 2nd Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 3rd Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # 4th Convolutional Layer
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 4 * 4, 128)  # for input size 256x256 with 6 pools
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input: [B, 3, 256, 256]
        x = self.pool(F.relu(self.conv0_0(x)))  # 256 -> 128
        x = self.pool(F.relu(self.conv0_1(x)))  # 128 -> 64
        x = self.pool(F.relu(self.conv1(x)))    # 64 -> 32
        x = self.pool(F.relu(self.conv2(x)))    # 32 -> 16
        x = self.pool(F.relu(self.conv3(x)))    # 16 -> 8
        x = self.pool(F.relu(self.conv4(x)))    # 8 -> 4

        # Flatten
        x = x.view(-1, 256 * 4 * 4)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


### 1. Load data and labels
### 1.1. Load AnnData and cell-type labels
adata = sc.read(adata_path)
if species == "human":
    adata.obs.loc[adata.obs["cell_type"] == "CD8+", "cell_type"] = "PBMC"

labels = adata.obs["cell_type"]

# Encode cell-type labels as integers [0, n_classes-1]
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)
classes = label_encoder.classes_

print("Number of cells:", len(labels))
print("Encoded labels range:", int(numeric_labels.min()), "to", int(numeric_labels.max()))
print("Classes:", classes)


### 2. Train / test split and datasets
class ImageCellDataset(Dataset):
    """
    Dataset that returns (image, label) pairs for single-cell nuclei.

    - Uses AnnData for cell metadata and labels.
    - Uses 'cell_id' to locate the corresponding image on disk.
    - Extension is configurable ('.jpg' for GT, '.png' for generated).
    """

    def __init__(self, adata, numeric_labels, image_dir, img_ext=".jpg"):
        self.adata = adata
        self.labels = numeric_labels
        self.image_dir = image_dir
        self.img_ext = img_ext

        self.cell_ids = self.adata.obs["cell_id"].to_numpy()
        self.len = len(self.cell_ids)

        # Resize to 256x256 to match SimpleCNN expectations
        self.transform = T.Compose([
            T.Resize((256, 256)),
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor [3, 256, 256] in [0, 1]
            label: int (cell-type index)
        """
        cell_id = self.cell_ids[idx]
        label = self.labels[idx]

        img_path = os.path.join(self.image_dir, f"{cell_id}{self.img_ext}")

        # Load image and normalize to [0, 1]
        image = torchvision.io.read_image(img_path).float() / 255.0

        # Resize and force 3 channels (if grayscale)
        image = self.transform(image)
        image = image.expand(3, 256, 256)

        return image, label


class MixedTrainDataset(Dataset):
    """
    Training dataset that mixes ground-truth and generated images.

    For each cell in the train split we create two samples:
      - index 2*k:   ground-truth image (jpg)
      - index 2*k+1: generated image (png)

    This way, each epoch sees both real and generated images for all training cells.
    """

    def __init__(self, adata_train, labels_train, gt_dir, gen_dir):
        self.cell_ids = adata_train.obs["cell_id"].to_numpy()
        self.labels = labels_train
        self.gt_dir = gt_dir
        self.gen_dir = gen_dir
        self.len_cells = len(self.cell_ids)
        self.len = self.len_cells * 2  # 2 samples per cell (gt + gen)

        self.transform = T.Compose([
            T.Resize((256, 256)),
        ])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor [3, 256, 256] in [0, 1]
            label: int
        """
        cell_idx = idx // 2
        use_gen = (idx % 2 == 1)

        cell_id = self.cell_ids[cell_idx]
        label = self.labels[cell_idx]

        if use_gen:
            img_path = os.path.join(self.gen_dir, f"{cell_id}.png")
        else:
            img_path = os.path.join(self.gt_dir, f"{cell_id}.jpg")

        image = torchvision.io.read_image(img_path).float() / 255.0
        image = self.transform(image)
        image = image.expand(3, 256, 256)

        return image, label


# Split indices: ::2 for train, 1::2 for test
idx_all = np.arange(len(adata.obs))
idx_train = idx_all[::2]
idx_test = idx_all[1::2]

# Subset AnnData and labels
adata_train = adata[idx_train, :].copy()
adata_test = adata[idx_test, :].copy()
labels_train = numeric_labels[idx_train]
labels_test = numeric_labels[idx_test]

print(f"Train cells: {len(adata_train)}, Test cells: {len(adata_test)}")

# Mixed train dataset (GT + generated)
mix_dataset_train = MixedTrainDataset(
    adata_train=adata_train,
    labels_train=labels_train,
    gt_dir=gt_image_dir,
    gen_dir=gen_image_dir,
)

# GT test dataset
gt_dataset_test = ImageCellDataset(
    adata=adata_test,
    numeric_labels=labels_test,
    image_dir=gt_image_dir,
    img_ext=".jpg",
)

# Generated test dataset
gen_dataset_test = ImageCellDataset(
    adata=adata_test,
    numeric_labels=labels_test,
    image_dir=gen_image_dir,
    img_ext=".png",
)

# Dataloaders
mix_loader_train = DataLoader(
    mix_dataset_train,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

gt_loader_test = DataLoader(
    gt_dataset_test,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    num_workers=8,
)

gen_loader_test = DataLoader(
    gen_dataset_test,
    batch_size=64,
    shuffle=False,
    drop_last=False,
    num_workers=8,
)


### 3. Initialize model, loss, and optimizer
num_classes = len(classes)
model = SimpleCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


### 4. Training, evaluation, and checkpointing
epochs = 50  # Number of epochs
best_acc_gt = 0.0  # track best GT-test accuracy for checkpointing

for epoch in range(epochs):

    # ------------------------
    # Training phase (on MIXED GT + GEN train images)
    # ------------------------
    model.train()
    loss_acc_train = 0.0
    num_train_batches = 0

    for images, labels_int in mix_loader_train:
        images = images.to(device)
        labels_int = labels_int.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels_int)

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_acc_train += loss.item()
        num_train_batches += 1

    avg_train_loss = loss_acc_train / max(1, num_train_batches)
    print(
        f"Epoch {epoch}/{epochs}, "
        f"Train Loss (mixed GT+GEN train images): {avg_train_loss:.4f}"
    )

    # ------------------------
    # Evaluation on GT test images
    # ------------------------
    model.eval()

    running_corrects_gt = 0
    count_gt = 0

    with torch.no_grad():
        for images, labels_int in gt_loader_test:
            images = images.to(device)
            labels_int = labels_int.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            running_corrects_gt += torch.sum(preds == labels_int)
            count_gt += images.size(0)

    acc_gt = running_corrects_gt.double() / max(1, count_gt)

    # ------------------------
    # Evaluation on GENERATED test images
    # ------------------------
    running_corrects_gen = 0
    count_gen = 0

    with torch.no_grad():
        for images, labels_int in gen_loader_test:
            images = images.to(device)
            labels_int = labels_int.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            running_corrects_gen += torch.sum(preds == labels_int)
            count_gen += images.size(0)

    acc_gen = running_corrects_gen.double() / max(1, count_gen)

    print(
        f"Epoch {epoch}/{epochs}, "
        f"Acc (GT test images): {acc_gt:.4f}, "
        f"Acc (Generated test images): {acc_gen:.4f}"
    )

    # ------------------------
    # Checkpointing
    # ------------------------
    # Save checkpoint each epoch
    ckpt_path_epoch = os.path.join(
        ckpt_dir,
        f"{species}_image_classifier_ep{epoch}_{acc_gt:.4f}_{acc_gen:.4f}.pt"
    )
    torch.save(model.state_dict(), ckpt_path_epoch)

    # Save best model (based on GT test accuracy)
    if acc_gt > best_acc_gt:
        best_acc_gt = float(acc_gt)
        ckpt_path_best = os.path.join(
            ckpt_dir,
            f"{species}_image_classifier_best.pt"
        )
        torch.save(model.state_dict(), ckpt_path_best)
        print(f"  -> New best model saved with GT test acc = {best_acc_gt:.4f}")

print("Image classification training and evaluation complete!")
