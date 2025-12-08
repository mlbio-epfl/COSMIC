import warnings
warnings.filterwarnings("ignore")

import argparse

import scanpy as sc
from scipy.stats import pearsonr, spearmanr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import itertools

### 0. CLI + Initialization
### 0.1. Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Evaluate how well predicted gene expression preserves cell-type identity."
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
         "If not set, will use 'cuda:1' if available, else 'cpu'.",
)

args = parser.parse_args()
species = args.species

### 0.2. Set device
if args.device is not None:
    device_str = args.device
else:
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

print(f"Running classifier evaluation for species = {species} on device = {device}")


### 0.3. Define the MLP architecture
class MLP(nn.Module):
    """
    Simple MLP classifier that maps gene expression vectors to cell-type logits.

    Input:
        - in_dim:  dimension of gene expression vector
    Hidden:
        - 512 units with ReLU activation
    Output:
        - out_dim logits (number of cell types)
    """

    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


### 1. Load data
### 1.1. Load AnnData and cell-type labels
if species == "human":
    adata = sc.read("/mlbio_scratch/wen2/cross-model-gen/img_dec/imagen/human_300725.h5ad")
    # Merge CD8+ into PBMC for this analysis
    adata.obs.loc[adata.obs["cell_type"] == "CD8+", "cell_type"] = "PBMC"
elif species == "mouse":
    adata = sc.read("/mlbio_scratch/wen2/cross-model-gen/img_dec/imagen/mouse_300725.h5ad")
else:
    raise ValueError(f"Unsupported species: {species}")

labels = adata.obs["cell_type"]

# Encode cell-type labels as integers [0, n_classes-1]
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)
classes = label_encoder.classes_

print("Encoded labels range:", int(numeric_labels.min()), "to", int(numeric_labels.max()))
print("Classes:", classes)


### 1.2. Load ground-truth and predicted gene expression
# Each file is expected to contain:
#   - "gt_genes":   [n_cells, n_genes]
#   - "pred_genes": [n_cells, n_genes]
pred_path = (
    f"/mlbio_scratch/wen2/cross-model-gen/seq_dec_diff_latest/genes/"
    f"{species}_0.5/predictions/hybrid_diffusion_residual_predictions_test_{species}.pt"
)
predictions = torch.load(pred_path, map_location="cpu")
gene_exp_gt = predictions["gt_genes"]
gene_exp_pred = predictions["pred_genes"]

print("GT shape:   ", gene_exp_gt.shape)
print("Pred shape: ", gene_exp_pred.shape)


### 2. Dataset and dataloaders
class CustomImageDataset(Dataset):
    """
    Dataset that returns (label, gene_exp_gt, gene_exp_pred) for each cell.

    Args:
        label:         1D array of integer labels (cell types).
        gene_exp_gt:   Tensor of ground-truth gene expression [n_cells, n_genes].
        gene_exp_pred: Tensor of predicted gene expression [n_cells, n_genes].
        stage:         'train' or 'test'.

    Notes on indexing:
        - For 'train', we sample from odd indices (2k+1), clamped to be within range.
        - For 'test', we sample from even indices (2k).
        This matches a pairing scheme where neighboring indices belong to the same cell.
    """

    def __init__(self, label, gene_exp_gt, gene_exp_pred, stage):
        self.len = gene_exp_pred.shape[0]
        self.label = label
        self.gene_exp_gt = gene_exp_gt
        self.gene_exp_pred = gene_exp_pred
        self.stage = stage

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Returns:
            label:          integer cell-type label
            gene_exp_gt:    ground-truth gene expression vector
            gene_exp_pred:  predicted gene expression vector
        """
        if self.stage == "train":
            # Use odd indices (2k+1), clamped to max valid odd index
            max_odd = self.len // 2 * 2 - 1
            base_idx = min(idx // 2 * 2 + 1, max_odd)
        else:
            # Use even indices (2k)
            base_idx = idx // 2 * 2

        return (
            self.label[base_idx],
            self.gene_exp_gt[base_idx],
            self.gene_exp_pred[base_idx],
        )


dataset_train = CustomImageDataset(numeric_labels, gene_exp_gt, gene_exp_pred, stage="train")
dataloader_train = DataLoader(
    dataset_train,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

dataset_test = CustomImageDataset(numeric_labels, gene_exp_gt, gene_exp_pred, stage="test")
dataloader_test = DataLoader(
    dataset_test,
    batch_size=256,
    shuffle=False,
    drop_last=False,
    num_workers=8,
)


### 3. Initialize model, loss, and optimizer
in_dim = gene_exp_gt.shape[1]
out_dim = len(classes)

model = MLP(in_dim=in_dim, out_dim=out_dim).to(device)

# Cross-entropy loss for multi-class classification
criterion = nn.CrossEntropyLoss()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)


### 4. Training and evaluation loop
epochs = 5  # Number of epochs

for epoch in range(epochs):

    # ------------------------
    # Training phase
    # ------------------------
    loss_acc_train = 0.0
    sample_num_train = 0

    model.train()
    for target_labels, input_gene_exp_gt, input_gene_exp_pred in dataloader_train:
        # Convert integer labels to one-hot vectors with 'out_dim' classes
        target_labels_oh = torch.nn.functional.one_hot(
            target_labels.to(dtype=torch.long),
            num_classes=out_dim,
        ).float().to(device)

        input_gene_exp_gt = input_gene_exp_gt.to(device)
        input_gene_exp_pred = input_gene_exp_pred.to(device)

        # Forward pass using ground-truth expression
        outputs = model(input_gene_exp_gt)

        # Cross-entropy expects class indices, but here we pass one-hot;
        # if you want, you can switch to integer labels and call criterion(outputs, target_labels).
        loss = criterion(outputs, target_labels_oh)

        loss_acc_train += loss.item()
        sample_num_train += 1

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        f"Epoch {epoch}/{epochs}, "
        f"Train Loss: {loss_acc_train / sample_num_train:.4f}"
    )

    # ------------------------
    # Evaluation phase
    # ------------------------
    model.eval()

    running_corrects_gt = 0
    running_corrects_pred = 0
    count = 0

    all_labels = []

    with torch.no_grad():
        for target_labels, input_gene_exp_gt, input_gene_exp_pred in dataloader_test:
            # Convert to one-hot for consistency with training
            target_labels_oh = torch.nn.functional.one_hot(
                target_labels.to(dtype=torch.long),
                num_classes=out_dim,
            ).float().to(device)

            input_gene_exp_gt = input_gene_exp_gt.to(device)
            input_gene_exp_pred = input_gene_exp_pred.to(device)

            # Forward pass: classifier on GT vs predicted expression
            outputs_gt = model(input_gene_exp_gt)
            outputs_pred = model(input_gene_exp_pred)

            # Convert one-hot labels and logits to integer class indices
            _, gt = torch.max(target_labels_oh, 1)
            _, preds_gt = torch.max(outputs_gt, 1)
            _, preds_pred = torch.max(outputs_pred, 1)

            all_labels.extend(gt.cpu().numpy())

            # Count correct predictions
            running_corrects_gt += torch.sum(preds_gt == gt)
            running_corrects_pred += torch.sum(preds_pred == gt)
            count += outputs_gt.shape[0]

    all_labels = torch.from_numpy(np.array(all_labels))

    # Compute accuracies
    epoch_acc_gt = running_corrects_gt.double() / count
    epoch_acc_pred = running_corrects_pred.double() / count

    print(
        f"Epoch {epoch}/{epochs}, "
        f"Acc test (GT expr): {epoch_acc_gt:.4f}, "
        f"Acc test (Pred expr): {epoch_acc_pred:.4f}"
    )

print("Evaluation complete!")
