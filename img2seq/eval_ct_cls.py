import scanpy as sc
from scipy.stats import pearsonr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

### 0. Initialization
### 0.1. Set device
device = "cuda:1" if torch.cuda.is_available() else "cpu"


### 0.2. Define the MLP architecture
class MLP(nn.Module):
    """
    Simple MLP classifier that maps gene expression vectors to cell-type logits.

    Input:  2000-dimensional gene expression vector
    Hidden: 512 units with ReLU activation
    Output: 5-class logits (one per cell type)
    """

    def __init__(self):
        super(MLP, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(2000, 512)
        self.relu = nn.ReLU()
        # Output layer (5 cell types)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


### 1. Load data
### 1.1. Load AnnData and labels
adata = sc.read("./data/IRIS_human.h5ad")

# Cell-type labels (categorical)
labels = adata.obs["cell_type"]

# Encode cell-type labels as integers [0, n_classes-1]
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)
classes = label_encoder.classes_

### 1.2. Load gene expression features
# Ground-truth gene expression (e.g., measured transcriptomics)
gene_exp_gt = torch.load("./features/gt.pt", map_location=torch.device("cpu"))

# Predicted gene expression (e.g., from an image-to-seq model)
gene_exp_pred = torch.load("./features/pred.pt", map_location=torch.device("cpu"))


### 2. Dataset and dataloaders
class CustomImageDataset(Dataset):
    """
    Dataset that returns (label, gene_exp_gt, gene_exp_pred) for each cell.

    Args:
        label:         1D array of integer labels (cell types).
        gene_exp_gt:   Tensor of ground-truth gene expression [n_cells, n_genes].
        gene_exp_pred: Tensor of predicted gene expression [n_cells, n_genes].
        stage:         'train' or 'test'. For 'train', indices are downsampled
                       by taking only even indices (idx // 2 * 2) to match
                       how other parts of the pipeline were constructed.
    """

    def __init__(self, label, gene_exp_gt, gene_exp_pred, stage):
        self.len = gene_exp_gt.shape[0]
        self.label = label
        self.gene_exp_gt = gene_exp_gt
        self.gene_exp_pred = gene_exp_pred
        self.stage = stage

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        Returns:
            label:       integer cell-type label
            gene_exp_gt: ground-truth gene expression vector
            gene_exp_pred: predicted gene expression vector
        """
        if self.stage == "train":
            # For training, use only even indices to keep alignment with paired data
            base_idx = idx // 2 * 2
            return (
                self.label[base_idx],
                self.gene_exp_gt[base_idx],
                self.gene_exp_pred[base_idx],
            )
        else:
            # For testing, use all indices
            return (
                self.label[idx],
                self.gene_exp_gt[idx],
                self.gene_exp_pred[idx],
            )


### 2.1. Construct dataloaders
dataset_train = CustomImageDataset(
    numeric_labels, gene_exp_gt, gene_exp_pred, stage="train"
)
dataloader_train = DataLoader(
    dataset_train,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

dataset_test = CustomImageDataset(
    numeric_labels, gene_exp_gt, gene_exp_pred, stage="test"
)
dataloader_test = DataLoader(
    dataset_test,
    batch_size=256,
    shuffle=False,
    drop_last=False,
    num_workers=8,
)


### 3. Initialize model, loss, and optimizer
model = MLP().to(device)

# Load pretrained classifier weights (optional fine-tuning)
model.load_state_dict(torch.load("./ckpt/celltype_classifier.pt"))

# Cross-entropy loss for multi-class classification
criterion = nn.CrossEntropyLoss()

# Adam optimizer for MLP parameters
optimizer = optim.Adam(model.parameters(), lr=0.0001)


### 4. Training and evaluation loop
epochs = 1  # Number of epochs

for epoch in range(epochs):

    # ------------------------
    # Training phase
    # ------------------------
    loss_acc_train = 0.0
    sample_num_train = 0

    for target_labels, input_gene_exp_gt, input_gene_exp_pred in dataloader_train:
        # Convert labels to one-hot vectors with 5 classes
        # (Note: CrossEntropyLoss typically expects integer labels; this
        # one-hot usage assumes compatibility with your environment/pipeline.)
        target_labels = torch.nn.functional.one_hot(
            target_labels.to(dtype=torch.long),
            num_classes=5,
        ).float().to(device)

        input_gene_exp_gt = input_gene_exp_gt.to(device)
        input_gene_exp_pred = input_gene_exp_pred.to(device)

        # Debug print for sanity checks on value ranges
        print(torch.max(input_gene_exp_gt), torch.max(input_gene_exp_pred))

        model.train()

        # Forward pass with ground-truth gene expression as input
        output = model(input_gene_exp_gt)

        # Compute classification loss
        loss = criterion(output, target_labels)

        loss_acc_train += loss.item()
        sample_num_train += 1

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(
        f"Epoch {epoch}/{epochs}, "
        f"Train Loss: {loss_acc_train / sample_num_train}"
    )

    # ------------------------
    # Evaluation phase
    # ------------------------
    loss_acc_test = 0.0
    sample_num_test = 0

    all_labels = []
    all_preds_gt = []
    all_preds_pred = []

    running_corrects_gt = 0
    running_corrects_pred = 0
    count = 0

    with torch.no_grad():
        for target_labels, input_gene_exp_gt, input_gene_exp_pred in dataloader_test:
            # Convert labels to one-hot vectors
            target_labels = torch.nn.functional.one_hot(
                target_labels.to(dtype=torch.long),
                num_classes=5,
            ).float().to(device)

            input_gene_exp_gt = input_gene_exp_gt.to(device)
            input_gene_exp_pred = input_gene_exp_pred.to(device)

            model.eval()

            # Forward pass:
            # - outputs_gt: classifier applied to ground-truth expression
            # - outputs_pred: classifier applied to predicted expression
            outputs_gt = model(input_gene_exp_gt)
            outputs_pred = model(input_gene_exp_pred)

            # Convert one-hot labels and logits to integer class indices
            _, gt = torch.max(target_labels, 1)
            _, preds_gt = torch.max(outputs_gt, 1)
            _, preds_pred = torch.max(outputs_pred, 1)

            # Merge class 3 into class 0 (dataset-specific decision)
            preds_gt[preds_gt == 3] = 0
            preds_pred[preds_pred == 3] = 0
            gt[gt == 3] = 0

            all_labels.extend(gt.cpu().numpy())

            # Count correct predictions
            running_corrects_gt += torch.sum(preds_gt == gt)
            running_corrects_pred += torch.sum(preds_pred == gt)
            count += outputs_gt.shape[0]

            sample_num_test += 1

    all_labels = torch.from_numpy(np.array(all_labels))

    # Compute accuracies for classifier on GT vs. predicted expression
    epoch_acc_gt = running_corrects_gt.double() / count
    epoch_acc_pred = running_corrects_pred.double() / count

    print(
        f"Epoch {epoch}/{epochs}, "
        f"Acc test (GT expr): {epoch_acc_gt}, "
        f"Acc test (Pred expr): {epoch_acc_pred}"
    )
