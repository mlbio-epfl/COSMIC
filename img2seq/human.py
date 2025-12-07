import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torchvision
from timm.models.vision_transformer import vit_large_patch16_224
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from lightly.models import utils
from lightly.models.modules import MaskedVisionTransformerTIMM

import scanpy as sc
from torchvision import transforms

# =========================
# Flags: choose which stages to run
# =========================
TRAIN_MSE_BASELINE = True          # Stage 1
TRAIN_RESIDUAL_DIFFUSION = True    # Stage 2
RUN_TEST = True                    # Stage 3

log_path = "training_loss_human.txt"

# =========================
# Paths
# =========================
MAE_PRETRAIN_PATH = './ckpt/mae_scimg_withoutIRIS_ep400.pt'
mse_ckpt_path = './ckpt/mse_baseline_human.pt'          # baseline-only checkpoint (encoder+regressor)
hybrid_ckpt_path = './ckpt/hybrid_full_model_human.pt'             # full unified model checkpoint

os.makedirs('./ckpt', exist_ok=True)
os.makedirs(os.path.dirname(hybrid_ckpt_path), exist_ok=True)

# =========================
# Utility: reproducibility
# =========================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

GENE_DIM = 17982

# =========================
# Diffusion setup (x0 on residual, regression-like)
# =========================
T = 6                   # fewer steps (was 10)
BETA_START = 5e-5       # smaller noise
BETA_END = 5e-4         # smaller noise
TIME_SKEW = 4.0         # stronger bias toward small t

# How strongly we trust the residual vs baseline at test time
LAMBDA_RESIDUAL = 0.3   # 0 = pure baseline, 1 = full residual, tune this

betas = torch.linspace(BETA_START, BETA_END, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

# register them on device when used
def to_device_diffusion_tensors(device):
    global betas, alphas, alphas_cumprod, alphas_cumprod_prev
    global sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance

    betas_ = betas.to(device)
    alphas_ = alphas.to(device)
    alphas_cumprod_ = alphas_cumprod.to(device)
    alphas_cumprod_prev_ = alphas_cumprod_prev.to(device)

    sqrt_alphas_cumprod_ = torch.sqrt(alphas_cumprod_)
    sqrt_one_minus_alphas_cumprod_ = torch.sqrt(1.0 - alphas_cumprod_)
    posterior_variance_ = betas_ * (1.0 - alphas_cumprod_prev_) / (1.0 - alphas_cumprod_)

    betas = betas_
    alphas = alphas_
    alphas_cumprod = alphas_cumprod_
    alphas_cumprod_prev = alphas_cumprod_prev_
    sqrt_alphas_cumprod = sqrt_alphas_cumprod_
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod_
    posterior_variance = posterior_variance_


def extract(a, t, x_shape):
    """Extract coefficients at given timesteps and reshape to [B, 1, ...]."""
    out = a.gather(-1, t)
    return out.view(-1, *([1] * (len(x_shape) - 1)))


def sample_timesteps(batch_size, device, skew=TIME_SKEW):
    """Sample timesteps with strong bias towards small t."""
    u = torch.rand(batch_size, device=device)  # U(0,1)
    t_cont = (u ** skew) * (T - 1)
    t = t_cont.long()
    return t

# =========================
# MAE encoder for image features
# =========================
class MAEEncoder(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.mask_ratio = 0.0
        self.patch_size = vit.patch_embed.patch_size[0]
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length

    def freeze_encoder_last_block_trainable(self):
        # freeze all
        for name, p in self.backbone.named_parameters():
            p.requires_grad = False
        # unfreeze only last ViT block
        for name, p in self.backbone.named_parameters():
            if 'vit.blocks.23' in name:
                p.requires_grad = True
        print("MAEEncoder: only last ViT block trainable.")

    def forward(self, images):
        b = images.shape[0]
        idx_keep, _ = utils.random_token_mask(
            size=(b, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.backbone.encode(images=images, idx_keep=idx_keep)
        feature = torch.mean(x_encoded, 1)  # [B, 1024]
        return feature

# =========================
# Baseline gene regressor
# =========================
class GeneRegressor(nn.Module):
    def __init__(self, in_dim=1024, gene_dim=GENE_DIM, hidden=2048):
        super().__init__()
        # Simple linear head
        self.net = nn.Linear(in_dim, gene_dim)

    def forward(self, feat):
        return self.net(feat)

# =========================
# Residual diffusion model (x0 on residual)
# =========================
class ResidualDiffusionModel(nn.Module):
    def __init__(
        self,
        gene_dim=GENE_DIM,
        cond_feat_dim=1024,
        cond_base_dim=GENE_DIM,
        time_dim=64,
        hidden_dim=2048,
        cond_scale=3.0,
    ):
        super().__init__()
        self.cond_scale = cond_scale

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # condition on residual r_t, MAE feature, baseline prediction
        in_dim = gene_dim + cond_feat_dim + cond_base_dim + time_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, gene_dim),
        )

    def forward(self, r_t, feat, base_pred, t):
        """
        r_t: noisy residual at time t [B, gene_dim]
        feat: MAE feature [B, 1024]
        base_pred: baseline prediction [B, gene_dim]
        t: timesteps [B]
        """
        feat = feat * self.cond_scale

        t = t.float().unsqueeze(-1) / T
        t_emb = self.time_mlp(t)

        x = torch.cat([r_t, feat, base_pred, t_emb], dim=-1)
        r0_hat = self.net(x)  # predicted clean residual
        return r0_hat

# =========================
# Unified model: encoder + regressor + residual diffusion
# =========================
class HybridGeneModel(nn.Module):
    def __init__(self, mae_pretrain_path=None):
        super().__init__()
        vit = vit_large_patch16_224()
        self.encoder = MAEEncoder(vit)
        if mae_pretrain_path is not None and os.path.exists(mae_pretrain_path):
            print(f"[HybridModel] Loading MAE pretrain from {mae_pretrain_path}")
            self.encoder.load_state_dict(
                torch.load(mae_pretrain_path, map_location='cpu'),
                strict=False
            )
        self.regressor = GeneRegressor()
        self.residual_model = ResidualDiffusionModel()

    # for Stage 1: fine-tune last block + regressor
    def freeze_encoder_last_block_trainable(self):
        self.encoder.freeze_encoder_last_block_trainable()

    # pure MSE forward
    def mse_predict(self, images):
        feat = self.encoder(images)
        return self.regressor(feat)

    # hybrid sampling (diffusion on residual) with uncertainty
    @torch.no_grad()
    def hybrid_sample(
        self,
        images,
        num_samples=7,
        lambda_residual=LAMBDA_RESIDUAL,
        return_uncertainty=True,
        device=None,
    ):
        """
        Hybrid prediction with uncertainty:

        1) x_base = regressor(encoder(image))
        2) Run residual diffusion num_samples times with different initial noise
        3) For each k: x_k = x_base + lambda_residual * r0_k
        4) Return:
            - mean over k: [B, GENE_DIM]
            - per-cell, per-gene uncertainty (var over k): [B, GENE_DIM]
        """
        if device is None:
            device = images.device

        images = images.to(device)
        b = images.size(0)

        feat = self.encoder(images)
        base_pred = self.regressor(feat)  # [B, GENE_DIM]

        x_samples = []

        for k in range(num_samples):
            # start residual from Gaussian noise
            r_t = torch.randn(b, GENE_DIM, device=device)

            for t_step in reversed(range(T)):
                t_batch = torch.full((b,), t_step, device=device, dtype=torch.long)

                r0_pred = self.residual_model(r_t, feat, base_pred, t_batch)

                alpha_t = extract(alphas, t_batch, r_t.shape)
                alpha_cum_t = extract(alphas_cumprod, t_batch, r_t.shape)
                alpha_cum_prev = extract(alphas_cumprod_prev, t_batch, r_t.shape)
                beta_t = extract(betas, t_batch, r_t.shape)
                var_t = extract(posterior_variance, t_batch, r_t.shape)

                coef_x0 = torch.sqrt(alpha_cum_prev) * beta_t / (1.0 - alpha_cum_t)
                coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_cum_prev) / (1.0 - alpha_cum_t)

                posterior_mean = coef_x0 * r0_pred + coef_xt * r_t

                # deterministic sampling: no extra noise term inside the step
                r_t = posterior_mean

            # final residual sample at t=0
            x_k = base_pred + lambda_residual * r_t   # [B, GENE_DIM]
            x_samples.append(x_k)

        x_samples = torch.stack(x_samples, dim=0)      # [num_samples, B, GENE_DIM]
        x_mean = x_samples.mean(dim=0)                 # [B, GENE_DIM]

        if return_uncertainty:
            # variance across the samples => uncertainty per cell & gene
            x_var = x_samples.var(dim=0, unbiased=False)  # [B, GENE_DIM]
            return x_mean, x_var
        else:
            return x_mean

# =========================
# Dataset
# =========================
class CustomImageDataset(Dataset):
    def __init__(self, status='train'):
        adata_seq = sc.read('./data/IRIS_human.h5ad')

        print(adata_seq)
        print('finish loading h5ad')

        sc.pp.normalize_total(adata_seq)
        sc.pp.log1p(adata_seq)

        self.cell_id = adata_seq.obs['cell_id']
        self.len = self.cell_id.shape[0]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.status = status
        self.gene_exp_gt = adata_seq.X

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        if self.status == 'test':
            idx_tmp = idx // 2 * 2
            cell_id_tmp = self.cell_id[idx_tmp]
            gene_tensor = self.gene_exp_gt[idx_tmp]

            image = torchvision.io.read_image(
                f'./data/images_human/{cell_id_tmp}.jpg'
            ).float() / 255.0

            image = self.transform(image)
            image = image.expand(3, 224, 224)

        else:
            idx_tmp = idx // 2 * 2 + 1
            if idx_tmp >= self.len:
                idx_tmp = 1

            cell_id_tmp = self.cell_id[idx_tmp]
            gene_tensor = self.gene_exp_gt[idx_tmp]

            image = torchvision.io.read_image(
                f'./data/images_human/{cell_id_tmp}.jpg'
            ).float() / 255.0

            image = self.transform(image)
            image = image.expand(3, 224, 224)

        return image, gene_tensor

# =========================
# DataLoaders
# =========================
dataset_train = CustomImageDataset(status='train')
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=32)

dataset_test = CustomImageDataset(status='test')
dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False, num_workers=32)

# put diffusion tensors on correct device once
to_device_diffusion_tensors(device)

# =========================
# Stage 1: Train MSE baseline (within unified model)
# =========================
if TRAIN_MSE_BASELINE:
    model = HybridGeneModel(mae_pretrain_path=MAE_PRETRAIN_PATH).to(device)
    model.freeze_encoder_last_block_trainable()

    # only train regressor + unfrozen encoder parameters
    mse_params = list(model.regressor.parameters()) + [
        p for p in model.encoder.parameters() if p.requires_grad
    ]
    optimizer_mse = optim.Adam(mse_params, lr=1e-4)
    criterion = nn.MSELoss()

    mse_epochs = 30  # adjust

    for epoch in range(mse_epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for images, genes in dataloader_train:
            images = images.to(device)
            x0 = genes.to(device).float()

            x_pred = model.mse_predict(images)

            loss = criterion(x_pred, x0)

            optimizer_mse.zero_grad()
            loss.backward()
            optimizer_mse.step()

            train_loss_sum += loss.item()
            train_batches += 1

            print(f"[MSE] Epoch {epoch} | Train step {train_batches} | loss = {loss.item():.6f}")

        avg_train_loss = train_loss_sum / max(train_batches, 1)
        print(f"[MSE] Epoch {epoch} | Avg train loss = {avg_train_loss:.6f}")

        # simple test MSE monitoring
        model.eval()
        test_loss_sum = 0.0
        test_batches = 0
        with torch.no_grad():
            for images, genes in dataloader_test:
                images = images.to(device)
                x0 = genes.to(device).float()
                x_pred = model.mse_predict(images)
                loss = criterion(x_pred, x0)
                test_loss_sum += loss.item()
                test_batches += 1
        avg_test_loss = test_loss_sum / max(test_batches, 1)
        print(f"[MSE] Epoch {epoch} | Avg test loss = {avg_test_loss:.6f}")
        with open(log_path, "a") as f:
            f.write(f"[MSE] Epoch {epoch} | Avg test loss = {avg_test_loss:.6f}\n")

    # save baseline (encoder+regressor) and full model
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "regressor": model.regressor.state_dict(),
        },
        mse_ckpt_path
    )
    torch.save(model.state_dict(), hybrid_ckpt_path)
    print(f"[MSE] Saved baseline to {mse_ckpt_path}")
    print(f"[MSE] Saved unified model (baseline only) to {hybrid_ckpt_path}")

# =========================
# Stage 2: Train residual diffusion (within same model)
# =========================
if TRAIN_RESIDUAL_DIFFUSION:
    model = HybridGeneModel(mae_pretrain_path=None).to(device)

    if os.path.exists(mse_ckpt_path):
        print(f"[Diff] Loading baseline from {mse_ckpt_path}")
        mse_ckpt = torch.load(mse_ckpt_path, map_location=device)
        model.encoder.load_state_dict(mse_ckpt["encoder"])
        model.regressor.load_state_dict(mse_ckpt["regressor"])
    elif os.path.exists(hybrid_ckpt_path):
        print(f"[Diff] Loading unified model from {hybrid_ckpt_path}")
        model.load_state_dict(torch.load(hybrid_ckpt_path, map_location=device))
    else:
        raise RuntimeError("No baseline checkpoint found for Stage 2.")

    # freeze encoder & regressor, only train residual_model
    model.encoder.eval()
    model.regressor.eval()
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.regressor.parameters():
        p.requires_grad = False

    optimizer_diff = optim.Adam(model.residual_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    diff_epochs = 10  # adjust

    for epoch in range(diff_epochs):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for images, genes in dataloader_train:
            images = images.to(device)
            x0 = genes.to(device).float()

            with torch.no_grad():
                feat = model.encoder(images)            # [B, 1024]
                base_pred = model.regressor(feat)       # [B, GENE_DIM]

            r0 = x0 - base_pred                         # residual target

            b = x0.size(0)
            t = sample_timesteps(b, device=device)

            noise = torch.randn_like(r0)
            sqrt_ab = extract(sqrt_alphas_cumprod, t, r0.shape)
            sqrt_1mab = extract(sqrt_one_minus_alphas_cumprod, t, r0.shape)
            r_t = sqrt_ab * r0 + sqrt_1mab * noise

            r0_hat = model.residual_model(r_t, feat, base_pred, t)

            loss = criterion(r0_hat, r0)

            optimizer_diff.zero_grad()
            loss.backward()
            optimizer_diff.step()

            train_loss_sum += loss.item()
            train_batches += 1

            print(f"[Diff] Epoch {epoch} | Train step {train_batches} | loss = {loss.item():.6f}")

        avg_train_loss = train_loss_sum / max(train_batches, 1)
        print(f"[Diff] Epoch {epoch} | Avg train residual loss = {avg_train_loss:.6f}")

        # simple val monitoring on residual MSE
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for images, genes in dataloader_test:
                images = images.to(device)
                x0 = genes.to(device).float()
                feat = model.encoder(images)
                base_pred = model.regressor(feat)
                r0 = x0 - base_pred

                b = x0.size(0)
                t = sample_timesteps(b, device=device)
                noise = torch.randn_like(r0)
                sqrt_ab = extract(sqrt_alphas_cumprod, t, r0.shape)
                sqrt_1mab = extract(sqrt_one_minus_alphas_cumprod, t, r0.shape)
                r_t = sqrt_ab * r0 + sqrt_1mab * noise

                r0_hat = model.residual_model(r_t, feat, base_pred, t)
                loss = criterion(r0_hat, r0)
                val_loss_sum += loss.item()
                val_batches += 1
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        print(f"[Diff] Epoch {epoch} | Avg val residual loss = {avg_val_loss:.6f}")
        with open(log_path, "a") as f:
            f.write(f"[Diff] Epoch {epoch} | Avg val residual loss = {avg_val_loss:.6f}\n")

    torch.save(model.state_dict(), hybrid_ckpt_path)
    print(f"[Diff] Saved unified hybrid model (baseline + diffusion) to {hybrid_ckpt_path}")

# =========================
# Stage 3: Test-time prediction
#         - MSE baseline (same unified model)
#         - Hybrid MSE + Diffusion (5 samples) + uncertainty
# =========================
if RUN_TEST:
    model = HybridGeneModel(mae_pretrain_path=None).to(device)
    print(f"[Test] Loading unified hybrid model from {hybrid_ckpt_path}")
    model.load_state_dict(torch.load(hybrid_ckpt_path, map_location=device))
    model.eval()

    out_dir = './genes/human_0.5/predictions'
    os.makedirs(out_dir, exist_ok=True)
    cell_ids = list(dataset_test.cell_id)

    # -------------------------
    # 3A. Pure MSE baseline predictions
    # -------------------------
    all_preds_mse = []
    all_gt_mse = []

    with torch.no_grad():
        for batch_idx, (images, genes) in enumerate(dataloader_test):
            print(f"[Test-MSE] Inference batch {batch_idx+1}")
            images = images.to(device)
            x0 = genes.to(device).float()

            x_pred = model.mse_predict(images)

            all_preds_mse.append(x_pred.cpu())
            all_gt_mse.append(x0.cpu())

    pred_genes_mse = torch.cat(all_preds_mse, dim=0)  # [N_test, GENE_DIM]
    gt_genes_mse = torch.cat(all_gt_mse, dim=0)

    mse_out_path = os.path.join(out_dir, 'mse_baseline_predictions_test_human.pt')
    torch.save(
        {
            "pred_genes": pred_genes_mse,
            "gt_genes": gt_genes_mse,
            "cell_ids": cell_ids,
            "mse_ckpt": mse_ckpt_path,
            "hybrid_ckpt": hybrid_ckpt_path,
        },
        mse_out_path
    )
    print(f"[Test-MSE] Saved baseline predictions to {mse_out_path}")

    # -------------------------
    # 3B. Hybrid MSE + Diffusion predictions (5 samples) + uncertainty
    # -------------------------
    all_preds_hybrid = []      # mean predictions
    all_uncert_hybrid = []     # per-cell, per-gene uncertainty
    all_gt_hybrid = []

    with torch.no_grad():
        for batch_idx, (images, genes) in enumerate(dataloader_test):
            print(f"[Test-Hybrid] Inference batch {batch_idx+1}")
            x_mean, x_var = model.hybrid_sample(
                images,
                num_samples=7,
                lambda_residual=LAMBDA_RESIDUAL,
                return_uncertainty=True,
                device=device,
            )
            all_preds_hybrid.append(x_mean.cpu())
            all_uncert_hybrid.append(x_var.cpu())
            all_gt_hybrid.append(genes.float().cpu())

    pred_genes_hybrid = torch.cat(all_preds_hybrid, dim=0)      # [N_test, GENE_DIM]
    uncert_genes_hybrid = torch.cat(all_uncert_hybrid, dim=0)   # [N_test, GENE_DIM]
    gt_genes_hybrid = torch.cat(all_gt_hybrid, dim=0)

    hybrid_out_path = os.path.join(out_dir, 'hybrid_diffusion_residual_predictions_test_human.pt')
    torch.save(
        {
            "pred_genes": pred_genes_hybrid,
            "pred_genes_uncert": uncert_genes_hybrid,  # NEW: uncertainty
            "gt_genes": gt_genes_hybrid,
            "cell_ids": cell_ids,
            "mse_ckpt": mse_ckpt_path,
            "hybrid_ckpt": hybrid_ckpt_path,
        },
        hybrid_out_path
    )

    print(f"[Test-Hybrid] Saved hybrid predictions + uncertainty to {hybrid_out_path}")
