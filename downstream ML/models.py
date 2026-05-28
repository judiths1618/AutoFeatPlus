"""
models.py — 7 Model Zoo for CSI Positioning + Denoising

CSI snapshot H ∈ C^{64×100} is treated as:
  - Flat vector [12800] for MLP, Autoencoder
  - 2D image (2, 64, 100) for CNN
  - Sequence (64, 100): 64 antenna "steps", 100 subcarrier "channels"
    → this interpretation feeds PatchTST, TimesNet, LSTM, FreqMLP

Models:
  Positioning (classification):
    1. mlp       — MLP baseline
    2. cnn       — 2D CNN (De Bast et al. style)
    3. patchtst  — Patch Transformer (antenna patches)
    4. timesnet  — FFT-guided 2D temporal variation
    5. lstm      — BiLSTM + temporal attention
    6. freqmlp   — Frequency-domain MLP with spectral filters

  Denoising (regression):
    7. autoencoder — Encoder-decoder for CSI reconstruction
    + All positioning models adapted with regression head
"""
from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Shared: Reshape flat input to sequence (B, 12800) → (B, 64, 100)
# =====================================================================

class ReshapeToSeq(nn.Module):
    """Reshape flat input to sequence for antenna-as-sequence models.

    For 2-channel input (amp_phase, real_imag): (B, 2*n_ant*n_sub) → (B, n_ant, 2*n_sub)
    For 1-channel input (amplitude): (B, n_ant*n_sub) → (B, n_ant, n_sub)
    """
    def __init__(self, n_ant: int = 64, n_sub: int = 100):
        super().__init__()
        self.n_ant = n_ant
        self.n_sub = n_sub  # This is already the full width per step (e.g., 200 for 2-ch, 100 for 1-ch)

    def forward(self, x):
        B = x.shape[0]
        return x.reshape(B, self.n_ant, self.n_sub)  # (B, n_ant, total_sub_width)


# =====================================================================
# 1. MLP — Simple baseline
# =====================================================================

class CSI_MLP(nn.Module):
    def __init__(self, input_dim=12800, output_dim=4, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, output_dim),
        )
    def forward(self, x):
        return self.net(x)


# =====================================================================
# 2. CNN — 2D convolution on (64, 100) CSI matrix
# =====================================================================

class CSI_CNN(nn.Module):
    def __init__(self, output_dim=4, n_ant=64, n_sub=100, in_channels=2, dropout=0.3):
        super().__init__()
        self.n_ant = n_ant
        self.n_sub = n_sub
        self.in_channels = in_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        if self.in_channels == 2:
            half = x.shape[1] // 2
            ch1 = x[:, :half].reshape(B, 1, self.n_ant, self.n_sub)
            ch2 = x[:, half:].reshape(B, 1, self.n_ant, self.n_sub)
            img = torch.cat([ch1, ch2], dim=1)
        else:
            # Single channel (e.g., amplitude only)
            img = x.reshape(B, 1, self.n_ant, self.n_sub)
        return self.head(self.features(img))


# =====================================================================
# 3. PatchTST — Patch Transformer on antenna dimension
# =====================================================================

class CSI_PatchTST(nn.Module):
    """Patches along antenna axis, processes subcarrier features."""
    def __init__(self, output_dim=4, n_ant=64, n_sub=200,
                 d_model=64, n_heads=4, n_layers=2, patch_len=8, stride=4, dropout=0.2):
        super().__init__()
        self.reshape = ReshapeToSeq(n_ant, n_sub)
        self.n_patches = (n_ant - patch_len) // stride + 1
        self.patch_proj = nn.Linear(patch_len * n_sub, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )
        self.patch_len = patch_len
        self.stride = stride
        self.n_sub = n_sub

    def forward(self, x):
        seq = self.reshape(x)
        B, T, C = seq.shape
        patches = seq.unfold(1, self.patch_len, self.stride)
        patches = patches.reshape(B, self.n_patches, -1)
        z = self.patch_proj(patches) + self.pos_emb
        z = self.encoder(z)
        z = z.mean(dim=1)
        return self.head(z)


# =====================================================================
# 4. TimesNet — FFT-guided 2D variation on antenna sequence
# =====================================================================

class CSI_TimesNet(nn.Module):
    def __init__(self, output_dim=4, n_ant=64, n_sub=200, d_model=64, n_layers=2,
                 top_k=3, dropout=0.2):
        super().__init__()
        self.reshape = ReshapeToSeq(n_ant, n_sub)
        self.n_ant = n_ant
        self.embed = nn.Linear(n_sub, d_model)
        self.top_k = top_k
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                "inception": nn.Sequential(
                    nn.Conv2d(d_model, d_model, (1, 3), padding=(0, 1)), nn.ReLU(),
                    nn.Conv2d(d_model, d_model, (1, 5), padding=(0, 2)), nn.ReLU(),
                ),
                "norm": nn.LayerNorm(d_model),
                "drop": nn.Dropout(dropout),
            }))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))

    def forward(self, x):
        seq = self.reshape(x)
        z = self.embed(seq)
        B, T, D = z.shape
        for block in self.blocks:
            residual = z
            freq = torch.fft.rfft(z.mean(dim=-1), dim=1)
            amp = freq.abs().mean(dim=0)
            amp[0] = 0
            _, top_idx = torch.topk(amp, min(self.top_k, len(amp) - 1))
            periods = (T / (top_idx.float() + 1e-6)).clamp(min=2).int()
            agg = torch.zeros_like(z)
            for i in range(len(top_idx)):
                p = min(int(periods[i].item()), T)
                n_seg = T // p
                trim = n_seg * p
                xi = z[:, :trim, :].reshape(B, n_seg, p, D).permute(0, 3, 1, 2)
                yi = block["inception"](xi).permute(0, 2, 3, 1).reshape(B, trim, D)
                agg[:, :trim, :] += yi
            z = block["norm"](block["drop"](agg / max(len(top_idx), 1)) + residual)
        return self.head(z.mean(dim=1))


# =====================================================================
# 5. LSTM — BiLSTM + Temporal Attention on antenna sequence
# =====================================================================

class CSI_LSTM(nn.Module):
    def __init__(self, output_dim=4, n_ant=64, n_sub=200, d_model=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.reshape = ReshapeToSeq(n_ant, n_sub)
        self.proj = nn.Linear(n_sub, d_model)
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, output_dim))

    def forward(self, x):
        seq = self.reshape(x)
        z = self.proj(seq)
        z, _ = self.lstm(z)
        attn_out, _ = self.attn(z, z, z)
        z = self.norm(z + attn_out)
        return self.head(z.mean(dim=1))


# =====================================================================
# 6. FreqMLP — Frequency-domain MLP with learnable spectral filters
# =====================================================================

class CSI_FreqMLP(nn.Module):
    def __init__(self, output_dim=4, n_ant=64, n_sub=200, d_model=64,
                 n_layers=2, dropout=0.2):
        super().__init__()
        self.reshape = ReshapeToSeq(n_ant, n_sub)
        self.embed = nn.Linear(n_sub, d_model)
        freq_len = n_ant // 2 + 1
        self.n_layers = n_layers

        # Store spectral filter parameters as ParameterLists
        self.w_reals = nn.ParameterList()
        self.w_imags = nn.ParameterList()
        self.mlps = nn.ModuleList()
        self.norm1s = nn.ModuleList()
        self.norm2s = nn.ModuleList()

        for _ in range(n_layers):
            self.w_reals.append(nn.Parameter(torch.randn(freq_len, d_model) * 0.02))
            self.w_imags.append(nn.Parameter(torch.randn(freq_len, d_model) * 0.02))
            self.mlps.append(nn.Sequential(
                nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            ))
            self.norm1s.append(nn.LayerNorm(d_model))
            self.norm2s.append(nn.LayerNorm(d_model))

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, output_dim))
        self.n_ant = n_ant

    def forward(self, x):
        seq = self.reshape(x)
        z = self.embed(seq)
        for i in range(self.n_layers):
            residual = z
            xf = torch.fft.rfft(z, dim=1)
            W = torch.complex(self.w_reals[i], self.w_imags[i])
            xf = xf * W.unsqueeze(0)
            z_mixed = torch.fft.irfft(xf, n=self.n_ant, dim=1)
            z = self.norm1s[i](z_mixed + residual)
            z = self.norm2s[i](self.mlps[i](z) + z)
        return self.head(z.mean(dim=1))


# =====================================================================
# 7. Autoencoder — For denoising task
# =====================================================================

class CSI_Autoencoder(nn.Module):
    def __init__(self, input_dim=12800, output_dim=12800, latent_dim=256, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, output_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


# =====================================================================
# Wrapper: adapt sequence models for denoising (regression head)
# =====================================================================

class DenoisingWrapper(nn.Module):
    """Wraps a sequence model: replaces classification head with regression."""
    def __init__(self, backbone: nn.Module, d_model: int, output_dim: int):
        super().__init__()
        self.backbone = backbone
        self.backbone.head = nn.Identity()
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, 512), nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.reg_head(features)


# =====================================================================
# 8. PredLSTM — LSTM for CSI time-series prediction
# =====================================================================

class CSI_PredLSTM(nn.Module):
    """LSTM that treats W time steps as sequence, each step = feat_dim features.

    Input:  (B, W * feat_dim) → reshape to (B, W, feat_dim)
    Output: (B, horizon * feat_dim)
    """
    def __init__(self, input_dim, output_dim, window=5, d_model=128,
                 n_layers=2, dropout=0.2):
        super().__init__()
        self.window = window
        self.feat_dim = input_dim // window
        self.proj = nn.Linear(self.feat_dim, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        seq = x.reshape(B, self.window, self.feat_dim)  # (B, W, feat_dim)
        z = self.proj(seq)                                # (B, W, d_model)
        out, _ = self.lstm(z)                             # (B, W, d_model)
        return self.head(out[:, -1, :])                   # (B, output_dim) — use last step


# =====================================================================
# 9. PredTransformer — Transformer for CSI time-series prediction
# =====================================================================

class CSI_PredTransformer(nn.Module):
    """Transformer encoder treating W time steps as tokens.

    Input:  (B, W * feat_dim) → reshape to (B, W, feat_dim)
    Output: (B, horizon * feat_dim)
    """
    def __init__(self, input_dim, output_dim, window=5, d_model=128,
                 n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.window = window
        self.feat_dim = input_dim // window
        self.proj = nn.Linear(self.feat_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, window, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Causal mask: each step can only attend to itself and past steps
        self.register_buffer("causal_mask",
                             torch.triu(torch.ones(window, window) * float('-inf'), diagonal=1))
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        seq = x.reshape(B, self.window, self.feat_dim)
        z = self.proj(seq) + self.pos_emb
        z = self.encoder(z, mask=self.causal_mask)
        return self.head(z[:, -1, :])  # predict from last token


# =====================================================================
# Factory
# =====================================================================

MODEL_REGISTRY = {
    "mlp": CSI_MLP,
    "cnn": CSI_CNN,
    "patchtst": CSI_PatchTST,
    "timesnet": CSI_TimesNet,
    "lstm": CSI_LSTM,
    "freqmlp": CSI_FreqMLP,
    "autoencoder": CSI_Autoencoder,
    "pred_lstm": CSI_PredLSTM,
    "pred_transformer": CSI_PredTransformer,
}

TASK_MODELS = {
    "positioning": ["mlp", "cnn", "patchtst", "timesnet", "lstm", "freqmlp"],
    "denoising": ["mlp", "autoencoder", "patchtst", "lstm", "freqmlp"],
    "prediction": ["mlp", "pred_lstm", "pred_transformer"],
}


def get_model(name: str, task: str, input_dim: int = 12800,
              num_classes: int = 4, output_dim: Optional[int] = None,
              **kwargs) -> nn.Module:
    """Factory: create model by name and task.

    For positioning: output_dim = num_classes
    For denoising:   output_dim = input_dim (reconstruct full CSI)
    For prediction:  output_dim must be provided (= horizon * feat_dim)

    kwargs may contain:
      - d_model: hidden dim for pred_lstm/pred_transformer
      - n_ant, n_sub: antenna/subcarrier counts for CNN/sequence models
    """
    if output_dim is None:
        output_dim = num_classes if task == "positioning" else input_dim

    # Extract model-specific kwargs, don't pass unknown ones
    d_model = kwargs.get("d_model", 128)
    n_ant = kwargs.get("n_ant", 64)
    n_sub = kwargs.get("n_sub", 100)
    n_channels = kwargs.get("n_channels", 2)  # 2 for amp_phase/real_imag, 1 for amplitude

    # n_sub for sequence models: after channel concat
    seq_n_sub = n_sub * n_channels

    if name == "mlp":
        return CSI_MLP(input_dim, output_dim)
    elif name == "cnn":
        return CSI_CNN(output_dim, n_ant=n_ant, n_sub=n_sub, in_channels=n_channels)
    elif name == "patchtst":
        patch_len = min(kwargs.get("patch_len", 8), n_ant)
        patch_len = max(patch_len, 1)
        stride = min(kwargs.get("stride", 4), patch_len)
        stride = max(stride, 1)
        return CSI_PatchTST(
            output_dim,
            n_ant=n_ant,
            n_sub=seq_n_sub,
            patch_len=patch_len,
            stride=stride,
        )
    elif name == "timesnet":
        return CSI_TimesNet(output_dim, n_ant=n_ant, n_sub=seq_n_sub)
    elif name == "lstm":
        return CSI_LSTM(output_dim, n_ant=n_ant, n_sub=seq_n_sub)
    elif name == "freqmlp":
        return CSI_FreqMLP(output_dim, n_ant=n_ant, n_sub=seq_n_sub)
    elif name == "autoencoder":
        return CSI_Autoencoder(input_dim, output_dim)
    elif name == "pred_lstm":
        return CSI_PredLSTM(input_dim, output_dim, d_model=d_model)
    elif name == "pred_transformer":
        return CSI_PredTransformer(input_dim, output_dim, d_model=d_model)
    else:
        raise ValueError(f"Unknown model: {name}. Options: {list(MODEL_REGISTRY.keys())}")


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()]).detach().cpu()

def set_params_from_flat(model: nn.Module, flat: torch.Tensor):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view_as(p).to(p.device))
        idx += n

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
