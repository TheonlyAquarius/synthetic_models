Of course. Here is the complete code for the key files in your project, reflecting the final two-stage `WeightsAutoencoder` and `LatentDenoiser` architecture we discussed.

Each file is presented in its own code block.

-----

### `src/data.py`

```python
import os
from typing import Iterator, List, Tuple, Dict, OrderedDict as OrderedDictType
from collections import OrderedDict

import torch
from safetensors import safe_open


def load_safetensors_in_file_order(path: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> Tuple[OrderedDictType[str, torch.Tensor], Dict[str, str]]:
    """
    Load a .safetensors file and return an OrderedDict of tensors strictly in file key order,
    plus the (optional) metadata. Only floating tensors are included.
    """
    tensors_od: OrderedDictType[str, torch.Tensor] = OrderedDict()
    meta: Dict[str, str] = {}
    with safe_open(path, framework="pt") as f:
        # keys() preserves on-disk order for SafeTensors
        for k in f.keys():
            t = f.get_tensor(k)
            if torch.is_floating_point(t):
                tensors_od[k] = t.to(device=device, dtype=dtype)
        # metadata, if any
        try:
            meta = dict(f.metadata())  # type: ignore[attr-defined]
        except Exception:
            meta = {}
    return tensors_od, meta


def iter_weights_in_order(path: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Yield (name, tensor) strictly in file order from a .safetensors file.
    Only floating tensors are yielded.
    """
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if torch.is_floating_point(t):
                yield k, t.to(device=device, dtype=dtype)


def list_safetensors_files_from_env() -> List[str]:
    """
    Discover training files based on $TRAIN_DIR and optional $TRAIN_GLOB.
    Returns a sorted list of paths.
    """
    root = os.environ.get("TRAIN_DIR", "")
    pattern = os.environ.get("TRAIN_GLOB", "**/*.safetensors")
    files: List[str] = []
    if root and os.path.isdir(root):
        import glob
        files = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
    # Optional explicit list file
    list_file = os.environ.get("TRAIN_LIST", "")
    if list_file and os.path.isfile(list_file):
        with open(list_file, "r") as f:
            listed = [ln.strip() for ln in f if ln.strip()]
        files = listed
    return [p for p in files if p.endswith(".safetensors")]


class SafeTensorsDataset(torch.utils.data.Dataset):
    """
    Dataset yielding ordered state dicts from .safetensors files listed by $TRAIN_DIR/$TRAIN_LIST.
    Each item is {"path": str, "state": OrderedDict[str, Tensor], "metadata": dict}.
    """
    def __init__(self, files: List[str], device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.files = files
        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.files)

    def __getitem__(self, idx: int):  # type: ignore[override]
        path = self.files[idx]
        state, meta = load_safetensors_in_file_order(path, device=self.device, dtype=self.dtype)
        return {"path": path, "state": state, "metadata": meta}


def collate_single(batch: List[Dict]):
    # Keep samples independent (batch size 1 typical due to variable sequence length)
    return batch[0]
```

-----

### `src/diffusion.py`

```python
from typing import List, Tuple

import torch


class DDPMScheduler:
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2, schedule: str = "linear", device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.T = int(T)
        self.device = device
        self.dtype = dtype
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.T, device=device, dtype=dtype)
        else:
            # cosine schedule (Nichol & Dhariwal)
            s = 0.008
            steps = torch.arange(self.T + 1, device=device, dtype=torch.float64)
            alphas_bar = torch.cos(((steps / self.T + s) / (1 + s)) * torch.pi * 0.5) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            betas = betas.clamp(1e-8, 0.999).to(device=device, dtype=dtype)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise_tokens(self, z0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to token sequence.
        z0: [B,N,D], t: [B]
        returns (zt, eps) each of shape [B,N,D]
        """
        B = z0.shape[0]
        ab = self.alpha_bars[t].view(B, 1, 1).to(z0.device).to(z0.dtype)
        eps = torch.randn_like(z0)
        zt = (ab.sqrt() * z0) + ((1.0 - ab).sqrt() * eps)
        return zt, eps
```

-----

### `src/ae.py`

```python
import math
from typing import Dict, List, Tuple, OrderedDict as OrderedDictType
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== Canonicalization =====

@torch.no_grad()
def canonicalize(t: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """
    Move the longest axis to the end. Flatten remaining axes into C.
    Returns A[C,L] and cache with inverse info.
    """
    s = list(t.shape)
    if len(s) == 0:
        t2 = t.view(1, 1)
        cache = {"perm": [0, 1], "orig_shape": (1, 1), "C": 1, "L": 1}
        return t2, cache
    j = max(range(len(s)), key=lambda i: s[i])
    perm = [i for i in range(len(s)) if i != j] + [j]
    A = t.permute(perm).contiguous()
    N = A.numel()
    L = A.shape[-1]
    C = N // L
    A = A.view(C, L)
    cache = {"perm": perm, "orig_shape": tuple(s), "C": C, "L": L, "last_axis": j}
    return A, cache


def invert_canonical(A: torch.Tensor, cache: Dict) -> torch.Tensor:
    """
    Inverse of canonicalize(): reshape A[C,L] back to original shape using the stored permutation.
    Steps:
    - View A into the permuted shape (dimensions ordered by perm).
    - Apply inverse permutation to return to original axis order.
    """
    orig = list(cache["orig_shape"])  # original dims
    perm = cache["perm"]              # order used in canonicalization
    # Build the shape in perm order
    dims_perm = [orig[i] for i in perm]
    t_perm = A.view(*dims_perm)
    # Invert permutation
    inv = [0] * len(perm)
    for idx, p in enumerate(perm):
        inv[p] = idx
    t = t_perm.permute(inv).contiguous()
    return t.view(*orig)


# ===== Utilities =====

def _row_quantiles(A: torch.Tensor, q: int) -> torch.Tensor:
    """Per-row quantiles across the last dimension (columns). A: [C,L] -> [C,q]."""
    qs = torch.linspace(0.0, 1.0, q, device=A.device, dtype=torch.float32)
    qt = torch.quantile(A, qs, dim=1)  # [q, C]
    return qt.transpose(0, 1).contiguous()  # [C, q]


def _col_quantiles(A: torch.Tensor, q: int) -> torch.Tensor:
    """Per-column quantiles across the first dimension (rows). A: [C,L] -> [L,q]."""
    qs = torch.linspace(0.0, 1.0, q, device=A.device, dtype=torch.float32)
    qt = torch.quantile(A, qs, dim=0)  # [q, L]
    return qt.transpose(0, 1).contiguous()  # [L, q]


def _stats_along(x: torch.Tensor, dim: int) -> torch.Tensor:
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=False) + 1e-8
    mx = x.max(dim=dim).values
    mn = x.min(dim=dim).values
    l2 = (x.pow(2).sum(dim=dim).sqrt()) / math.sqrt(max(1, x.shape[dim]))
    mabs = x.abs().mean(dim=dim)
    return torch.stack([mean, std, l2, mx, mn, mabs], dim=-1)


def _shape_vec(C: int, L: int, orig: Tuple[int, ...], device, dtype) -> torch.Tensor:
    rank = len(orig)
    logs = [math.log1p(d) for d in list(orig)[:4]]
    while len(logs) < 4:
        logs.append(0.0)
    numel = int(torch.tensor(orig).prod().item()) if len(orig) > 0 else 1
    v = torch.tensor([
        float(rank), math.log1p(float(numel)), math.log1p(float(C)), math.log1p(float(L)),
        logs[0], logs[1], logs[2], logs[3]
    ], device=device, dtype=dtype)
    return v


class AttentionPool(nn.Module):
    def __init__(self, d_model: int, n_heads: int, k_tokens: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(k_tokens, d_model) / math.sqrt(d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D]; returns [K, D]
        q = self.q.unsqueeze(0)  # [1, K, D]
        y = self.attn(q, self.ln(x.unsqueeze(0)), self.ln(x.unsqueeze(0)), need_weights=False)[0]
        return y.squeeze(0)


class TensorEncoder(nn.Module):
    """
    Axis-wise tokenization encoder for one canonicalized tensor A[C,L].
    - Build per-row tokens from stats/quantiles along columns.
    - Build per-column tokens from stats/quantiles along rows, then pool to Kc tokens via AttentionPool.
    - Dual small Transformers process row and pooled column tokens.
    - Aggregate to a single tensor embedding of size d_model.
    """
    def __init__(self, d_model: int = 512, q_row: int = 8, q_col: int = 8,
                 k_row: int = 16, k_col: int = 16, heads: int = 8, depth: int = 2):
        super().__init__()
        self.d_model = d_model
        self.k_row = k_row
        self.k_col = k_col
        row_feat_dim = 6 + q_row
        col_feat_dim = 6 + q_col
        self.row_proj = nn.Linear(row_feat_dim, d_model)
        self.col_proj = nn.Linear(col_feat_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
                                               dropout=0.0, batch_first=True, norm_first=True)
        self.row_tr = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.col_tr = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.pool_row = AttentionPool(d_model, heads, k_row)
        self.pool_col = AttentionPool(d_model, heads, k_col)
        self.fuse = nn.Sequential(nn.LayerNorm(2 * d_model + 8), nn.Linear(2 * d_model + 8, d_model), nn.SiLU())

    def forward(self, A: torch.Tensor, orig_shape: Tuple[int, ...]) -> torch.Tensor:
        # A: [C,L]
        C, L = A.shape
        device = A.device
        dtype = A.dtype
        # row tokens (C, d)
        row_stats = _stats_along(A, dim=1)  # [C, 6]
        row_q = _row_quantiles(A, q=8)  # [C, 8]
        row_feat = torch.cat([row_stats, row_q], dim=-1)
        row_tok = self.row_proj(row_feat)  # [C, d]
        row_tok = self.row_tr(row_tok.unsqueeze(0)).squeeze(0)  # [C, d]
        row_tok = self.pool_row(row_tok)  # [k_row, d]

        # column tokens (L, d) -> pool to k_col
        col_stats = _stats_along(A, dim=0)  # [L, 6]
        col_q = _col_quantiles(A, q=8)  # [L, 8]
        col_feat = torch.cat([col_stats, col_q], dim=-1)  # [L, 6+8]
        col_tok = self.col_proj(col_feat)  # [L, d]
        col_tok = self.pool_col(col_tok)   # [k_col, d]
        col_tok = self.col_tr(col_tok.unsqueeze(0)).squeeze(0)  # [k_col, d]

        # Aggregate summaries
        row_sum = row_tok.mean(dim=0)
        col_sum = col_tok.mean(dim=0)
        shape_v = _shape_vec(C, L, orig_shape, device, dtype)
        e = torch.cat([row_sum, col_sum, shape_v.to(row_sum.dtype)], dim=-1)
        return self.fuse(e)


class GlobalAggregator(nn.Module):
    def __init__(self, d_model: int = 512, latent_dim: int = 512, heads: int = 8, depth: int = 4, max_tokens: int = 4096):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
        self.pos = nn.Embedding(max_tokens + 1, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
                                               dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, latent_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [N, d]
        N, D = tokens.shape
        x = tokens.unsqueeze(0)
        cls = self.cls
        seq = torch.cat([cls, x], dim=1)
        pos_ids = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
        seq = seq + self.pos(pos_ids)
        h = self.tr(seq)
        z = self.head(h[:, 0, :])  # [1, latent]
        return z.squeeze(0)


class TensorDecoder(nn.Module):
    """
    Decode one tensor from the global latent z using a low-rank factorization A ≈ U @ V^T.
    U ∈ R^{C×r}, V ∈ R^{L×r}. Shapes (C,L) come from cache.
    """
    def __init__(self, latent_dim: int = 512, rank: int = 32, hidden: int = 1024):
        super().__init__()
        self.rank = rank
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 8, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.u_head = nn.Linear(hidden, rank)
        self.v_head = nn.Linear(hidden, rank)

    def forward(self, z: torch.Tensor, cache: Dict) -> torch.Tensor:
        C = int(cache["C"])
        L = int(cache["L"])
        shape_v = _shape_vec(C, L, cache["orig_shape"], z.device, z.dtype)
        h = self.fc(torch.cat([z, shape_v.to(z.dtype)], dim=-1))
        u_vec = self.u_head(h)  # [r]
        v_vec = self.v_head(h)  # [r]
        # Broadcast to full shapes using learned bases per position
        # Use simple sinusoidal positional features to expand to C and L
        def pos_feats(n: int, d: int) -> torch.Tensor:
            pos = torch.arange(n, device=z.device, dtype=torch.float32).unsqueeze(1)
            freqs = torch.exp(-math.log(10000.0) * torch.arange(0, d, device=z.device).float() / max(d - 1, 1))
            return torch.cat([torch.sin(pos * freqs), torch.cos(pos * freqs)], dim=1)[:, :d]

        # Build U and V via linear maps from positional features
        Phi_C = pos_feats(C, self.rank)  # [C,r]
        Phi_L = pos_feats(L, self.rank)  # [L,r]
        U = Phi_C * u_vec.unsqueeze(0)   # [C,r]
        V = Phi_L * v_vec.unsqueeze(0)   # [L,r]
        A = U @ V.t()  # [C,L]
        return A


class WeightsAutoencoder(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 latent_dim: int = 512,
                 heads: int = 8,
                 depth_tensor: int = 2,
                 depth_global: int = 4,
                 k_row: int = 16,
                 k_col: int = 16,
                 rank: int = 32,
                 max_tokens: int = 8192):
        super().__init__()
        self.tensor_enc = TensorEncoder(d_model=d_model, k_row=k_row, k_col=k_col, heads=heads, depth=depth_tensor)
        self.global_agg = GlobalAggregator(d_model=d_model, latent_dim=latent_dim, heads=heads, depth=depth_global, max_tokens=max_tokens)
        self.tensor_dec = TensorDecoder(latent_dim=latent_dim, rank=rank, hidden=4 * d_model)

    def encode_state(self, state: OrderedDictType[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Dict]]:
        tokens: List[torch.Tensor] = []
        caches: List[Dict] = []
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        for name, t in state.items():
            if not torch.is_floating_point(t):
                continue
            A, cache = canonicalize(t.to(device=device, dtype=dtype))
            e = self.tensor_enc(A, cache["orig_shape"])  # [d]
            tokens.append(e)
            caches.append(cache)
        if not tokens:
            return torch.zeros((0,), device=device, dtype=dtype), []
        tok = torch.stack(tokens, dim=0)  # [N,d]
        z = self.global_agg(tok)  # [latent]
        return z, caches

    def decode_state(self, z: torch.Tensor, caches: List[Dict]) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        for cache in caches:
            A = self.tensor_dec(z, cache)  # [C,L]
            t = invert_canonical(A, cache)
            outs.append(t)
        return outs

    def forward(self, state: OrderedDictType[str, torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        z, caches = self.encode_state(state)
        recons = self.decode_state(z, caches)
        return z, recons
```

-----

### `src/train_ae.py`

```python
#!/usr/bin/env python3
import os
from collections import OrderedDict
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import list_safetensors_files_from_env, SafeTensorsDataset, collate_single
from .ae import WeightsAutoencoder


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and len(v) > 0 else int(default)


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None and len(v) > 0 else float(default)


def env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and len(v) > 0 else default


def main():
    device = env_str("DEVICE", ("cuda" if torch.cuda.is_available() else "cpu"))
    d_model = env_int("EMBED_DIM", 512)
    latent_dim = env_int("LATENT_DIM", 512)
    heads = env_int("HEADS", 8)
    depth_tensor = env_int("AE_TENSOR_DEPTH", 2)
    depth_global = env_int("AE_GLOBAL_DEPTH", 4)
    k_row = env_int("AE_K_ROW", 16)
    k_col = env_int("AE_K_COL", 16)
    rank = env_int("DEC_RANK", 32)
    max_tokens = env_int("MAX_TOKENS", 8192)
    lr = env_float("LEARNING_RATE", 2e-4)
    epochs = env_int("EPOCHS", 1)
    output_dir = env_str("OUTPUT_DIR", "./outputs")

    files = list_safetensors_files_from_env()
    if not files:
        raise SystemExit("No training files found. Set $TRAIN_DIR or $TRAIN_LIST.")
    dl = DataLoader(SafeTensorsDataset(files, device="cpu", dtype=torch.float32), batch_size=1, shuffle=True, collate_fn=collate_single)

    model = WeightsAutoencoder(d_model=d_model, latent_dim=latent_dim, heads=heads,
                               depth_tensor=depth_tensor, depth_global=depth_global,
                               k_row=k_row, k_col=k_col, rank=rank, max_tokens=max_tokens).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # Use new torch.amp API to avoid deprecation warnings
    scaler = torch.amp.GradScaler('cuda', enabled=(device.startswith("cuda")))

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for item in dl:
            state = item["state"]
            # Move tensors lazily inside the model; compute reconstruction loss layer-wise to save memory
            opt.zero_grad(set_to_none=True)
            # Forward
            with torch.amp.autocast('cuda', enabled=device.startswith("cuda")):
                z, caches = model.encode_state(state)
                recons = model.decode_state(z, caches)
                # Compute MSE across layers
                loss = 0.0
                i = 0
                for name, t in state.items():
                    if not torch.is_floating_point(t):
                        continue
                    t_rec = recons[i].to(t.device, dtype=t.dtype)
                    loss = loss + F.mse_loss(t_rec, t)
                    i += 1
                loss = loss / max(1, i)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach().cpu())
            steps += 1
            if steps % 10 == 0:
                print(f"epoch {epoch} step {steps} loss={total/steps:.6f}")

        print(f"[epoch {epoch}] avg_loss={total/max(1,steps):.6f}")
        # checkpoint
        ckpt_path = os.path.join(output_dir, f"ae_{epoch:03d}.pt")
        torch.save({
            "model": model.state_dict(),
            "config": {
                "d_model": d_model, "latent_dim": latent_dim, "heads": heads,
                "AE_TENSOR_DEPTH": depth_tensor, "AE_GLOBAL_DEPTH": depth_global,
                "AE_K_ROW": k_row, "AE_K_COL": k_col, "DEC_RANK": rank,
                "MAX_TOKENS": max_tokens
            }
        }, ckpt_path)
        print(f"saved {ckpt_path}")


if __name__ == "__main__":
    main()
```

-----

### `src/train_latent.py`

```python
#!/usr/bin/env python3
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import list_safetensors_files_from_env, SafeTensorsDataset, collate_single
from .ae import WeightsAutoencoder
from .diffusion import DDPMScheduler


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and len(v) > 0 else int(default)


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None and len(v) > 0 else float(default)


def env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and len(v) > 0 else default


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim: int = 512, t_dim: int = 128, hidden: int = 1024):
        super().__init__()
        self.time = nn.Sequential(
            nn.Linear(t_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.to_t = nn.Linear(1, t_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim)
        )

    def forward(self, zt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # zt: [B, latent]
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        temb = self.time(self.to_t(t.float()))
        x = torch.cat([zt, temb], dim=-1)
        return self.net(x)


def main():
    device = env_str("DEVICE", ("cuda" if torch.cuda.is_available() else "cpu"))
    latent_dim = env_int("LATENT_DIM", 512)
    t_dim = env_int("T_DIM", 128)
    timesteps = env_int("TIMESTEPS", 1000)
    beta_start = env_float("BETA_START", 1e-4)
    beta_end = env_float("BETA_END", 2e-2)
    beta_schedule = env_str("BETA_SCHEDULE", "linear")
    lr = env_float("LEARNING_RATE", 1e-4)
    epochs = env_int("EPOCHS", 1)
    output_dir = env_str("OUTPUT_DIR", "./outputs")
    ae_ckpt = env_str("AE_CKPT", "")

    if not ae_ckpt or not os.path.isfile(ae_ckpt):
        raise SystemExit("Set $AE_CKPT to a trained autoencoder checkpoint (from src/train_ae.py)")

    files = list_safetensors_files_from_env()
    if not files:
        raise SystemExit("No training files found. Set $TRAIN_DIR or $TRAIN_LIST.")
    dl = DataLoader(SafeTensorsDataset(files, device="cpu", dtype=torch.float32), batch_size=1, shuffle=True, collate_fn=collate_single)

    # Load AE and freeze
    ckpt = torch.load(ae_ckpt, map_location=device)
    cfg = ckpt.get("config", {})
    ae = WeightsAutoencoder(d_model=cfg.get("d_model", 512), latent_dim=cfg.get("latent_dim", latent_dim), heads=cfg.get("heads", 8),
                               depth_tensor=cfg.get("AE_TENSOR_DEPTH", 2), depth_global=cfg.get("AE_GLOBAL_DEPTH", 4),
                               k_row=cfg.get("AE_K_ROW", 16), k_col=cfg.get("AE_K_COL", 16), rank=cfg.get("DEC_RANK", 32),
                               max_tokens=cfg.get("MAX_TOKENS", 8192)).to(device)
    ae.load_state_dict(ckpt["model"], strict=True)
    for p in ae.parameters():
        p.requires_grad = False
    ae.eval()

    denoiser = LatentDenoiser(latent_dim=latent_dim, t_dim=t_dim).to(device)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=lr)
    sched = DDPMScheduler(T=timesteps, beta_start=beta_start, beta_end=beta_end, schedule=beta_schedule, device=device, dtype=torch.float32)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        denoiser.train()
        total = 0.0
        steps = 0
        for item in dl:
            state = item["state"]
            with torch.no_grad():
                z0, _ = ae.encode_state(state)
            z0 = z0.unsqueeze(0)  # [1, latent]
            t = torch.randint(0, timesteps, (1,), device=device, dtype=torch.long)
            zt, eps = sched.add_noise_tokens(z0, t)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                eps_hat = denoiser(zt, t)
                loss = F.mse_loss(eps_hat, eps)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach().cpu())
            steps += 1
            if steps % 10 == 0:
                print(f"epoch {epoch} step {steps} loss={total/steps:.6f}")
        print(f"[epoch {epoch}] avg_loss={total/max(1,steps):.6f}")

    torch.save({"denoiser": denoiser.state_dict(), "latent_dim": latent_dim, "t_dim": t_dim, "timesteps": timesteps},
               os.path.join(output_dir, "latent_denoiser.pt"))
    print(f"saved {os.path.join(output_dir, 'latent_denoiser.pt')}")


if __name__ == "__main__":
    main()
```

-----

### `src/sample_latent.py`

```python
#!/usr/bin/env python3
import os
from collections import OrderedDict
from typing import List, Tuple

import torch
from safetensors.torch import save_file

from .data import load_safetensors_in_file_order
from .ae import WeightsAutoencoder, invert_canonical, canonicalize
from .train_latent import LatentDenoiser
from .diffusion import DDPMScheduler


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None and len(v) > 0 else int(default)


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v is not None and len(v) > 0 else float(default)


def env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and len(v) > 0 else default


def main():
    device = env_str("DEVICE", ("cuda" if torch.cuda.is_available() else "cpu"))
    ae_ckpt = env_str("AE_CKPT", "")
    den_ckpt = env_str("DENOISER_CKPT", "")
    target_path = env_str("TARGET_MODEL", "")
    out_path = env_str("OUTPUT_FILE", "synthesized.safetensors")
    timesteps = env_int("TIMESTEPS", 1000)
    beta_start = env_float("BETA_START", 1e-4)
    beta_end = env_float("BETA_END", 2e-2)
    beta_schedule = env_str("BETA_SCHEDULE", "linear")

    if not (ae_ckpt and os.path.isfile(ae_ckpt)):
        raise SystemExit("Set $AE_CKPT to a trained AE checkpoint")
    if not (den_ckpt and os.path.isfile(den_ckpt)):
        raise SystemExit("Set $DENOISER_CKPT to a trained latent denoiser checkpoint")
    if not (target_path and os.path.isfile(target_path)):
        raise SystemExit("Set $TARGET_MODEL to a .safetensors file providing shapes/order")

    # Load AE
    ckpt = torch.load(ae_ckpt, map_location=device)
    cfg = ckpt.get("config", {})
    ae = WeightsAutoencoder(d_model=cfg.get("d_model", 512), latent_dim=cfg.get("latent_dim", 512), heads=cfg.get("heads", 8),
                               depth_tensor=cfg.get("AE_TENSOR_DEPTH", 2), depth_global=cfg.get("AE_GLOBAL_DEPTH", 4),
                               k_row=cfg.get("AE_K_ROW", 16), k_col=cfg.get("AE_K_COL", 16), rank=cfg.get("DEC_RANK", 32),
                               max_tokens=cfg.get("MAX_TOKENS", 8192)).to(device)
    ae.load_state_dict(ckpt["model"], strict=True)
    for p in ae.parameters():
        p.requires_grad = False
    ae.eval()

    # Load denoiser
    dc = torch.load(den_ckpt, map_location=device)
    latent_dim = dc.get("latent_dim", 512)
    den = LatentDenoiser(latent_dim=latent_dim, t_dim=dc.get("t_dim", 128)).to(device)
    den.load_state_dict(dc["denoiser"])
    den.eval()

    # Shapes/order from target
    state, meta = load_safetensors_in_file_order(target_path, device="cpu", dtype=torch.float32)
    names: List[str] = list(state.keys())
    # Build caches by canonicalizing shapes from target file
    caches = []
    for n in names:
        t = state[n]
        A, cache = canonicalize(t)
        caches.append(cache)

    # Sampler
    sched = DDPMScheduler(T=timesteps, beta_start=beta_start, beta_end=beta_end, schedule=beta_schedule, device=device, dtype=torch.float32)
    with torch.no_grad():
        z = torch.randn(1, latent_dim, device=device)
        for t in reversed(range(timesteps)):
            ts = torch.full((1,), t, device=device, dtype=torch.long)
            eps_hat = den(z, ts)
            beta_t = sched.betas[t]
            alpha_bar_t = sched.alpha_bars[t]
            z0_hat = (z - (1.0 - alpha_bar_t).sqrt() * eps_hat) / (alpha_bar_t.sqrt())
            mean = (sched.alpha_bars[t - 1].sqrt() if t > 0 else torch.tensor(1.0, device=device, dtype=z.dtype)) * z0_hat + \
                   ((1.0 - (sched.alpha_bars[t - 1] if t > 0 else torch.tensor(1.0, device=device, dtype=z.dtype))).sqrt()) * eps_hat
            if t > 0:
                noise = torch.randn_like(z)
                z = mean + (beta_t.sqrt()) * noise
            else:
                z = mean
        z0 = z.squeeze(0)

        # Decode with AE decoder and write out
        tensors = ae.decode_state(z0, caches)
        out_od = OrderedDict()
        for n, x in zip(names, tensors):
            out_od[n] = x.detach().to("cpu")
        save_file(out_od, out_path, metadata=meta)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
```

-----

### `src/export_hf.py`

```python
#!/usr/bin/env python3
import os
import sys

import torch
from safetensors.torch import save_file


def env_str(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v is not None and len(v) > 0 else default


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def main():
    # Allow an optional positional arg as HF_LOCAL_DIR or HF_MODEL_ID for convenience
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    local_dir = env_str("HF_LOCAL_DIR")
    hf_id = env_str("HF_MODEL_ID")
    if arg and not (local_dir or hf_id):
        if os.path.isdir(arg):
            local_dir = arg
        else:
            hf_id = arg

    out_file = env_str("OUT_FILE")
    rand_out = env_str("RANDOM_OUT_FILE")
    local_only = env_bool("LOCAL_FILES_ONLY", False)

    if not (local_dir or hf_id):
        raise SystemExit("Set $HF_LOCAL_DIR or $HF_MODEL_ID (or pass one as a positional arg)")
    if not out_file and not rand_out:
        raise SystemExit("Set $OUT_FILE and/or $RANDOM_OUT_FILE")

    try:
        from transformers import AutoModel, AutoConfig
    except Exception as e:
        raise SystemExit("transformers not installed. pip install transformers safetensors torch")

    src = local_dir if local_dir else hf_id
    kwargs = {"local_files_only": local_only}

    if out_file:
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        print(f"loading pretrained from {src} (local_only={local_only})")
        m = AutoModel.from_pretrained(src, **kwargs)
        sd = {k: v.detach().to("cpu") for k, v in m.state_dict().items() if torch.is_floating_point(v)}
        meta = {"source": src, "kind": "pretrained"}
        save_file(sd, out_file, metadata=meta)
        print(f"wrote {out_file}")

    if rand_out:
        os.makedirs(os.path.dirname(rand_out) or ".", exist_ok=True)
        print(f"loading config from {src} (local_only={local_only})")
        cfg = AutoConfig.from_pretrained(src, **kwargs)
        mr = AutoModel.from_config(cfg)  # randomly initialized architecture
        sd_r = {k: v.detach().to("cpu") for k, v in mr.state_dict().items() if torch.is_floating_point(v)}
        meta_r = {"source": src, "kind": "random_init"}
        save_file(sd_r, rand_out, metadata=meta_r)
        print(f"wrote {rand_out}")


if __name__ == "__main__":
    main()
```

-----

### `src/env_config.py`

```python
#!/usr/bin/env python3
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple


HISTORY_FILE = ".env_config_history.json"


def _boolstr(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _detect_paths() -> Dict[str, str]:
    cand_train = []
    for p in [
        "project/Model-Gen/models",
        "NEW/traj_example",
        "POC",
        ".",
    ]:
        if os.path.isdir(p):
            cand_train.append(p)
    return {
        "TRAIN_DIR": next((p for p in cand_train if any(fn.endswith(".safetensors") for fn in _glob_safetensors(p))), cand_train[0] if cand_train else os.getcwd()),
        "OUTPUT_DIR": "outputs",
    }


def _glob_safetensors(root: str) -> List[str]:
    import glob
    return sorted(glob.glob(os.path.join(root, "**/*.safetensors"), recursive=True))


def _detect_runtime() -> Dict[str, str]:
    device = "cuda" if _torch_has_cuda() else "cpu"
    denoiser = "transformer"
    # Detect without importing native extensions to avoid crashes
    try:
        import importlib.util
        if importlib.util.find_spec("mamba_ssm") is not None:
            denoiser = "mamba"
    except Exception:
        pass
    return {
        "DEVICE": device,
        "DENOISER": denoiser,
    }


def _torch_has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _env_default(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and len(v) > 0 else default


def _merge_prefill(base: Dict[str, str], overrides: Dict[str, str]) -> Dict[str, str]:
    out = dict(base)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def _default_train() -> Dict[str, str]:
    detected = _merge_prefill(_detect_paths(), _detect_runtime())
    defaults = {
        "TRAIN_DIR": detected.get("TRAIN_DIR", os.getcwd()),
        "TRAIN_GLOB": "**/*.safetensors",
        "TRAIN_LIST": "",
        "OUTPUT_DIR": detected.get("OUTPUT_DIR", "outputs"),
        "DEVICE": detected.get("DEVICE", "cpu"),
        "DENOISER": detected.get("DENOISER", "transformer"),
        "EMBED_DIM": "512",
        "DEPTH": "8",
        "HEADS": "8",
        "T_DIM": "128",
        "MAX_TOKENS": "8192",
        "TIMESTEPS": "1000",
        "BETA_START": "1e-4",
        "BETA_END": "2e-2",
        "BETA_SCHEDULE": "linear",
        "LEARNING_RATE": "1e-4",
        "BATCH_SIZE": "1",
        "EPOCHS": "1",
    }
    # apply environment overrides
    for k in list(defaults.keys()):
        defaults[k] = _env_default(k, defaults[k])
    return defaults


def _default_export() -> Dict[str, str]:
    detected = _detect_paths()
    # Suggest a local dir containing tokenizer/config as HF_LOCAL_DIR if present
    hf_local = ""
    for sub in ["distilbert-base-uncased", "distilbert-base-uncased-finetuned", "Qwen3-0.6B"]:
        cand = os.path.join("project/Model-Gen/models", sub)
        if os.path.isdir(cand):
            hf_local = cand
            break
    defaults = {
        "HF_LOCAL_DIR": hf_local,
        "HF_MODEL_ID": "",  # optional alternative to local dir
        "OUT_FILE": "pretrained.safetensors",
        "RANDOM_OUT_FILE": "random_init.safetensors",
        "LOCAL_FILES_ONLY": "1" if hf_local else "0",
    }
    for k in list(defaults.keys()):
        defaults[k] = _env_default(k, defaults[k])
    return defaults


MODES = ("train_ae", "train_latent", "sample_latent", "export")


def _defaults_for_mode(mode: str) -> Dict[str, str]:
    if mode == "train_ae":
        d = _default_train()
        # add AE-specific vars
        d.update({
            "LATENT_DIM": "512",
            "AE_TENSOR_DEPTH": "2",
            "AE_GLOBAL_DEPTH": "4",
            "AE_K_ROW": "16",
            "AE_K_COL": "16",
            "DEC_RANK": "32",
        })
        return d
    if mode == "train_latent":
        d = {
            "AE_CKPT": _env_default("AE_CKPT", ""),
            "TRAIN_DIR": _env_default("TRAIN_DIR", _detect_paths().get("TRAIN_DIR", os.getcwd())),
            "OUTPUT_DIR": _env_default("OUTPUT_DIR", "outputs"),
            "DEVICE": _env_default("DEVICE", _detect_runtime().get("DEVICE", "cpu")),
            "LATENT_DIM": _env_default("LATENT_DIM", "512"),
            "T_DIM": _env_default("T_DIM", "128"),
            "TIMESTEPS": _env_default("TIMESTEPS", "1000"),
            "BETA_START": _env_default("BETA_START", "1e-4"),
            "BETA_END": _env_default("BETA_END", "2e-2"),
            "BETA_SCHEDULE": _env_default("BETA_SCHEDULE", "linear"),
            "LEARNING_RATE": _env_default("LEARNING_RATE", "1e-4"),
            "EPOCHS": _env_default("EPOCHS", "1"),
        }
        return d
    if mode == "sample_latent":
        d = {
            "AE_CKPT": _env_default("AE_CKPT", ""),
            "DENOISER_CKPT": _env_default("DENOISER_CKPT", os.path.join(_detect_paths().get("OUTPUT_DIR", "outputs"), "latent_denoiser.pt")),
            "TARGET_MODEL": _env_default("TARGET_MODEL", ""),
            "OUTPUT_FILE": _env_default("OUTPUT_FILE", "synthesized.safetensors"),
            "DEVICE": _env_default("DEVICE", _detect_runtime().get("DEVICE", "cpu")),
            "TIMESTEPS": _env_default("TIMESTEPS", "1000"),
            "BETA_START": _env_default("BETA_START", "1e-4"),
            "BETA_END": _env_default("BETA_END", "2e-2"),
            "BETA_SCHEDULE": _env_default("BETA_SCHEDULE", "linear"),
        }
        return d
    if mode == "export":
        return _default_export()
    raise ValueError(f"unknown mode: {mode}")


def _print_config(mode: str, cfg: Dict[str, str]) -> None:
    print(f"\n[{mode} config]\n")
    for k in sorted(cfg.keys()):
        print(f"{k}={cfg[k]}")
    print("")


def _select_var(cfg: Dict[str, str]) -> Optional[str]:
    keys = sorted(cfg.keys())
    for i, k in enumerate(keys, 1):
        print(f"{i:2d}. {k} = {cfg[k]}")
    s = input("Select number to edit (Enter to cancel): ").strip()
    if not s:
        return None
    try:
        i = int(s)
        if 1 <= i <= len(keys):
            return keys[i - 1]
    except Exception:
        pass
    print("Invalid selection")
    return None


def _edit_var(cfg: Dict[str, str]) -> None:
    key = _select_var(cfg)
    if not key:
        return
    cur = cfg[key]
    val = input(f"Enter new value for {key} (current '{cur}') [empty to unset]: ").strip()
    cfg[key] = val


def _print_exports(mode: str, cfg: Dict[str, str]) -> None:
    print("# To use: source this file or paste into your shell")
    for k, v in cfg.items():
        if v is None or v == "":
            continue
        print(f"export {k}={v}")
    if mode == "train_ae":
        print("python -m src.train_ae")
    elif mode == "train_latent":
        print("python -m src.train_latent")
    elif mode == "sample_latent":
        print("python -m src.sample_latent")
    elif mode == "export":
        print("python -m src.export_hf")


def _save_exports(mode: str, cfg: Dict[str, str]) -> None:
    path = input("Save export lines to file (e.g., run_env.sh): ").strip() or (f"run_{mode}_env.sh")
    with open(path, "w") as f:
        f.write("# generated by src/env_config.py\n")
        for k, v in cfg.items():
            if v is None or v == "":
                continue
            f.write(f"export {k}={v}\n")
        if mode == "train_ae":
            f.write("python -m src.train_ae\n")
        elif mode == "train_latent":
            f.write("python -m src.train_latent\n")
        elif mode == "sample_latent":
            f.write("python -m src.sample_latent\n")
        elif mode == "export":
            f.write("python -m src.export_hf\n")
    print(f"wrote {path}")


def _load_history() -> List[Dict[str, str]]:
    if not os.path.isfile(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        return []
    return []


def _save_history(mode: str, cfg: Dict[str, str]) -> None:
    hist = _load_history()
    entry = {
        "ts": int(time.time()),
        "mode": mode,
        "config": cfg,
    }
    hist.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(hist, f, indent=2)


def _menu(mode: str, cfg: Dict[str, str]) -> Tuple[str, Dict[str, str], bool]:
    while True:
        print("\n=== env_config ===")
        print(f"mode: {mode}")
        print("1) View config")
        print("2) Edit a variable")
        print("3) Switch mode (train_ae/train_latent/sample_latent/export)")
        print("4) Print export lines")
        print("5) Save export lines to file")
        print("6) Reset to defaults")
        print("7) History (list/load)")
        print("0) Quit")
        choice = input("> ").strip()
        if choice == "1":
            _print_config(mode, cfg)
        elif choice == "2":
            _edit_var(cfg)
        elif choice == "3":
            new_mode = input("Enter mode [train_ae/train_latent/sample_latent/export]: ").strip().lower() or mode
            if new_mode in MODES:
                mode = new_mode
                cfg = _defaults_for_mode(mode)
            else:
                print("unknown mode")
        elif choice == "4":
            _print_exports(mode, cfg)
        elif choice == "5":
            _save_exports(mode, cfg)
        elif choice == "6":
            cfg = _defaults_for_mode(mode)
            print("reset.")
        elif choice == "7":
            hist = _load_history()
            if not hist:
                print("no history yet")
                continue
            for i, e in enumerate(hist[-10:], 1):
                print(f"{i}. {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e['ts']))} [{e['mode']}]")
            s = input("select to load (empty to cancel): ").strip()
            if s:
                try:
                    i = int(s)
                    if 1 <= i <= min(10, len(hist)):
                        entry = hist[-10:][i - 1]
                        mode = entry["mode"]
                        cfg = entry["config"]
                except Exception:
                    print("invalid selection")
        elif choice == "0":
            # Save to history on exit
            _save_history(mode, cfg)
            return mode, cfg, True
        else:
            print("unknown selection")


def main():
    mode = (sys.argv[1].strip().lower() if len(sys.argv) > 1 else os.environ.get("RUN_MODE", "train_ae").strip().lower())
    if mode not in MODES:
        mode = "train_ae"
    cfg = _defaults_for_mode(mode)
    # prefill from environment for current mode
    for k in cfg.keys():
        cfg[k] = _env_default(k, cfg[k])
    _, _, _ = _menu(mode, cfg)


if __name__ == "__main__":
    main()
```
