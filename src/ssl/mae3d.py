"""3D Masked Autoencoder (MAE) for self-supervised pretraining of
:class:`~src.models.mri_cnn_transformer_encoder.MRICNNTransformerEncoder`.

Design
------
* The **encoder** is ``MRICNNTransformerEncoder`` configured with
  ``return_tokens=True`` so we get per-patch tokens.
* Voxel patches are masked **before** the encoder, preventing target anatomy
  from leaking through the CNN or global self-attention.
* A lightweight **decoder** reconstructs the *masked* patches from the
  visible tokens + learnable mask tokens.
* After pretraining only the **encoder** weights are saved; the decoder is
  discarded.

References
----------
He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from src.models.mri_cnn_transformer_encoder import MRICNNTransformerEncoder


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class MAE3DDecoder(nn.Module):
    """Lightweight Transformer decoder that reconstructs masked patch tokens.

    Parameters
    ----------
    encoder_dim : int
        Dimension of the encoder's patch tokens (``d_model``).
    decoder_dim : int
        Internal dimension of the decoder Transformer.
    decoder_depth : int
        Number of Transformer layers in the decoder.
    decoder_heads : int
        Number of attention heads.
    decoder_ff_dim : int
        Feed-forward dimension.
    max_tokens : int
        Maximum number of spatial tokens (used to size positional embeddings).
        Computed as the product of the encoder's spatial grid after CNN
        downsampling.  Will be interpolated at runtime if the actual grid
        differs.
    patch_voxels : int
        Number of voxels per patch token (for the reconstruction target).
        This equals the product of voxel dimensions that collapse into one
        token through the CNN backbone.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        decoder_dim: int = 256,
        decoder_depth: int = 2,
        decoder_heads: int = 4,
        decoder_ff_dim: int = 1024,
        max_tokens: int = 891,   # 9*11*9 for [144,176,144] at stride 16
        patch_voxels: int = 4096,  # 16^3
    ) -> None:
        super().__init__()
        self.decoder_dim = decoder_dim
        self.max_tokens = max_tokens

        # Map encoder tokens → decoder dim
        self.proj_in = nn.Linear(encoder_dim, decoder_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embedding for decoder
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, decoder_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=decoder_depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(decoder_dim)

        # Reconstruct voxels from each token
        self.proj_out = nn.Linear(decoder_dim, patch_voxels)

    def _interpolate_pos(self, n_tokens: int) -> torch.Tensor:
        """Return positional embeddings of length *n_tokens*."""
        if n_tokens == self.max_tokens:
            return self.pos_embed
        return F.interpolate(
            self.pos_embed.transpose(1, 2),  # [1, D, T]
            size=n_tokens,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)  # [1, T, D]

    def forward(
        self,
        visible_tokens: torch.Tensor,
        visible_indices: torch.Tensor,
        masked_indices: torch.Tensor,
        n_total: int,
    ) -> torch.Tensor:
        """Reconstruct masked patches.

        Parameters
        ----------
        visible_tokens : [B, N_vis, encoder_dim]
        visible_indices : [B, N_vis]  index of each visible token
        masked_indices : [B, N_mask]  index of each masked token
        n_total : int  total number of tokens

        Returns
        -------
        pred : [B, N_mask, patch_voxels]  predicted voxel values for masked
        """
        B = visible_tokens.shape[0]

        # Project visible tokens
        vis = self.proj_in(visible_tokens)  # [B, N_vis, decoder_dim]

        # Expand mask tokens
        n_mask = masked_indices.shape[1]
        mask_tokens = self.mask_token.to(dtype=vis.dtype).expand(B, n_mask, -1)

        # Concatenate and restore original order
        all_tokens = torch.zeros(B, n_total, self.decoder_dim, device=vis.device, dtype=vis.dtype)
        all_tokens.scatter_(1, visible_indices.unsqueeze(-1).expand(-1, -1, self.decoder_dim), vis)
        all_tokens.scatter_(1, masked_indices.unsqueeze(-1).expand(-1, -1, self.decoder_dim), mask_tokens)

        # Add positional embeddings
        pos = self._interpolate_pos(n_total).to(all_tokens.dtype)
        all_tokens = all_tokens + pos

        # Decode
        decoded = self.transformer(all_tokens)
        decoded = self.norm(decoded)

        # Extract only masked positions
        masked_decoded = decoded.gather(
            1, masked_indices.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        )  # [B, N_mask, decoder_dim]

        return self.proj_out(masked_decoded)  # [B, N_mask, patch_voxels]


# ---------------------------------------------------------------------------
# Full MAE model
# ---------------------------------------------------------------------------

class MaskedAutoEncoder3D(nn.Module):
    """3D Masked Autoencoder wrapping ``MRICNNTransformerEncoder``.

    A random fraction (``mask_ratio``) of voxel patches is zero-masked before
    encoding. The decoder reconstructs those patches from encoder tokens whose
    source voxels remained visible.
    """

    def __init__(
        self,
        encoder: MRICNNTransformerEncoder,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        ssl_cfg = config["ssl"]
        enc_cfg = config["encoder"]

        self.encoder = encoder
        self.mask_ratio = float(ssl_cfg.get("mask_ratio", 0.60))
        self.norm_target = bool(ssl_cfg.get("norm_target", True))
        self.reconstruction_loss = str(ssl_cfg.get("reconstruction_loss", "mse")).lower()
        self.mask_block_size = tuple(int(v) for v in ssl_cfg.get("mask_block_size", [1, 1, 1]))
        if len(self.mask_block_size) != 3 or any(v < 1 for v in self.mask_block_size):
            raise ValueError("ssl.mask_block_size must contain three positive integers")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("ssl.mask_ratio must be strictly between 0 and 1")
        if self.reconstruction_loss not in {"mse", "l1", "smooth_l1"}:
            raise ValueError("ssl.reconstruction_loss must be one of: mse, l1, smooth_l1")

        # Determine spatial grid and patch voxels from the encoder architecture
        # The CNN backbone downsamples by stride^n_blocks
        stride = int(enc_cfg.get("downsample_stride", 2))
        n_blocks = len(enc_cfg.get("block_channels", [32, 64, 128, 256]))
        total_stride = stride ** n_blocks  # 2^4 = 16

        # For the default target_shape [144, 176, 144]:
        # grid = [144/16, 176/16, 144/16] = [9, 11, 9]
        target_shape = config.get("preprocessing_3mt_style", {}).get("target_shape", [144, 176, 144])
        # If random crop is used, use crop size for grid computation
        crop_size = ssl_cfg.get("augmentation", {}).get("random_crop_size")
        if crop_size:
            effective_shape = [int(s) for s in crop_size]
        else:
            effective_shape = [int(s) for s in target_shape]

        self.grid = tuple(max(s // total_stride, 1) for s in effective_shape)
        self.n_tokens = int(math.prod(self.grid))
        self.patch_voxels = total_stride ** 3  # 16^3 = 4096
        self.total_stride = total_stride
        self.effective_shape = tuple(effective_shape)

        d_model = int(enc_cfg.get("d_model", 512))
        self.decoder = MAE3DDecoder(
            encoder_dim=d_model,
            decoder_dim=int(ssl_cfg.get("decoder_dim", 256)),
            decoder_depth=int(ssl_cfg.get("decoder_depth", 2)),
            decoder_heads=int(ssl_cfg.get("decoder_heads", 4)),
            decoder_ff_dim=int(ssl_cfg.get("decoder_ff_dim", 1024)),
            max_tokens=self.n_tokens,
            patch_voxels=self.patch_voxels,
        )

    @staticmethod
    def _patchify(volume: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Convert [B, 1, D, H, W] → [B, N_tokens, patch_voxels].

        Splits the volume into non-overlapping cubes of ``patch_size^3``.
        """
        B, C, D, H, W = volume.shape
        p = patch_size
        assert D % p == 0 and H % p == 0 and W % p == 0, (
            f"Volume spatial dims {(D,H,W)} must be divisible by patch_size={p}"
        )
        gd, gh, gw = D // p, H // p, W // p
        # [B, 1, gd, p, gh, p, gw, p]
        x = volume.reshape(B, C, gd, p, gh, p, gw, p)
        # [B, gd, gh, gw, C*p*p*p]
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        return x.reshape(B, gd * gh * gw, C * p * p * p)

    def _random_masking(
        self,
        grid: tuple[int, int, int],
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate spatially coherent visible/masked index splits.

        Returns
        -------
        visible_idx : [B, N_vis]
        masked_idx  : [B, N_mask]
        restore_idx : [B, N_tokens]  (for unshuffling)
        """
        n_tokens = int(math.prod(grid))
        n_mask = max(1, min(n_tokens - 1, int(round(n_tokens * self.mask_ratio))))
        n_vis = n_tokens - n_mask

        # Give tokens in the same configured 3D block a shared random score.
        # A tiny per-token jitter makes the exact requested mask ratio possible
        # when the final selected block crosses the ratio boundary.
        gd, gh, gw = grid
        bd, bh, bw = self.mask_block_size
        coarse = torch.rand(
            batch_size,
            math.ceil(gd / bd),
            math.ceil(gh / bh),
            math.ceil(gw / bw),
            device=device,
        )
        scores = coarse.repeat_interleave(bd, 1).repeat_interleave(bh, 2).repeat_interleave(bw, 3)
        scores = scores[:, :gd, :gh, :gw].flatten(1)
        scores = scores + torch.rand_like(scores) * 1e-6
        ids_shuffle = scores.argsort(dim=1)
        masked_idx = ids_shuffle[:, :n_mask]
        visible_idx = ids_shuffle[:, n_mask : n_mask + n_vis]
        mask = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=device)
        mask.scatter_(1, masked_idx, True)
        return visible_idx, masked_idx, mask

    def _mask_voxel_patches(
        self,
        volume: torch.Tensor,
        token_mask: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> torch.Tensor:
        """Replace masked non-overlapping voxel patches with normalized zero."""
        B, _, D, H, W = volume.shape
        gd, gh, gw = grid
        expected = (gd * self.total_stride, gh * self.total_stride, gw * self.total_stride)
        if (D, H, W) != expected:
            raise ValueError(
                f"Input shape {(D, H, W)} must equal token grid {grid} x stride "
                f"{self.total_stride} = {expected}"
            )
        voxel_mask = token_mask.reshape(B, gd, gh, gw)
        voxel_mask = voxel_mask.repeat_interleave(self.total_stride, 1)
        voxel_mask = voxel_mask.repeat_interleave(self.total_stride, 2)
        voxel_mask = voxel_mask.repeat_interleave(self.total_stride, 3)
        return volume.masked_fill(voxel_mask.unsqueeze(1), 0.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : [B, 1, D, H, W]

        Returns
        -------
        dict with keys: ``loss``, ``pred``, ``target``, ``mask_ratio``
        """
        B = x.shape[0]
        device = x.device

        grid = tuple(int(v // self.total_stride) for v in x.shape[-3:])
        if any(v * self.total_stride != int(s) for v, s in zip(grid, x.shape[-3:])):
            raise ValueError(
                f"Input spatial shape {tuple(x.shape[-3:])} must be divisible by {self.total_stride}"
            )

        # Mask raw voxel patches before either the CNN or transformer sees the
        # volume. This is the key anti-leakage requirement for masked modeling.
        vis_idx, mask_idx, token_mask = self._random_masking(grid, B, device)
        masked_x = self._mask_voxel_patches(x, token_mask, grid)
        enc_out = self.encoder(masked_x, return_tokens=True)
        all_tokens = enc_out["patch_tokens"]  # [B, N, d_model]
        n_tokens = all_tokens.shape[1]
        if n_tokens != int(math.prod(grid)):
            raise RuntimeError(f"Encoder produced {n_tokens} tokens for expected grid {grid}")

        # Gather visible tokens for decoder
        visible_tokens = all_tokens.gather(
            1, vis_idx.unsqueeze(-1).expand(-1, -1, all_tokens.shape[-1])
        )

        # --- Decode ---
        pred = self.decoder(visible_tokens, vis_idx, mask_idx, n_tokens)
        # pred: [B, N_mask, patch_voxels]

        # --- Build reconstruction target ---
        # Patchify the input volume to get per-token voxel targets
        # Ensure volume is divisible by total_stride; center-crop/pad if needed
        _, _, D, H, W = x.shape
        tD = (D // self.total_stride) * self.total_stride
        tH = (H // self.total_stride) * self.total_stride
        tW = (W // self.total_stride) * self.total_stride
        x_cropped = x[:, :, :tD, :tH, :tW]
        all_patches = self._patchify(x_cropped, self.total_stride)  # [B, N, patch_voxels]

        # Gather only masked patch targets
        target = all_patches.gather(
            1, mask_idx.unsqueeze(-1).expand(-1, -1, all_patches.shape[-1])
        )  # [B, N_mask, patch_voxels]

        # Optionally normalize targets per-patch
        if self.norm_target:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var.sqrt() + 1e-6)

        # --- Reconstruction loss on masked patches only ---
        if self.reconstruction_loss == "mse":
            loss = F.mse_loss(pred, target)
        elif self.reconstruction_loss == "l1":
            loss = F.l1_loss(pred, target)
        else:
            loss = F.smooth_l1_loss(pred, target)

        return {
            "loss": loss,
            "pred": pred,
            "target": target,
            "mask_ratio": x.new_tensor(self.mask_ratio),
            "n_mask": mask_idx.shape[1],
            "n_vis": vis_idx.shape[1],
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_mae3d(config: dict[str, Any]) -> MaskedAutoEncoder3D:
    """Instantiate the full MAE from config."""
    enc_cfg = config["encoder"]
    encoder = MRICNNTransformerEncoder(
        input_channels=int(enc_cfg.get("input_channels", 1)),
        stem_channels=int(enc_cfg.get("stem_channels", 16)),
        block_channels=list(enc_cfg.get("block_channels", [32, 64, 128, 256])),
        downsample_stride=int(enc_cfg.get("downsample_stride", 2)),
        dropout3d=float(enc_cfg.get("dropout3d", 0.0)),
        d_model=int(enc_cfg.get("d_model", 512)),
        transformer_layers=int(enc_cfg.get("transformer_layers", 2)),
        transformer_heads=int(enc_cfg.get("transformer_heads", 8)),
        transformer_ff_dim=int(enc_cfg.get("transformer_ff_dim", 1024)),
        transformer_dropout=float(enc_cfg.get("transformer_dropout", 0.0)),
        output_dim=int(enc_cfg.get("output_dim", 512)),
        return_tokens=True,
    )
    return MaskedAutoEncoder3D(encoder, config)
