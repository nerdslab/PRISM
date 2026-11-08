import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os, math
from src.arch.layers.MCD import MCD
from src.arch.layers.SEBlock import SEBlock
from src.arch.visual import save_individual_bands_plot
from src.arch.layers.filters import (
    dog_gauss_bands,
    ema_ladder_bands,
    binomial_pyramid_bands,
    fft_dyadic_cosine_bands,
    fft_dyadic_rect_bands,
    fft_butterworth_diff_bands,
    dwt_haar_bands_full,
    dwt_wavelet_bands_full,
    diffuse_multiscale_plus,
    LearnableConvFilter,
)


class prism(nn.Module):
    """
    PRISM model for time series forecasting with hierarchical time decomposition.
    Uses TimeSplitBackbone with TimeTree for multi-scale temporal representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_components: int = 5,
        dropout: float = 0.0,
        seq_len: int = 336,
        pred_len: int = 96,
        save_dir: Optional[str] = None,
        overlap: Optional[int] = 24,
        decomp_kind: str = "haar",
        tree_depth: int = 2,
        use_last_layer_only: bool = False,
        selector_hidden: int = 32,
        disable_internal_norm: bool = False,
        l1_lambda: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_components = num_components
        self.dropout = dropout
        self.K = num_components  # Use num_components as the unified parameter
        self.tree_depth = tree_depth
        self.l1_lambda = l1_lambda
        # Default save dir to exp_prism if not provided
        self.save_dir = (
            save_dir
            if save_dir is not None
            else os.path.join("./temp", "logs", "exp_prism")
        )
        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception:
            pass
        self.backbone = TimeSplitBackbone(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            tree_depth=self.tree_depth,
            overlap=overlap,
            save_dir=self.save_dir,
            num_components=num_components,
            decomp_kind=decomp_kind,
            use_last_layer_only=use_last_layer_only,
            selector_hidden=selector_hidden,
            disable_internal_norm=disable_internal_norm,
            l1_lambda=self.l1_lambda,
        )

        # Print tree structure after initialization (only once)
        if not hasattr(prism, "_tree_printed"):
            print(f"\n=== PRISM Tree Structure ===")
            print(f"Input shape: [B, L={self.seq_len}, C={self.input_dim}]")
            if hasattr(self.backbone, "tree"):
                self.backbone.tree.print_tree_structure_by_level(
                    input_shape=f"[B, L={self.seq_len}, C={self.input_dim}]",
                    model_name="PRISM",
                )
            print("=" * 40)
            prism._tree_printed = True

    def forecast(
        self,
        x: torch.Tensor,
        tk: Optional[torch.Tensor] = None,
        bxm: Optional[torch.Tensor] = None,
        bym: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate only the future horizon from pooled latent z.
        Returns x_fore: [B, pred_len, input_dim]
        """
        # Enable test_mode for band plotting when y is available
        # Debug: Check input
        ssm = self.backbone(x, mode="forecast", y=y)  # → [B, T, C]
        return ssm

    def reconstruct(
        self,
        x: torch.Tensor,
        tk: Optional[torch.Tensor] = None,
        bxm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ssm = self.backbone(x)  # [B, seq_len, input_dim]
        return ssm

    def forward(self, batch_x, batch_y=None, bxm=None, bym=None):
        """Forward pass - returns forecast tensor (not loss)"""
        forecast = self.forecast(batch_x, y=batch_y)  # Forecast future
        # forecast_loss = F.mse_loss(forecast, batch_y, reduction='mean')  # MSE loss for reconstruction
        # # # Check for NaNs in loss
        # # recon = self.reconstruct(batch_x)  # Forecast future
        # # recon_x_loss = F.mse_loss(recon, batch_x, reduction='mean')
        # loss = forecast_loss #+ recon_x_loss
        return forecast

    def set_current_epoch(self, epoch):
        """Set the current epoch for plotting purposes."""
        self.backbone.set_current_epoch(epoch)


class TimeSplitBackbone(nn.Module):
    """
    Input:  x[B,L,C]   Output: [B,L,C]
    Channel-separable fanout -> TimeTree over **time** -> head.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_features=0,
        dropout=0.0,
        tree_depth=2,
        overlap=8,
        seq_len=336,
        pred_len=None,
        save_dir=None,
        decomp_kind="haar",
        num_components=6,
        use_last_layer_only=False,
        selector_hidden=32,
        disable_internal_norm=False,
        l1_lambda: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        # Only use internal normalization if not disabled
        self.norm = SeriesNorm() if not disable_internal_norm else None

        # channel-separable expand: [B,C,L] -> [B*C, H, L]
        # self.cs_init = nn.Conv1d(input_dim, input_dim*hidden_dim, 1, groups=input_dim, bias=True)
        self.tree = TimeTree(
            C=input_dim,
            total_depth=tree_depth,
            depth=tree_depth,
            overlap=overlap,
            seq_len=seq_len,  # infer from input at forward()
            pred_len=pred_len,
            K_imf=num_components,  # number of bands (use num_components parameter)
            temperature=0.2,  # Gumbel-softmax routing temp
            save_dir=save_dir,
            decomp_kind=decomp_kind,
            use_last_layer_only=use_last_layer_only,
            selector_hidden=selector_hidden,
            l1_lambda=l1_lambda,
        )

    def forward(self, x, t=None, features=None, mode="route", y=None):  # x: [B,L,C]
        B, L, C = x.shape
        x_c = x.transpose(1, 2).contiguous()  # [B,C,L]
        if self.norm:
            # normalize per (B,C) over time
            x_in, stats = self.norm(x_c)
        else:
            x_in, stats = x_c, None

        # Normalize y with the same (mu, sigma) as x if provided
        y_in = None
        if y is not None:
            y_c = y.transpose(1, 2).contiguous()  # [B,C,H]
            if self.norm and stats is not None:
                mu, sig = stats
                y_in = ((y_c - mu) / sig).transpose(1, 2)
            else:
                y_in = y_c.transpose(1, 2)

        out, _ = self.tree(x_in, y=y_in, mode=mode)  # forecast: [B,C,H], else: [B,C,L]
        if mode == "forecast":
            if self.norm:  # invert to original scale
                out = self.norm.invert(out, stats)  # [B,C,H]
            return out.transpose(1, 2).contiguous()  # [B,H,C]

        # route/reconstruct: residual in normalized space, then invert
        # out = out + x_in                       # [B,C,L]
        if self.norm:
            out = self.norm.invert(out, stats)  # back to original scale
        return out.transpose(1, 2).contiguous()  # [B,L,C]

    def set_current_epoch(self, epoch):
        """Set the current epoch for plotting purposes."""
        if hasattr(self, "tree") and hasattr(self.tree, "set_current_epoch"):
            self.tree.set_current_epoch(epoch)


# -------------------------
# Your TimeTree with modes
# -------------------------
class TimeTree(nn.Module):
    def __init__(
        self,
        C,
        total_depth,
        depth,
        kernel_size=7,
        overlap=24,
        dropout=0.0,
        K_imf=6,
        temperature=0.7,
        router_type="linear",
        selector_hidden=32,
        seq_len=336,
        pred_len=None,
        forecast_hidden=128,
        save_dir=None,
        decomp_kind="haar",
        use_last_layer_only=False,
        print_tree_on_init=False,
        l1_lambda=0.01,
    ):
        super().__init__()
        self.depth = int(depth)
        self.total_depth = int(total_depth)
        self.C = C
        self.overlap = int(overlap)
        # causal removed from tree hyperparameters
        self.use_last_layer_only = use_last_layer_only
        # IMF/decomp params
        self.K = int(K_imf)
        self.temperature = float(temperature)
        # decomposition kind (set in _decompose)
        self.decomp_kind = decomp_kind  # or: "haar","dog","fft_dycos","fft_rect","fft_butter","ema","binom"
        # optional knobs removed: pass constants directly at call sites
        if self.decomp_kind == "learnable_gauss":
            self.learnable_gauss = LearnableConvFilter(C=C, K=self.K)
        # forecast head (root only uses it)
        self.pred_len = None if pred_len is None else int(pred_len)
        self.seq_len = int(seq_len)
        self.save_dir = save_dir

        # test mode flag for plotting control
        self.test_mode = False

        # Initialize current epoch for plotting
        self._current_epoch = 0

        self.l1_lambda = l1_lambda
        # self.band_feat = IMFTimeFeat(feat_ch=C, k=self.K)
        # self.mlp_score = BandSelectorMLP(F=C, hidden=selector_hidden, temperature=self.temperature)
        if self.depth == 0:
            self.left = self.right = None
            # Add band_feat for leaf nodes too
            # Use simple feature extractor (6 features)
            self.band_feat = IMFSharpFeatTiny()
            self.mlp_score = BandSelectorMLP(
                F=6, hidden=selector_hidden, temperature=self.temperature
            )

            # self.mlp_score = BandSelectorMLP(F=14, hidden=selector_hidden, temperature=self.temperature,
            #                                hidden_sizes=[selector_hidden, selector_hidden//2], dropout=0.1)
        elif self.depth != 0 and self.total_depth > self.depth:
            self.band_feat = IMFSharpFeatTiny()
            self.mlp_score = BandSelectorMLP(
                F=6, hidden=selector_hidden, temperature=self.temperature
            )
            # self.mlp_scoreL = BandSelectorMLP(
            #     F=6, hidden=selector_hidden, temperature=self.temperature
            # )
            # self.mlp_scoreR = BandSelectorMLP(
            #     F=6, hidden=selector_hidden, temperature=self.temperature
            # )
            self.left = TimeTree(
                C,
                total_depth,
                depth - 1,
                kernel_size=kernel_size,
                overlap=overlap,
                dropout=dropout,
                K_imf=self.K,
                temperature=self.temperature,
                router_type=router_type,
                selector_hidden=selector_hidden,
                seq_len=seq_len,
                pred_len=pred_len,
                forecast_hidden=forecast_hidden,
                decomp_kind=self.decomp_kind,
            )
            self.right = TimeTree(
                C,
                total_depth,
                depth - 1,
                kernel_size=kernel_size,
                overlap=overlap,
                dropout=dropout,
                K_imf=self.K,
                temperature=self.temperature,
                router_type=router_type,
                selector_hidden=selector_hidden,
                seq_len=seq_len,
                pred_len=pred_len,
                forecast_hidden=forecast_hidden,
                decomp_kind=self.decomp_kind,
            )
            self.mix = nn.Identity()
        else:
            # Calculate sizes based on tree depth
            if self.use_last_layer_only:
                # For 2-layer system using only last layer, calculate only level0 size
                level_sizes = self.calculate_level_sizes(
                    self.seq_len, overlap, self.total_depth
                )
                level0_input_size = level_sizes[
                    0
                ]  # First layer (level0) for 2-layer system
                # Level sizes calculated for projection layers
                # Initialize only level0 projection layers
                self.forecast_proj1_level0 = nn.ModuleList(
                    [
                        nn.Linear(level0_input_size, forecast_hidden)
                        for i in range(2 * self.K)
                    ]
                )
                self.forecast_proj2_level0 = nn.ModuleList(
                    [
                        nn.Linear(forecast_hidden, self.pred_len)
                        for i in range(2 * self.K)
                    ]
                )
            else:
                # Initialize projection layers for all levels (collect all levels)
                # Calculate level sizes and projection layer counts dynamically
                level_sizes = self.calculate_level_sizes(
                    self.seq_len, overlap, self.total_depth
                )
                # Initialize projection layers for each level
                for level in range(self.total_depth):
                    level_size = level_sizes[level]
                    num_proj_layers = (2**level) * 2 * self.K
                    # Initialize proj1 and proj2 layers for this level
                    setattr(
                        self,
                        f"forecast_proj1_level{level}",
                        nn.ModuleList(
                            [
                                nn.Linear(level_size, forecast_hidden)
                                for _ in range(num_proj_layers)
                            ]
                        ),
                    )
                    setattr(
                        self,
                        f"forecast_proj2_level{level}",
                        nn.ModuleList(
                            [
                                nn.Linear(forecast_hidden, self.pred_len)
                                for _ in range(num_proj_layers)
                            ]
                        ),
                    )

            self.activate = nn.LeakyReLU()
            # self.band_feat = IMFTimeFeat(feat_ch=C, k=self.K)
            # self.band_feat = IMFMorphoFeatScalar(kernel_size=(1,3), soft_max=True, beta=20)
            # Use simple feature extractor (6 features)
            self.band_feat = IMFSharpFeatTiny()

            if self.use_last_layer_only:
                # When using only last layer, total_bands is just 2*K
                total_bands = 2 * self.K
            else:
                # When using all levels, calculate total bands from all levels
                total_bands = 2 * self.K * (2**self.total_depth - 1)
            self.SEBlock_all = SEBlock(mode="avg", K_IMP=total_bands)
            self.mlp_score = BandSelectorMLP(
                F=6, hidden=selector_hidden, temperature=self.temperature
            )
            # self.mlp_scoreL = BandSelectorMLP(
            #     F=6, hidden=selector_hidden, temperature=self.temperature
            # )
            # self.mlp_scoreR = BandSelectorMLP(
            #     F=6, hidden=selector_hidden, temperature=self.temperature
            # )
            self.left = TimeTree(
                C,
                total_depth,
                depth - 1,
                kernel_size=kernel_size,
                overlap=overlap,
                dropout=dropout,
                K_imf=self.K,
                temperature=self.temperature,
                router_type=router_type,
                selector_hidden=selector_hidden,
                pred_len=pred_len,
                forecast_hidden=forecast_hidden,
                seq_len=seq_len,
                decomp_kind=self.decomp_kind,
            )
            self.right = TimeTree(
                C,
                total_depth,
                depth - 1,
                kernel_size=kernel_size,
                overlap=overlap,
                dropout=dropout,
                K_imf=self.K,
                temperature=self.temperature,
                router_type=router_type,
                selector_hidden=selector_hidden,
                pred_len=pred_len,
                forecast_hidden=forecast_hidden,
                seq_len=seq_len,
                decomp_kind=self.decomp_kind,
            )
            self.mix = nn.Identity()

        # Always print once at the root to show tree structure
        if self.depth == self.total_depth and not hasattr(self, "_printed"):
            try:
                self.print_tree_structure()
            except Exception:
                pass
            setattr(self, "_printed", True)

    # Calculate input sizes dynamically based on overlap (non-recursive)
    def calculate_level_sizes(self, seq_len, overlap, depth):
        """Compute per-level time lengths using actual split logic."""

        # Simulate the actual split logic from _split_with_overlap
        def simulate_split(L, overlap):
            mid = L // 2
            ov = min(overlap, mid, L - mid)
            if ov > 0:
                # Ensure we have enough space for overlap
                if (mid + ov) > L:
                    ov = L - mid
                if (mid - ov) < 0:
                    ov = mid
                l0, l1 = 0, mid + ov
                r0, r1 = mid - ov, L
                split_len = min(l1 - l0, r1 - r0)
                return split_len
            else:
                # No overlap case
                return mid

        level_sizes = []
        current_len = seq_len

        for level in range(depth):
            split_len = simulate_split(current_len, overlap)
            level_sizes.append(split_len)
            current_len = split_len

        return level_sizes

    # ---------- helpers ----------
    def _split_with_overlap(self, x):
        B, C, L = x.shape
        mid = L // 2
        ov = min(self.overlap, mid, L - mid)

        # Proper overlap handling to eliminate edge effects
        if ov > 0:
            # Ensure we have enough space for overlap
            if (mid + ov) > L:
                ov = L - mid
            if (mid - ov) < 0:
                ov = mid

            l0, l1 = 0, mid + ov
            r0, r1 = mid - ov, L
            split_len = min(l1 - l0, r1 - r0)
            l1 = l0 + split_len
            r0 = r1 - split_len
        else:
            # No overlap case
            l0, l1 = 0, mid
            r0, r1 = mid, L

        xL = x[..., l0:l1]
        xR = x[..., r0:r1]
        return xL, xR, (l0, l1, r0, r1, ov, L)

    def _stitch(self, yL, yR, meta, t_emb=None):
        l0, l1, r0, r1, ov, L = meta
        B, C = yL.size(0), yL.size(1)
        y = yL.new_zeros(B, C, L)

        # Linear interpolation for overlapping region concatenation
        if ov > 0 and r0 < l1:
            # Non-overlapping left part [0, r0)
            if r0 > 0:
                y[..., :r0] = yL[..., :r0]

            # Overlapping region [r0, l1)
            overlap_len = max(0, l1 - r0)
            if overlap_len > 0:
                # yL corresponds to global [0:l1], overlap segment is [r0:l1]
                left_overlap = yL[..., r0:l1]
                # yR corresponds to global [r0:L], overlap segment starts at [0:overlap_len]
                right_overlap = yR[..., :overlap_len]
                w = torch.linspace(
                    0, 1, steps=overlap_len, device=y.device, dtype=y.dtype
                )
                w = w.view(1, 1, -1)
                y[..., r0:l1] = left_overlap * (1 - w) + right_overlap * w

            # Non-overlapping right part [l1, L)
            if l1 < L:
                # Remaining part of yR starts from overlap_len
                y[..., l1:] = yR[..., overlap_len : overlap_len + (L - l1)]
        else:
            # Direct concatenation when no overlap or index anomaly
            if l1 <= L:
                y[..., :l1] = yL
            else:
                y[..., :L] = yL[..., :L]

            if r0 < L:
                start_idx = max(0, l1 - r0)
                end_idx = min(L, r1 - r0)
                if start_idx < end_idx:
                    y[..., start_idx:end_idx] = yR[..., : (end_idx - start_idx)]

        return y

    @torch.no_grad()
    def _decompose(self, x, levels=None):  # [B,C,L] -> [B,C,K,L]
        kind = getattr(self, "decomp_kind", "diffuse")  # set this attr in __init__
        B, C, L = x.shape

        if kind == "diffuse":
            # Use fixed defaults; causal behavior not propagated
            result = diffuse_multiscale_plus(x, K=self.K, steps0=1, grow=2)
        elif kind == "learnable_gauss":
            with torch.enable_grad():
                if not hasattr(self, "learnable_gauss"):
                    self.learnable_gauss = LearnableConvFilter(
                        C=C, K=self.K, sig0=1.0, ratio=1.6
                    )
                return self.learnable_gauss(x)
        elif kind == "haar" or kind.startswith("db") or kind.startswith("sym"):
            # Check if Haar decomposition is feasible
            max_levels = int(torch.log2(torch.tensor(L, dtype=torch.float32)).item())
            required_levels = self.K - 1  # Haar needs levels=K-1 to produce K bands

            if required_levels > max_levels:
                # Haar decomposition cannot produce enough bands, switch to EMA method
                # print(f"Warning: Haar decomposition cannot produce {self.K} bands for L={L} (max_levels={max_levels}), switching to EMA method")
                result = ema_ladder_bands(x, K=self.K, tau0=8.0, grow=3.0)
            else:
                levels = required_levels if levels is None else int(levels)
                if kind == "haar":
                    result = dwt_haar_bands_full(x, levels)
                else:
                    result = dwt_wavelet_bands_full(x, levels, wavelet=kind)
        elif kind == "dog":
            result = dog_gauss_bands(x, K=self.K, sig0=1.0, ratio=1.6)
        elif kind == "ema":
            result = ema_ladder_bands(x, K=self.K, tau0=8.0, grow=3.0)
        elif kind == "binom":
            result = binomial_pyramid_bands(x, K=self.K, k0=3, kgrow=2)
        elif kind == "fft_dycos":
            result = fft_dyadic_cosine_bands(x, K=self.K, transition=0.1)
        elif kind == "fft_rect":
            result = fft_dyadic_rect_bands(x, K=self.K)
        elif kind == "fft_butter":
            result = fft_butterworth_diff_bands(x, K=self.K, order=4, w0=None)
        elif kind == "mcd":
            # Initialize MCD if not already done
            if not hasattr(self, "mcd_decomposer"):
                self.mcd_decomposer = MCD(
                    K_IMP=self.K,
                    kernel_size=getattr(self, "mcd_kernel_size", (1, 3)),
                    soft_max=getattr(self, "mcd_soft_max", True),
                    beta=getattr(self, "mcd_beta", 20),
                )
            # Move MCD to the same device as input tensor
            self.mcd_decomposer = self.mcd_decomposer.to(x.device)
            # MCD expects input shape [B, N, T] and returns [B, N, K, T]
            # We need to transpose from [B, C, L] to [B, C, L] for MCD
            result = self.mcd_decomposer(x)  # [B, C, L] -> [B, C, K, L]
        else:
            raise ValueError(f"unknown decomp_kind={kind}")

        return result

        # -------- short helpers --------

    def _sum_bands(self, imfs, w):  # imfs:[B,C,K,L], w:[B,C,K] -> [B,C,L]
        return torch.einsum("bckl,bck->bcl", imfs, w)

    def _get_per_band_forecast(self, bands):
        """Generate per-band forecast contributions for visualization"""
        per_band_forecast = []

        # Use the projection layers for the current node's level
        # Map depth to projection layer names: depth=0->level0, depth=1->level1, etc.
        # Map depth to projection level: depth=3->level0, depth=2->level1, depth=1->level2
        proj_level = f"level{self.total_depth - self.depth}"
        proj1_attr = f"forecast_proj1_{proj_level}"
        proj2_attr = f"forecast_proj2_{proj_level}"

        if hasattr(self, proj1_attr) and hasattr(self, proj2_attr):
            proj1_layers = getattr(self, proj1_attr)
            proj2_layers = getattr(self, proj2_attr)
        else:
            raise AttributeError(
                f"Per-band forecast projection layers are not initialized for level {proj_level}."
            )
        num_bands = min(bands.shape[2], len(proj1_layers))
        for i in range(num_bands):
            band_input = bands[:, :, i, :]  # [B, C, L]
            band_forecast = proj2_layers[i](self.activate(proj1_layers[i](band_input)))
            per_band_forecast.append(band_forecast)
        return torch.stack(per_band_forecast, dim=2)  # [B,C,num_bands,pred_len]

    def set_current_epoch(self, epoch):
        """Set the current epoch for plotting purposes."""
        self._current_epoch = epoch
        # Reset plotted epochs when epoch changes to allow plotting in new epoch
        if hasattr(self, "_plotted_epochs"):
            self._plotted_epochs.clear()
        # Recursively set epoch for all child nodes
        if hasattr(self, "left") and self.left is not None:
            self.left.set_current_epoch(epoch)
        if hasattr(self, "right") and self.right is not None:
            self.right.set_current_epoch(epoch)

    def print_tree_structure(self, level=0, prefix="", input_shape=None):
        """Print the tree structure with input/output shapes"""
        level_name = f"Level{level}"
        indent = "  " * level

        # Get current node info
        node_type = "Leaf" if self.depth == 0 else "Internal"
        overlap_info = f"overlap={self.overlap}" if hasattr(self, "overlap") else ""

        print(
            f"{indent}{prefix}{level_name} ({node_type}): depth={self.depth}, {overlap_info}"
        )

        if input_shape is not None:
            print(f"{indent}  Input shape: {input_shape}")

        # Calculate output shape for this node
        if hasattr(self, "K") and hasattr(self, "overlap"):
            # For internal nodes, output is the bands
            if self.depth > 0:
                output_shape = f"[B, C, {self.K}, L_local]"
            else:
                output_shape = f"[B, C, {self.K}, L_local] (leaf)"
            print(f"{indent}  Output shape: {output_shape}")

        # Recursively print children
        if hasattr(self, "left") and self.left is not None:
            print(f"{indent}  ├─ Left child:")
            self.left.print_tree_structure(level + 1, "├─ ", input_shape)

        if hasattr(self, "right") and self.right is not None:
            print(f"{indent}  └─ Right child:")
            self.right.print_tree_structure(level + 1, "└─ ", input_shape)

    def print_tree_structure_by_level(self, input_shape=None, model_name="PRISM"):
        """Print tree structure organized by level with practical information"""
        # Extract input dimensions
        if input_shape:
            # Parse input_shape like "[B, L=48, C=109]"
            import re

            L_match = re.search(r"L=(\d+)", input_shape)
            C_match = re.search(r"C=(\d+)", input_shape)
            seq_len = int(L_match.group(1)) if L_match else 48
            channels = int(C_match.group(1)) if C_match else 1
        else:
            seq_len = 48
            channels = 1

        # Calculate sizes for each level using non-recursive method
        level_sizes = []
        for level in range(self.total_depth):
            if level == 0:  # Root level
                size = seq_len // 2 + self.overlap
            else:  # Deeper levels
                size = seq_len // (2 ** (level + 1)) + self.overlap
            level_sizes.append(size)

        # Calculate total bands used in SEBlock (all bands from all levels)
        # For depth=d: total_bands = 2K * (1 + 2 + 4 + ... + 2^(d-1)) = 2K * (2^d - 1)
        total_se_bands = 2 * self.K * (2**self.total_depth - 1)

        print(f"\n=== {model_name} Tree Structure ===")
        print(f"Input shape: [B, L={seq_len}, C={channels}]")
        print(f"Tree Depth: {self.total_depth}, Overlap: {self.overlap}")
        print(f"SEBlock Total Bands: {total_se_bands} (all levels combined)")
        print("=" * 50)

        # Print practical info for each level in one line
        for level in range(self.total_depth):
            level_name = f"Level{level}"
            if level == self.total_depth - 1:
                node_type = "Leaf"
                children_info = "Leaf node"
            else:
                node_type = "Internal"
                children_info = f"Children: Left & Right (Level{level+1})"

            # Calculate practical information
            input_seq_len = level_sizes[level]
            bands_count = self.K  # Each level has K bands
            mlp_input_dim = 16  # IMFAdvancedFeat output dimension
            total_bands = bands_count  # Each level has K bands, not multiplied

            # Get actual projection layer information for this level
            # Map tree levels to projection levels: Level0->level0, Level1->level1, Level2->level2
            proj1_attr = f"forecast_proj1_level{level}"
            proj2_attr = f"forecast_proj2_level{level}"

            if hasattr(self, proj1_attr) and hasattr(self, proj2_attr):
                proj1_layers = getattr(self, proj1_attr)
                proj2_layers = getattr(self, proj2_attr)
                proj_input_len = (
                    proj1_layers[0].in_features if len(proj1_layers) > 0 else "N/A"
                )
                proj_output_len = (
                    proj2_layers[0].out_features if len(proj2_layers) > 0 else "N/A"
                )
                proj_count = len(proj1_layers)
                proj_info = f"Proj[{proj_input_len}->{proj_output_len}]x{proj_count}"
            else:
                proj_info = f"Proj[NotInitialized]"

            # One line format with all practical info
            print(
                f"{level_name} ({node_type}): Input[B,L={input_seq_len},C={channels}] -> Bands[{bands_count}] -> MLP[{mlp_input_dim}] -> TotalBands[{total_bands}] -> {proj_info} | Overlap:{self.overlap} | {children_info}"
            )

        print("=" * 50)

    def forward(
        self, x, mode="route", meta=None, side=None, global_step=0, y=None, stats=None
    ):
        """
        Returns:
        if mode == 'forecast' and at root: [B,C,H]
        else: (y, bands_up)
            y:        [B,C,L_local]          (time-domain at this node)
            bands_up: [B,C,K_or_2K,L_parent] (bands already aligned to parent length)
        """

        # ---- leaf ----
        if self.depth == 0:
            # 1) decompose into K bands at the leaf
            imfs_leaf = self._decompose(x)  # [B,C,K,L_leaf]
            # Compute per-band features and weights at leaves too
            feats = self.band_feat(imfs_leaf)  # [B,C,K,F]
            w = self.mlp_score(feats)  # [B,C,K]
            weighted = imfs_leaf * w.unsqueeze(-1)  # [B,C,K,L_leaf]
            # Return as level0 bands; will be concatenated up the tree
            return x, {"level0": weighted}

        # ---- internal node: decompose once, route each child by band-softmax ----
        # print(f"[d={self.depth}] before split x: {tuple(x.shape)}")
        xL_raw, xR_raw, meta = self._split_with_overlap(x)
        # print(f"[d={self.depth}] after split xL: {tuple(xL_raw.shape)}, xR: {tuple(xR_raw.shape)}")
        imfsL = self._decompose(xL_raw)  # [B,C,K,L]
        imfsR = self._decompose(xR_raw)  # [B,C,K,L]
        # print(f"[d={self.depth}] bands L: {tuple(imfsL.shape)}, R: {tuple(imfsR.shape)}")

        featsL = self.band_feat(imfsL)  # [B,C,K,feat_ch]
        featsR = self.band_feat(imfsR)  # [B,C,K,feat_ch]
        wL_raw = self.mlp_score(featsL)  # [B,C,K]
        wR_raw = self.mlp_score(featsR)  # [B,C,K]

        # Add inter-branch competition: left and right branches compete
        # Concatenate weights from both branches for softmax competition
        combined_weights = torch.stack([wL_raw, wR_raw], dim=-1)  # [B,C,K,2]
        branch_competition = torch.softmax(
            combined_weights / self.temperature, dim=-1
        )  # [B,C,K,2]

        # Extract competitive weights for each branch
        wL = branch_competition[
            ..., 0
        ]  # [B,C,K] - left branch weights after competition
        wR = branch_competition[
            ..., 1
        ]  # [B,C,K] - right branch weights after competition
        # Expose current weights for analysis/plotting.
        self._current_wL = wL
        self._current_wR = wR
        self._last_wL = wL
        self._last_wR = wR

        # Store weighted bands for this level
        weighted_bands_L = imfsL * wL.unsqueeze(-1)  # [B,C,K,L]
        weighted_bands_R = imfsR * wR.unsqueeze(-1)  # [B,C,K,L]

        # Debug: Check band sizes at this level
        # print(f"Level {self.depth} - Left bands: {weighted_bands_L.shape}, Right bands: {weighted_bands_R.shape}")
        # print(f"  - Left time dim: {weighted_bands_L.shape[-1]}, Right time dim: {weighted_bands_R.shape[-1]}")

        # band-weighted inputs per child (use precomputed weighted bands)
        xL = weighted_bands_L.sum(dim=2)  # [B,C,LL]
        xR = weighted_bands_R.sum(dim=2)  # [B,C,LR]

        # recurse with same mode (route vs reconstruct differs only at leaves)
        yL, dictL = self.left(
            xL,
            mode=mode,
            meta=meta,
            side="L",
            global_step=global_step,
            y=y,
            stats=stats,
        )
        yR, dictR = self.right(
            xR,
            mode=mode,
            meta=meta,
            side="R",
            global_step=global_step,
            y=y,
            stats=stats,
        )

        # stitch siblings
        out = self._stitch(yL, yR, meta)

        # Merge bands dicts from children by concatenating along band axis if both present
        bands_dict = {}
        child_keys = set(dictL.keys()) | set(dictR.keys())
        for key in child_keys:
            if key in dictL and key in dictR:
                bands_dict[key] = torch.cat([dictL[key], dictR[key]], dim=2)
            elif key in dictL:
                bands_dict[key] = dictL[key]
            else:
                bands_dict[key] = dictR[key]

        # Current level bands at this node
        current_bands = torch.cat(
            [weighted_bands_L, weighted_bands_R], dim=2
        )  # [B,C,2K,L_here]
        # Assign to semantic levels to achieve root totals
        # Map depth to level names: depth=total_depth->level0, depth=total_depth-1->level1, etc.
        level_idx = self.total_depth - self.depth
        level_name = f"level{level_idx}"
        bands_dict[level_name] = current_bands

        # ---- forecast at root ----
        if mode == "forecast" and self.depth == self.total_depth:

            # Use all collected levels for forecasting
            # print collected bands at root for debugging if needed
            # print(', '.join([f"{k}:{tuple(v.shape)}" for k,v in bands_dict.items()]))
            total_imf = current_bands

            # Plotting logic - do this BEFORE forecast_all_levels to use root bands
            current_epoch = self._current_epoch
            should_plot = getattr(self, "test_mode", False) and y is not None
            if should_plot:
                with torch.no_grad():
                    per_band_forecast = self._get_per_band_forecast(
                        total_imf
                    )  # [B,C,2K,pred_len]
                B = total_imf.shape[0]  # Get batch size
                sample_idx = current_epoch % B  # Cycle through different samples
                # Compose subfolder by hyperparameters for band images
                subfolder = f"pd{self.pred_len}_s{getattr(self, 'seed', 'NA')}_w{self.K}_dp_{getattr(self, 'dropout', 'NA')}"
                bands_dir = (
                    os.path.join(self.save_dir, "bands_imgs", subfolder)
                    if self.save_dir
                    else None
                )
                save_individual_bands_plot(
                    bands_left=imfsL,
                    bands_right=imfsR,
                    weights_left=wL,
                    weights_right=wR,
                    x_ref=x,
                    per_band_forecast=per_band_forecast,
                    side_tag="root",
                    tag=f"test_epoch{current_epoch}_root_d{self.depth}",
                    b_idx=sample_idx,
                    c_idx=0,
                    original_future=y,
                    save_dir=bands_dir,
                    seq_len=x.shape[2],
                    pred_len=self.pred_len,  # x.shape[2] is C (channels/features)
                    current_epoch=current_epoch,
                )

                # if hasattr(wandb, "run") and wandb.run is not None and img_path:
                #     wandb.log(
                #         {
                #             "bands/worst_sample": wandb.Image(img_path),
                #         },
                #         step=current_epoch,
                #     )

                self._last_plotted_epoch = current_epoch

            out = self.forecast_all_levels(bands_dict)  # [B,C,H]
            return out, None
        return self.mix(out), bands_dict

    def forecast_all_levels(self, all_levels_bands):
        """
        Iterate through each level, call project_level,
        concatenate and optionally pass through SEBlock_all, then sum over K dimension to get final prediction.
        """
        # Only print bands info on first call (optional, can be commented out if not needed)
        # if not hasattr(self, '_bands_monitored'):
        #     available_levels = list(all_levels_bands.keys())
        #     print(f"[Bands monitoring] Received levels: {available_levels}")
        #     for k, v in all_levels_bands.items():
        #         print(f"  {k}: {v.shape}")
        #     self._bands_monitored = True

        per_level_y = []
        # If only using last layer (your use_last_layer_only logic), can be preserved at outer level
        for level in range(self.total_depth):
            level_name = f"level{level}"
            if level_name not in all_levels_bands:
                continue
            y_imf = self.project_level(
                all_levels_bands[level_name], level_name
            )  # [B,C,Ki,H]
            if y_imf.shape[2] > 0:
                per_level_y.append(y_imf)

        if len(per_level_y) == 0:
            # Extreme case: all levels were skipped, return zeros
            any_bands = list(all_levels_bands.values())[0]
            return torch.zeros(
                any_bands.size(0),
                any_bands.size(1),
                self.pred_len,
                device=any_bands.device,
            )

        y_all = torch.cat(per_level_y, dim=2)  # [B, C, K_total_used, pred_len]

        # Disable SEBlock
        # if self.use_seblock and hasattr(self, 'SEBlock_all') and self.SEBlock_all is not None and y_all.shape[2] > 0:
        y_all = self.SEBlock_all(y_all)

        return torch.sum(y_all, dim=2)  # [B, C, pred_len]

    def project_level(self, bands, level_name):
        """
        Project bands at a given level using level-specific projection layers.
        Returns: [B, C, K_used, pred_len]
        """
        # Get existing projection layers, check if they exist
        proj1_layers = None
        proj2_layers = None

        if level_name == "level0" and hasattr(self, "forecast_proj1_level0"):
            proj1_layers = self.forecast_proj1_level0
            proj2_layers = self.forecast_proj2_level0
        elif level_name == "level1" and hasattr(self, "forecast_proj1_level1"):
            proj1_layers = self.forecast_proj1_level1
            proj2_layers = self.forecast_proj2_level1
        elif level_name == "level2" and hasattr(self, "forecast_proj1_level2"):
            proj1_layers = self.forecast_proj1_level2
            proj2_layers = self.forecast_proj2_level2
        elif level_name == "level3" and hasattr(self, "forecast_proj1_level3"):
            proj1_layers = self.forecast_proj1_level3
            proj2_layers = self.forecast_proj2_level3
        elif level_name == "level4" and hasattr(self, "forecast_proj1_level4"):
            proj1_layers = self.forecast_proj1_level4
            proj2_layers = self.forecast_proj2_level4

        # If this level has no projection layers (e.g., leaf node), return empty tensor
        if proj1_layers is None or proj2_layers is None:
            B, C, K_in, L = bands.shape
            return torch.zeros(B, C, 0, self.pred_len, device=bands.device)

        B, C, K_in, L = bands.shape
        num_heads = min(K_in, len(proj1_layers))
        y_chunks = []

        for i in range(num_heads):
            band_i = bands[:, :, i, :]  # [B, C, L]
            z = self.activate(proj1_layers[i](band_i))  # [B, C, H]
            y_i = proj2_layers[i](z)  # [B, C, pred_len]
            y_chunks.append(y_i.unsqueeze(2))  # Stack into K dimension

        return torch.cat(y_chunks, dim=2)  # [B, C, K_used, pred_len]

    def forecast_level(self, bands, level_name):
        """Forecast using level-specific projection layers"""
        num_bands = bands.shape[2]  # Get actual number of bands from input

        # Select the appropriate projection layers for this level (shared SE)
        if level_name == "level0":
            proj1_layers = self.forecast_proj1_level0
            proj2_layers = self.forecast_proj2_level0
        elif level_name == "level1":
            proj1_layers = self.forecast_proj1_level1
            proj2_layers = self.forecast_proj2_level1
        elif level_name == "level2":
            proj1_layers = self.forecast_proj1_level2
            proj2_layers = self.forecast_proj2_level2
        elif level_name == "level3":
            proj1_layers = self.forecast_proj1_level3
            proj2_layers = self.forecast_proj2_level3
        else:
            raise ValueError(f"Unknown level: {level_name}")

        # print(f"Level {level_name} - num_bands: {num_bands}, proj_layers: {len(proj1_layers)}")

        for i in range(num_bands):
            y_current_imf = proj2_layers[i](
                self.activate(proj1_layers[i](bands[:, :, i, :]))
            )
            y_current_imf = y_current_imf.unsqueeze(2)
            if i == 0:
                y_imf = y_current_imf
            else:
                y_imf = torch.cat((y_imf, y_current_imf), axis=2)
        # Apply SE over bands if available; otherwise, leave as-is
        y_imf = self.SEBlock_all(y_imf)
        y = torch.sum(y_imf, 2)  # (B, N, T) where T is already pred_len

        return y  # (B, N, T) where T = pred_len

    def forecast(self, total_imf):
        # Use ALL bands from the combined multi-level representation
        num_bands = total_imf.shape[2]  # Get actual number of bands from input
        max_proj_layers = len(self.forecast_proj1)  # Available projection layers

        print(
            f"Prism forecast - num_bands: {num_bands}, max_proj_layers: {max_proj_layers}, total_imf shape: {total_imf.shape}"
        )

        for i in range(num_bands):
            # Now we have enough projection layers for all bands
            proj_idx = i

            y_current_imf = self.forecast_proj2[proj_idx](
                self.activate(self.forecast_proj1[proj_idx](total_imf[:, :, i, :]))
            )
            y_current_imf = y_current_imf.unsqueeze(2)
            if i == 0:
                y_imf = y_current_imf
            else:
                y_imf = torch.cat((y_imf, y_current_imf), axis=2)
        y_imf = self.SEBlock(y_imf)
        y = torch.sum(y_imf, 2)  # (B, N, T) where T is already pred_len

        return y  # (B, N, T) where T = pred_len


# -------------------------
# Minimal dispatcher for your TimeTree._decompose
# -------------------------
class SeriesNorm(nn.Module):
    """
    Per-(B,C) z-norm over the *time* axis.
    layout='BCL'  -> expects x:[B,C,L]  (time is last dim)
    layout='BLC'  -> expects x:[B,L,C]  (time is middle dim)
    """

    def __init__(self, eps: float = 1e-5, layout: str = "BCL"):
        super().__init__()
        assert layout in ("BCL", "BLC")
        self.eps = eps
        self.layout = layout

    def forward(self, x):
        if self.layout == "BCL":
            # x: [B,C,L], time = -1
            mu = x.mean(dim=-1, keepdim=True)
            # Check if we have enough elements to compute std
            if x.shape[-1] > 1:
                sig = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
            else:
                sig = torch.ones_like(mu) * self.eps
        else:
            # x: [B,L,C], time = 1
            mu = x.mean(dim=1, keepdim=True)
            # Check if we have enough elements to compute std
            if x.shape[1] > 1:
                sig = x.std(dim=1, keepdim=True).clamp_min(self.eps)
            else:
                sig = torch.ones_like(mu) * self.eps
        return (x - mu) / sig, (mu, sig)

    def invert(self, y, stats):
        mu, sig = stats
        return y * sig + mu


class IMFSharpFeatTiny(nn.Module):
    """
    imfs: [B,C,K,L] -> [B,C,K,F] with very small F (default 6).
    Sharp but cheap band descriptors (no convs, no params).
    """

    def __init__(self, eps: float = 1e-6, use_no_grad: bool = True):
        super().__init__()
        self.eps = eps
        self.use_no_grad = use_no_grad

    def forward(self, imfs: torch.Tensor) -> torch.Tensor:
        # imfs: [B,C,K,L]
        ctx = torch.no_grad() if self.use_no_grad else torch.enable_grad()
        with ctx:
            # 1) amplitude & spread
            mean = imfs.mean(dim=-1)  # [B,C,K]
            # Check if we have enough elements to compute std
            if (
                imfs.shape[-1] > 1
                and imfs.numel() > imfs.shape[0] * imfs.shape[1] * imfs.shape[2]
            ):
                std = imfs.std(dim=-1, unbiased=False)  # [B,C,K]
            else:
                std = torch.zeros_like(imfs.mean(dim=-1))  # [B,C,K]
            amax = imfs.abs().amax(dim=-1)  # [B,C,K]

            # 2) "sharpness" via finite differences
            d1 = torch.diff(imfs, dim=-1)  # [B,C,K,L-1]
            d2 = torch.diff(d1, dim=-1)  # [B,C,K,L-2]
            rough = d1.abs().mean(dim=-1)  # [B,C,K]
            bend = d2.abs().mean(dim=-1)  # [B,C,K]

            # 3) burstiness / contrast (dimensionless)
            burst = amax / (std + self.eps)  # [B,C,K]

            feats = torch.stack(
                [mean, std, amax, rough, bend, burst], dim=-1
            )  # [B,C,K,6]
            # (Optional) light normalization for numerical stability:
            # Ensure output is on same device as input
            feats = feats.to(imfs.device)
        return feats


class IMFHybridFeat(nn.Module):
    """
    Hybrid feature extractor for high-channel scenarios (336+ channels).
    Combines statistical features with compressed raw information.
    imfs: [B,C,K,L] -> [B,C,K,F] with F=16 features.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        use_no_grad: bool = True,
        raw_compression_ratio: float = 0.25,
    ):
        super().__init__()
        self.eps = eps
        self.use_no_grad = use_no_grad
        self.raw_compression_ratio = raw_compression_ratio

    def forward(self, imfs: torch.Tensor) -> torch.Tensor:
        # imfs: [B,C,K,L]
        ctx = torch.no_grad() if self.use_no_grad else torch.enable_grad()
        with ctx:
            B, C, K, L = imfs.shape

            # 1) Enhanced statistical features (8 features)
            mean = imfs.mean(dim=-1)  # [B,C,K]
            std = (
                imfs.std(dim=-1, unbiased=False)
                if imfs.shape[-1] > 1
                else torch.zeros_like(mean)
            )
            amax = imfs.abs().amax(dim=-1)  # [B,C,K]
            amin = imfs.abs().amin(dim=-1)  # [B,C,K]

            # Temporal dynamics
            d1 = torch.diff(imfs, dim=-1)  # [B,C,K,L-1]
            d2 = torch.diff(d1, dim=-1) if d1.shape[-1] > 1 else torch.zeros_like(d1)
            rough = d1.abs().mean(dim=-1)  # [B,C,K]
            bend = (
                d2.abs().mean(dim=-1) if d2.shape[-1] > 0 else torch.zeros_like(rough)
            )

            # 2) Frequency domain features (4 features)
            fft = torch.fft.fft(imfs, dim=-1)
            power = fft.abs()  # [B,C,K,L]
            dominant_freq = power.argmax(dim=-1).float() / L  # [B,C,K]
            spectral_centroid = (
                power * torch.arange(L, device=power.device).float()
            ).sum(dim=-1) / (power.sum(dim=-1) + self.eps)
            spectral_spread = (
                (
                    torch.arange(L, device=power.device).float()
                    - spectral_centroid.unsqueeze(-1)
                )
                ** 2
                * power
            ).sum(dim=-1) / (power.sum(dim=-1) + self.eps)
            energy = (imfs**2).sum(dim=-1)  # [B,C,K]

            # 3) Compressed raw information (4 features)
            # Downsample the raw signal to preserve key patterns
            compressed_len = max(1, int(L * self.raw_compression_ratio))
            if L > compressed_len:
                # Use max pooling to preserve important peaks
                compressed = (
                    torch.nn.functional.adaptive_max_pool1d(
                        imfs.view(B * C * K, L).unsqueeze(1), compressed_len
                    )
                    .squeeze(1)
                    .view(B, C, K, compressed_len)
                )
            else:
                compressed = imfs

            # Extract key points from compressed signal
            raw_max = compressed.abs().amax(dim=-1)  # [B,C,K]
            raw_mean = compressed.mean(dim=-1)  # [B,C,K]
            raw_std = (
                compressed.std(dim=-1, unbiased=False)
                if compressed.shape[-1] > 1
                else torch.zeros_like(raw_mean)
            )
            raw_range = compressed.amax(dim=-1) - compressed.amin(dim=-1)  # [B,C,K]

            # Combine all features
            feats = torch.stack(
                [
                    mean,
                    std,
                    amax,
                    amin,
                    rough,
                    bend,
                    dominant_freq,
                    spectral_centroid,
                    spectral_spread,
                    energy,
                    raw_max,
                    raw_mean,
                    raw_std,
                    raw_range,
                ],
                dim=-1,
            )  # [B,C,K,14]

            feats = feats.to(imfs.device)
        return feats


# Previous simpler MLP (kept for rollback)
class BandSelectorMLP(nn.Module):
    def __init__(self, F, hidden=32, temperature=0.7):
        super().__init__()
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(F, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):  # feats: [B,C,K,F]
        B, C, K, F = feats.shape
        feats_flat = feats.view(B * C * K, F)
        logits = self.net(feats_flat).view(B, C, K)
        weights = torch.softmax(logits / self.temperature, dim=2)
        denom = weights.sum(dim=2, keepdim=True)
        return weights / denom


# class BandSelectorMLP(nn.Module):
#     def __init__(self, F: int, hidden: int = 32, hidden_sizes: list | None = None, dropout: float = 0.0, temperature: float = 0.7):
#         super().__init__()
#         self.temperature = temperature
#         # Default to a stronger 2-layer MLP if not specified
#         layer_sizes = hidden_sizes if hidden_sizes is not None else [512, 512]
#         layers = []
#         in_dim = F
#         for hs in layer_sizes:
#             layers.append(nn.Linear(in_dim, hs))
#                 # Add batch normalization for 4D tensor: [B, C, K, F] -> [B*C*K, F] -> [B*C*K, hs]
#             layers.append(nn.BatchNorm1d(hs))
#             layers.append(nn.SiLU())
#             if dropout and dropout > 0:
#                 layers.append(nn.Dropout(p=float(dropout)))
#             in_dim = hs
#         layers.append(nn.Linear(in_dim, 1))
#         self.net = nn.Sequential(*layers)

#     def forward(self, feats: torch.Tensor) -> torch.Tensor:  # feats: [B,C,K,F]
#         B, C, K, Fdim = feats.shape
#         feats_flat = feats.view(B * C * K, Fdim)
#         logits = self.net(feats_flat).view(B, C, K)
#         weights = torch.softmax(logits / self.temperature, dim=2)
#         denom = weights.sum(dim=2, keepdim=True)
#         return weights / denom
