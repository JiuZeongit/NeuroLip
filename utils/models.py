# utils/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class LearnableTemporalAllocation(nn.Module):
    """LTA: lightweight MLP for temporal allocation weights."""

    def __init__(self, num_bins: int):
        super().__init__()
        self.num_bins = num_bins
        self.time_scale = nn.Parameter(torch.tensor(1.0))
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, t_diff: torch.Tensor, bin_idx: torch.Tensor) -> torch.Tensor:
        x = torch.stack([t_diff * self.time_scale, bin_idx.float()], dim=1)
        return self.mlp(x).squeeze(-1)


class TemporalAwareVoxelEncoder(nn.Module):
    """TVE = LTA + local spatial aggregation + temporal channel reweighting."""

    def __init__(self, voxel_dimension=(9, 160, 200)):
        super().__init__()
        self.num_bins, self.height, self.width = voxel_dimension
        self.lta = LearnableTemporalAllocation(num_bins=self.num_bins)
        self.temporal_channel_reweight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * self.num_bins, 2 * self.num_bins, kernel_size=1),
            nn.Sigmoid(),
        )

    def _empty_voxel(self, device: torch.device, batch_size: int = 1):
        return torch.zeros(batch_size, 2 * self.num_bins, self.height, self.width, device=device)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """events: [N, 5] => columns [x, y, t, p, b]"""
        if events.numel() == 0:
            return self._empty_voxel(events.device if events.is_cuda else torch.device("cpu"), batch_size=1)

        # infer batch size from appended batch index
        batch_size = int(events[:, -1].max().item()) + 1
        num_bins, h, w = self.num_bins, self.height, self.width
        device = events.device

        x = events[:, 0].long()
        y = events[:, 1].long()
        t = events[:, 2].clone()
        p = ((events[:, 3] + 1) / 2).long()  # {-1,+1} -> {0,1}
        b = events[:, 4].long()

        # Per-sample time normalization to [0,1]
        for bi in range(batch_size):
            mask = (b == bi)
            if not mask.any():
                continue
            t_b = t[mask]
            t_min = t_b.min()
            t_max = t_b.max()
            t[mask] = (t_b - t_min) / (t_max - t_min + 1e-6)

        num_voxels = batch_size * 2 * num_bins * h * w
        vox_flat = events.new_zeros((num_voxels,), dtype=torch.float32)

        # LSA asymmetric offsets (self, +x, +y)
        offsets = ((0, 0), (1, 0), (0, 1))
        denom = (num_bins - 1) if num_bins > 1 else 1

        for bi in range(batch_size):
            batch_mask = (b == bi)
            if not batch_mask.any():
                continue

            xb = x[batch_mask]
            yb = y[batch_mask]
            tb = t[batch_mask]
            pb = p[batch_mask]
            num_events = xb.numel()
            if num_events == 0:
                continue

            # Precompute LTA weights for all bins
            weights_per_bin = []
            for i_bin in range(num_bins):
                t_diff = tb - (i_bin / denom)
                bin_idx = torch.full((num_events,), i_bin, device=device, dtype=tb.dtype)
                weights_per_bin.append(self.lta(t_diff, bin_idx))

            batch_offset = bi * 2 * num_bins * h * w
            polarity_offset_base = pb * (num_bins * h * w)

            for dx, dy in offsets:
                x_shift = xb + dx
                y_shift = yb + dy
                valid = (x_shift >= 0) & (x_shift < w) & (y_shift >= 0) & (y_shift < h)
                if not valid.any():
                    continue

                xv = x_shift[valid]
                yv = y_shift[valid]
                pov = polarity_offset_base[valid]
                spatial_offset = yv * w + xv

                for i_bin in range(num_bins):
                    weights = weights_per_bin[i_bin][valid]
                    channel_offset = i_bin * h * w
                    idx = batch_offset + pov + channel_offset + spatial_offset
                    vox_flat.put_(idx, weights, accumulate=True)

        vox = vox_flat.view(batch_size, 2, num_bins, h, w)
        vox = torch.cat([vox[:, 0], vox[:, 1]], dim=1)  # [B,2C,H,W]
        att = self.temporal_channel_reweight(vox)
        return vox * att


class StructureAwareSpatialEnhancer(nn.Module):
    """SSE = channel compression + direction-aware smoothing + channel attention."""

    def __init__(self, in_channels: int, compressed_channels: int = 16):
        super().__init__()
        self.channel_compression = nn.Conv2d(in_channels, compressed_channels, kernel_size=1)
        # Direction-aware smoothing (horizontal smoothing via depthwise Conv1d over width after reshape)
        self.direction_aware_smoothing = nn.Sequential(
            nn.Conv1d(compressed_channels, compressed_channels, kernel_size=3, padding=1, groups=compressed_channels),
            nn.BatchNorm1d(compressed_channels),
            nn.ReLU(),
        )
        self.spatial_channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(compressed_channels, compressed_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_compression(x)
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w)
        x_smooth = self.direction_aware_smoothing(x_flat).view(b, c, h, w)
        return x_smooth * self.spatial_channel_attention(x_smooth)


class NeuroLipClassifier(nn.Module):
    def __init__(
        self,
        voxel_dimension=(9, 160, 200),
        crop_dimension=(224, 224),
        compressed_channels=16,
        num_classes=50,
        pretrained=False,
    ):
        super().__init__()
        self.num_bins, _, _ = voxel_dimension
        self.crop_dimension = crop_dimension
        self.num_classes = num_classes

        # TVE + SSE
        self.tve = TemporalAwareVoxelEncoder(voxel_dimension=voxel_dimension)
        self.sse = StructureAwareSpatialEnhancer(in_channels=2 * self.num_bins, compressed_channels=compressed_channels)

        # PCR reconstruction head
        self.pcr_head = nn.Sequential(
            nn.Conv2d(compressed_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=1),
        )

        # Backbone
        self.backbone = resnet34(pretrained=pretrained)
        self.backbone.conv1 = nn.Conv2d(compressed_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.fc = nn.Linear(512, num_classes)

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=self.crop_dimension, mode="bilinear", align_corners=False)

    def forward(self, events: torch.Tensor, return_aux: bool = False):
        voxel_att = self.tve(events)           # [B,2C,H,W]
        enhanced = self.sse(voxel_att)         # [B,C',H,W]

        pcr_recon = self.pcr_head(enhanced) if self.training else None

        x = self._resize(enhanced)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        features = F.adaptive_avg_pool2d(x, 1).flatten(1)
        logits = self.fc(features)

        if return_aux:
            return logits, voxel_att, pcr_recon, features
        return logits, voxel_att


def polarity_consistency_regularization_loss(recon: torch.Tensor, voxel_att: torch.Tensor, num_bins: int) -> torch.Tensor:
    """PCR loss: L1 between normalized reconstructed polarity maps and voxel-derived polarity maps."""
    # voxel_att channel layout: [neg bins..., pos bins...] in original code semantics may be pos then neg;
    # keep consistent with existing training behavior by using first C and second C partitions.
    part_a = voxel_att[:, :num_bins].sum(dim=1, keepdim=True)
    part_b = voxel_att[:, num_bins:].sum(dim=1, keepdim=True)
    target = torch.cat([part_a, part_b], dim=1)

    # Normalize spatial distributions (per channel map jointly as original simplified implementation)
    target = target / (target.sum(dim=[1, 2, 3], keepdim=True) + 1e-6)
    recon = recon / (recon.sum(dim=[1, 2, 3], keepdim=True) + 1e-6)
    return F.l1_loss(recon, target)


def train_step(model: NeuroLipClassifier, events: torch.Tensor, labels: torch.Tensor, optimizer, use_pcr: bool = True, pcr_lambda: float = 0.05):
    model.train()
    optimizer.zero_grad()

    logits, voxel_att, pcr_recon, _ = model(events, return_aux=True)
    ce_loss = F.cross_entropy(logits, labels)

    if use_pcr and pcr_recon is not None:
        pcr_loss = polarity_consistency_regularization_loss(pcr_recon, voxel_att, model.num_bins)
        total_loss = ce_loss + pcr_lambda * pcr_loss
    else:
        total_loss = ce_loss

    total_loss.backward()
    optimizer.step()

    pred_labels = logits.argmax(dim=1)
    return total_loss.item(), pred_labels
