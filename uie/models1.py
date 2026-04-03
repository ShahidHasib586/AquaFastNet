import torch
import torch.nn as nn
import torch.nn.functional as F


# Channel Shuffle

def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    b, c, h, w = x.shape
    if groups <= 1 or (c % groups) != 0:
        return x
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, c, h, w)
    return x


# Dynamic 1x1 (CondConv-style)

class DynamicPWConv(nn.Module):
    """
    Per-sample dynamic 1x1 conv: mixture of K experts.
    Efficient because it's 1x1 only.
    """
    def __init__(self, c_in: int, c_out: int, K: int = 4, r: int = 16):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.K = K

        self.weight = nn.Parameter(torch.randn(K, c_out, c_in, 1, 1) * 0.02)
        self.bias = nn.Parameter(torch.zeros(K, c_out))

        mid = max(4, c_in // r)
        self.route = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in, mid, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, K, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        alpha = self.route(x).flatten(1)         # (B,K)
        alpha = torch.softmax(alpha, dim=1)

        W = torch.einsum("bk,kocij->bocij", alpha, self.weight)  # (B,Co,Ci,1,1)
        bias = torch.einsum("bk,kc->bc", alpha, self.bias)       # (B,Co)

        # group-conv trick
        x_g = x.reshape(1, b * c, h, w)
        W_g = W.reshape(b * self.c_out, self.c_in, 1, 1)
        y = F.conv2d(x_g, W_g, bias=None, stride=1, padding=0, groups=b)
        y = y.reshape(b, self.c_out, h, w)
        y = y + bias.view(b, self.c_out, 1, 1)
        return y


# Depthwise 3x3 + Dynamic 1x1 + BN + SiLU

class DWConvDyn(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, K: int = 4):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in, bias=False)
        self.pw = DynamicPWConv(c_in, c_out, K=K)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# SE Attention

class SE(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        mid = max(1, c // r)
        self.fc1 = nn.Conv2d(c, mid, 1)
        self.fc2 = nn.Conv2d(mid, c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


# OSA Block (fixed for downsampling)

class OSA_Block(nn.Module):
    """
    One-Shot Aggregation:
      y1 = conv1(x) [optionally downsample]
      y2 = conv2(y1)
      ...
      concat [x_id, y1, y2, ...] -> 1x1 fuse -> shuffle -> SE
    IMPORTANT: if down=True, x_id must be downsampled to match y1 resolution.
    """
    def __init__(self, c_in: int, c_out: int, n_layers: int = 3, down: bool = False, K: int = 4, shuffle_groups: int = 2):
        super().__init__()
        self.down = down
        self.shuffle_groups = shuffle_groups

        stride0 = 2 if down else 1

        self.layers = nn.ModuleList()
        c_prev = c_in
        for i in range(n_layers):
            s = stride0 if i == 0 else 1
            self.layers.append(DWConvDyn(c_prev, c_out, stride=s, K=K))
            c_prev = c_out

        # downsample identity if needed
        self.id_pool = nn.AvgPool2d(2, 2) if down else nn.Identity()

        c_cat = c_in + n_layers * c_out
        self.fuse = nn.Sequential(
            nn.Conv2d(c_cat, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True)
        )
        self.se = SE(c_out, r=8)

    def forward(self, x):
        x_id = self.id_pool(x)  # matches spatial size if down=True

        feats = [x_id]
        y = x
        for conv in self.layers:
            y = conv(y)
            feats.append(y)

        # now all feats have same H,W
        z = torch.cat(feats, dim=1)
        z = self.fuse(z)
        z = channel_shuffle(z, groups=self.shuffle_groups)
        z = self.se(z)
        return z


# UpBlock

class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int, n_layers: int = 3, K: int = 4, shuffle_groups: int = 2):
        super().__init__()
        self.fuse = OSA_Block(c_in + c_skip, c_out, n_layers=n_layers, down=False, K=K, shuffle_groups=shuffle_groups)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

# Main Model

class FastUNetEnhancer(nn.Module):
    """
    HybridEnhanceNet v1:
    - Depthwise + Dynamic 1x1 (CondConv)
    - OSA aggregation blocks
    - Channel shuffle
    - SE attention
    - Proper UNet decode: first upblock concatenates s2
    """
    def __init__(self, base: int = 32, K: int = 4, osa_layers: int = 3, shuffle_groups: int = 2):
        super().__init__()
        b = base

        # Encoder
        self.e0 = OSA_Block(3,   b,   n_layers=osa_layers, down=False, K=K, shuffle_groups=shuffle_groups)  # H
        self.e1 = OSA_Block(b,   b*2, n_layers=osa_layers, down=True,  K=K, shuffle_groups=shuffle_groups)  # H/2
        self.e2 = OSA_Block(b*2, b*4, n_layers=osa_layers, down=True,  K=K, shuffle_groups=shuffle_groups)  # H/4
        self.e3 = OSA_Block(b*4, b*6, n_layers=osa_layers, down=True,  K=K, shuffle_groups=shuffle_groups)  # H/8

        self.mid = OSA_Block(b*6, b*6, n_layers=osa_layers, down=False, K=K, shuffle_groups=shuffle_groups)  # H/8

        # Decoder (FIXED: first up uses s2)
        self.u2 = UpBlock(b*6, b*4, b*4, n_layers=osa_layers, K=K, shuffle_groups=shuffle_groups)  # -> H/4, skip s2
        self.u1 = UpBlock(b*4, b*2, b*2, n_layers=osa_layers, K=K, shuffle_groups=shuffle_groups)  # -> H/2, skip s1
        self.u0 = UpBlock(b*2, b,   b,   n_layers=osa_layers, K=K, shuffle_groups=shuffle_groups)  # -> H,   skip s0

        self.head = nn.Conv2d(b, 3, 1)

    def forward(self, x):
        x_in = x

        s0 = self.e0(x)     # H
        s1 = self.e1(s0)    # H/2
        s2 = self.e2(s1)    # H/4
        s3 = self.e3(s2)    # H/8

        m  = self.mid(s3)   # H/8

        d2 = self.u2(m,  s2)  # H/4
        d1 = self.u1(d2, s1)  # H/2
        d0 = self.u0(d1, s0)  # H

        out = torch.sigmoid(self.head(d0))
        out = torch.clamp(out + 0.10 * x_in, 0.0, 1.0)
        return out
