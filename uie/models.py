import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, stride=s, padding=p, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        mid = max(1, c // r)
        self.fc1 = nn.Conv2d(c, mid, 1)
        self.fc2 = nn.Conv2d(mid, c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class Block(nn.Module):
    def __init__(self, c_in, c_out, down=False):
        super().__init__()
        stride = 2 if down else 1
        self.c1 = DWConv(c_in, c_out, s=stride)
        self.c2 = DWConv(c_out, c_out, s=1)
        self.se = SE(c_out)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return self.se(x)

class UpBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.fuse = Block(c_in + c_skip, c_out, down=False)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)

class FastUNetEnhancer(nn.Module):
    """
    Fully convolutional, preserves spatial size.
    Input : (B,3,H,W) in [0..1]
    Output: (B,3,H,W) in [0..1]
    """
    def __init__(self, base=32):
        super().__init__()
        b = base
        # encoder
        self.e0 = Block(3, b, down=False)       # H
        self.e1 = Block(b, b*2, down=True)      # H/2
        self.e2 = Block(b*2, b*4, down=True)    # H/4
        self.e3 = Block(b*4, b*6, down=True)    # H/8

        # bottleneck
        self.mid = Block(b*6, b*6, down=False)

        # decoder (upsample x3 back to H)
        self.u2 = UpBlock(b*6, b*6, b*4)        # H/4  (skip e3)
        self.u1 = UpBlock(b*4, b*4, b*2)        # H/2  (skip e2)
        self.u0 = UpBlock(b*2, b*2, b)          # H    (skip e1)

        self.head = nn.Conv2d(b, 3, 1)

    def forward(self, x):
        x_in = x
        s0 = self.e0(x)     # H
        s1 = self.e1(s0)    # H/2
        s2 = self.e2(s1)    # H/4
        s3 = self.e3(s2)    # H/8

        m = self.mid(s3)

        d2 = self.u2(m, s3)  # -> H/4? (actually ups to s3 size, so H/8; then fuse; keep H/8)
        # Important: to properly restore scale, we upsample to next skip sizes in sequence:
        d1 = self.u1(d2, s2) # -> H/4
        d0 = self.u0(d1, s1) # -> H/2

        # final upsample to H using s0 size
        d0 = F.interpolate(d0, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        out = torch.sigmoid(self.head(d0))

        # Residual on input (same shape guaranteed)
        out = torch.clamp(out + 0.10 * x_in, 0.0, 1.0)
        return out
