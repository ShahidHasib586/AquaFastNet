import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

def ssim_loss(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x*x, 3, 1, 1) - mu_x*mu_x
    sigma_y = F.avg_pool2d(y*y, 3, 1, 1) - mu_y*mu_y
    sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x + sigma_y + C2)
    s = num / (den + 1e-8)
    return torch.clamp((1 - s.mean()), 0, 1)

class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.slice = nn.Sequential(*[vgg[i] for i in range(16)]).eval()
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, x, y):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        x = (x - mean) / std
        y = (y - mean) / std
        fx = self.slice(x)
        fy = self.slice(y)
        return F.l1_loss(fx, fy)

class ComboLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.2, w_perc=0.05, objective="composite"):
        super().__init__()
        if objective not in {"composite", "l1", "mse"}:
            raise ValueError("objective must be composite, l1, or mse")
        self.objective = objective
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_perc = w_perc
        self.perc = VGGPerceptual() if objective == "composite" and w_perc else None
    def forward(self, pred, gt):
        l1 = F.l1_loss(pred, gt)
        if self.objective in {"l1", "mse"}:
            total = l1 if self.objective == "l1" else F.mse_loss(pred, gt)
            return total, {"l1": l1.item(), "ssim": 0.0, "perc": 0.0}
        ls = ssim_loss(pred, gt)
        lp = self.perc(pred, gt) if self.perc is not None else pred.new_tensor(0.0)
        total = self.w_l1*l1 + self.w_ssim*ls + self.w_perc*lp
        return total, {"l1": l1.item(), "ssim": ls.item(), "perc": lp.item()}
