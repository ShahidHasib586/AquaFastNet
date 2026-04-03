import csv, random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def load_rgb(path: str):
    return Image.open(path).convert("RGB")

def to_tensor(img: Image.Image):
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).contiguous()

class PairDataset(Dataset):
    """
    training=True  : random crop + flips (augmentation)
    training=False : deterministic center crop (stable benchmark)
    """
    def __init__(self, csv_path: str, crop=256, training=True):
        self.rows = []
        with open(csv_path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                self.rows.append((row["input"], row["gt"]))
        self.crop = int(crop)
        self.training = bool(training)

    def __len__(self):
        return len(self.rows)

    def _pad_if_needed(self, x, y, c):
        _, h, w = x.shape
        pad_h = max(0, c - h)
        pad_w = max(0, c - w)
        if pad_h == 0 and pad_w == 0:
            return x, y

        # reflect requires pad < dim; otherwise replicate is safe
        def choose_mode(h, w, ph, pw):
            if h > 1 and w > 1 and ph < h and pw < w:
                return "reflect"
            return "replicate"

        mode = choose_mode(h, w, pad_h, pad_w)
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        y = F.pad(y, (0, pad_w, 0, pad_h), mode=mode)
        return x, y

    def _random_crop(self, x, y, c):
        x, y = self._pad_if_needed(x, y, c)
        _, H, W = x.shape
        top = random.randint(0, H - c)
        left = random.randint(0, W - c)
        return x[:, top:top+c, left:left+c], y[:, top:top+c, left:left+c]

    def _center_crop(self, x, y, c):
        x, y = self._pad_if_needed(x, y, c)
        _, H, W = x.shape
        top = (H - c) // 2
        left = (W - c) // 2
        return x[:, top:top+c, left:left+c], y[:, top:top+c, left:left+c]

    def __getitem__(self, idx):
        a, b = self.rows[idx]
        x = to_tensor(load_rgb(a))
        y = to_tensor(load_rgb(b))

        c = self.crop

        if self.training:
            x, y = self._random_crop(x, y, c)
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2]); y = torch.flip(y, dims=[2])
            if random.random() < 0.5:
                x = torch.flip(x, dims=[1]); y = torch.flip(y, dims=[1])
        else:
            # stable evaluation
            x, y = self._center_crop(x, y, c)

        return x, y, Path(a).name
