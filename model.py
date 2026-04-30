"""Model architecture, constants, and inference utilities for land cover segmentation."""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

NUM_CLASSES = 6
IGNORE_INDEX = 255
CLASS_NAMES = ['urban', 'agriculture', 'rangeland', 'forest', 'water', 'barren']

PALETTE = np.array([
    [  0, 255, 255],   # urban
    [255, 255,   0],   # agriculture
    [255,   0, 255],   # rangeland
    [  0, 255,   0],   # forest
    [  0,   0, 255],   # water
    [255, 255, 255],   # barren
], dtype=np.uint8)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=NUM_CLASSES, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_channels, base)
        self.d2 = DoubleConv(base,   base*2)
        self.d3 = DoubleConv(base*2, base*4)
        self.d4 = DoubleConv(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.c3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.c2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.c1 = DoubleConv(base*2, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.pool(x1))
        x3 = self.d3(self.pool(x2))
        x4 = self.d4(self.pool(x3))
        x = self.u3(x4); x = self.c3(torch.cat([x, x3], 1))
        x = self.u2(x);  x = self.c2(torch.cat([x, x2], 1))
        x = self.u1(x);  x = self.c1(torch.cat([x, x1], 1))
        return self.head(x)


def colorize(mask_np: np.ndarray) -> np.ndarray:
    """Map class ids -> RGB using PALETTE; ignore_index -> black."""
    out = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    valid = mask_np != IGNORE_INDEX
    ids = np.clip(mask_np, 0, NUM_CLASSES - 1)
    out[valid] = PALETTE[ids[valid]]
    return out


def load_model(weights_path: str = "model/best_model.pth") -> SmallUNet:
    model = SmallUNet(num_classes=NUM_CLASSES)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_image(model: SmallUNet, pil_image: Image.Image, size: int = 256) -> np.ndarray:
    """Run inference on a PIL image and return a colorized RGB prediction."""
    img = pil_image.convert('RGB').resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(1)[0].numpy()
    return colorize(pred)
