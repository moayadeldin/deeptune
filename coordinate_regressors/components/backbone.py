import io
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision import models

class RegressorBackbone(nn.Module):

    def __init__(self, backbone="resnet18",pretrained=True,heatmap_size=28):

        super().__init__()

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        else:
            weights = None

        net = getattr(models,backbone)(weights=weights)

        self.features = nn.Sequential(*list(net.children())[:-2])
        self.heatmap_size = heatmap_size

        # upsample from the backbone's feature map  (7x7 for a 224px input on resnet 18) back up to heatmap_size x heatmap_size, then collapse to a single-channel heatmap

        feat_dim = net.fc.in_features if hasattr(net,"fc") else 512

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, 256, kernel_size=4, stride=2, padding=1),  # 7  -> 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),        # 14 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),         # 28 -> 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.heatmap_conv = nn.Conv2d(64, 1, kernel_size=1) # collapse to a single-channel heatmap

        # normalized coordinate grids for soft-argmax. Registerd as buffers so they move with the correct device when model.to(device) automatically called are not learnable parameters.

        ys, xs = torch.meshgrid(
            torch.linspace(0,1,heatmap_size),
            torch.linspace(0,1,heatmap_size),
            indexing = "ij",
        )

        self.register_buffer("grid_x", xs.reshape(1,-1)) # (1,H*W)
        self.register_buffer("grid_y", ys.reshape(1,-1)) # (1,H*W)

    def forward(self,x):

        feat = self.features(x) # (B,512,7,7) for resnet18 with 224px input 
        feat = self.deconv(feat) # (B, 64, 56, 56)
        heatmap = self.heatmap_conv(feat) # (B,1,56,56)

        B = heatmap.shape[0]
        flat = heatmap.view(B,-1) # (B,H*W)
        probs = torch.softmax(flat,dim=1) # spatial softmax to get a probability distribution over the heatmap

        # use the probability distribution to compute the expected coordinates

        x_coord = (probs * self.grid_x).sum(dim=1, keepdim=True)
        y_coord = (probs * self.grid_y).sum(dim=1, keepdim=True)
        coords = torch.cat([x_coord, y_coord], dim=1)  # (B, 2), each in [0, 1]

        return coords, heatmap.view(B, self.heatmap_size, self.heatmap_size)
