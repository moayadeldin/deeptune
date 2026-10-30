import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class RegressorBackbone(nn.Module):

    def __init__(self, backbone="resnet18",pretrained=True,heatmap_size=28):

        super().__init__()

        self.backbone = backbone

        weights = models.get_model_weights(backbone).DEFAULT if pretrained else None
        net = getattr(models,backbone)(weights=weights)

        if backbone.startswith("vit_"):
            self.features = net
            self.features.heads = nn.Identity()
            feat_dim = net.hidden_dim

        elif backbone.startswith("swin_"):
            self.features = nn.Sequential(net.features, net.norm)
            feat_dim = net.head.in_features

        elif backbone.startswith("resnet"):
            self.features = nn.Sequential(*list(net.children())[:-2])
            feat_dim = net.fc.in_features

        else:
            self.features = net.features

            if backbone.startswith("densenet"):
                feat_dim = net.classifier.in_features
            elif backbone.startswith("efficientnet"):
                feat_dim = net.classifier[-1].in_features
            else:
                feat_dim = 512

        self.heatmap_size = heatmap_size

        # upsample from the backbone's feature map  (7x7 for a 224px input on resnet 18) back up to heatmap_size x heatmap_size, then collapse to a single-channel heatmap

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

        if self.backbone.startswith("vit_"):
            tokens = self.features._process_input(x)
            B, N, C = tokens.shape

            tokens = torch.cat([
                self.features.class_token.expand(B, -1, -1),
                tokens
            ], dim=1)

            tokens = self.features.encoder(tokens)[:, 1:]

            side = int(N ** 0.5)
            feat = tokens.transpose(1, 2).reshape(B, C, side, side)

        else:
            feat = self.features(x) # (B,512,7,7) for resnet18 with 224px input

            if self.backbone.startswith("swin_"):
                feat = feat.permute(0, 3, 1, 2)
            elif self.backbone.startswith("densenet"):
                feat = F.relu(feat)

        feat = self.deconv(feat) # (B, 64, 56, 56)
        heatmap = self.heatmap_conv(feat) # (B,1,56,56)

        heatmap = F.interpolate(
            heatmap,
            size=(self.heatmap_size, self.heatmap_size),
            mode="bilinear",
            align_corners=False
        )

        B = heatmap.shape[0]
        flat = heatmap.view(B,-1) # (B,H*W)
        probs = torch.softmax(flat,dim=1) # spatial softmax to get a probability distribution over the heatmap

        # use the probability distribution to compute the expected coordinates

        x_coord = (probs * self.grid_x).sum(dim=1, keepdim=True)
        y_coord = (probs * self.grid_y).sum(dim=1, keepdim=True)
        coords = torch.cat([x_coord, y_coord], dim=1)  # (B, 2), each in [0, 1]

        return coords, heatmap.view(B, self.heatmap_size, self.heatmap_size)