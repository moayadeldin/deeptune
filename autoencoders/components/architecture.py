import torch
import torch.nn as nn
import torchvision.transforms as T
# from autoencoders.components.zscore import ZScore

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm="group", groups=8) -> torch.Tensor:
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)]
        if norm == "group":
            g = min(groups, out_ch)
            while out_ch % g != 0:
                g -= 1
            layers += [nn.GroupNorm(g, out_ch)]
        layers += [nn.SiLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class AutoEncoder(nn.Module):

    def __init__(self,use_deeper=False, in_ch=1, base=32, blur_k=5, blur_simga=1.0) -> torch.Tensor: #change in_ch to 3 if RGB

        super().__init__()

        # self.norm = ZScore()

        self.depth = "deep" if use_deeper else None

        self.blur = T.GaussianBlur(kernel_size=blur_k, sigma=blur_simga)

        # Encoder : downsample by 2x at each block 3 times

        self.enc1 = ConvBlock(in_ch,base)
        self.down1 = nn.Conv2d(base,base*2,kernel_size=4, stride=2, padding=1)

        self.enc2 = ConvBlock(base*2, base*2)
        self.down2 = nn.Conv2d(base*2, base*4, kernel_size=4, stride=2, padding=1, bias=False)       

        self.enc3 = ConvBlock(base*4, base*4)
        self.down3 = nn.Conv2d(base*4, base*8, kernel_size=4, stride=2, padding=1, bias=False)

        if self.depth == "deep":

            self.enc4 = ConvBlock(base*8, base*8)
            self.down4 = nn.Conv2d(base*8, base*16, kernel_size=4, stride=2, padding=1, bias=False)

            self.enc5 = ConvBlock(base*16, base*16)
            self.down5 = nn.Conv2d(base*16, base*32, kernel_size=4, stride=2, padding=1, bias=False)

            self.bottleneck = ConvBlock(base*32, base*32)
            self.up5 = nn.ConvTranspose2d(base*32, base*16, kernel_size=4, stride=2, padding=1, bias=False)
            self.dec5 = ConvBlock(base*16, base*16)
            self.up4 = nn.ConvTranspose2d(base*16, base*8, kernel_size=4, stride=2, padding=1, bias=False)
            self.dec4 = ConvBlock(base*8, base*8)
        else:

            self.bottleneck = ConvBlock(base*8, base*8)

            # Decoder : upsample by 2x at each block 3 times

        self.up3 = nn.ConvTranspose2d(base*8, base*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec3 = ConvBlock(base*4, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec2 = ConvBlock(base*2, base*2)

        self.up1 = nn.ConvTranspose2d(base*2, base, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec1 = ConvBlock(base, base)

        # Last layer to get back to in_ch channels
        self.out = nn.Conv2d(base, in_ch, kernel_size=3, padding=1)

    def forward(self,x):

        # x_target = self.norm(x)
        x_target = x
        x_blur = self.blur(x_target)

        x1 = self.enc1(x_blur)
        x = self.down1(x1)

        x2 = self.enc2(x)
        x = self.down2(x2)

        x3 = self.enc3(x)
        x = self.down3(x3)

        if self.depth == "deep":

            x4 = self.enc4(x)
            x = self.down4(x4)
            x5 = self.enc5(x)
            x = self.down5(x5)
            z = self.bottleneck(x)
            x = self.up5(z)
            x = self.dec5(x)
            x = self.dec4(self.up4(x))

        else:
            x = self.bottleneck(x)

        x = self.up3(x)
        x = self.dec3(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        out = self.out(x)
        out = torch.sigmoid(out)

        return out, x_target