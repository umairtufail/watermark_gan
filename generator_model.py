import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True, **kwargs):
        super(ConvBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=False, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.cnn(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.survival_prob = 0.8
        self.block1 = ConvBlock(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.block2 = ConvBlock(in_channels, in_channels, kernel_size=3, padding=1, stride=1, use_act=False)

    def stochastic_depth(self, x):
        if not self.training:
            return x
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.stochastic_depth(out) + x
        return out

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, act="relu"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if act == "relu" else nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    def __init__(self, in_channels, features=64, num_residuals=9):
        super(Generator, self).__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=1, padding=3, padding_mode="reflect", bias=True),
            nn.ReLU(inplace=True),
        )
        self.down1 = Block(features, features * 2, act="relu")
        self.down2 = Block(features * 2, features * 4, act="relu")
        self.down3 = Block(features * 4, features * 8, act="relu")
        self.down4 = Block(features * 8, features * 16, act="relu")
        self.residuals = nn.Sequential(*[ResidualBlock(features * 16) for _ in range(num_residuals)])
        self.up1 = Block(features * 16, features * 8, act="relu")
        self.up2 = Block(features * 8 * 2, features * 4, act="relu")
        self.up3 = Block(features * 4 * 2, features * 2, act="relu")
        self.up4 = Block(features * 2 * 2, features, act="relu")
        self.final_conv = nn.Sequential(
    Block(features * 2, features, stride=1, act="relu"),
    nn.ConvTranspose2d(features, features // 2, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Conv2d(features // 2, in_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect", bias=True),
    nn.Tanh(),
)


    def adjust_channels(self, x, target_channels):
        if x.size(1) == target_channels:
            return x
        else:
            adjustment_layer = nn.Conv2d(x.size(1), target_channels, kernel_size=1, stride=1, padding=0, bias=False)
            return adjustment_layer(x)

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        residuals = self.residuals(d5) + d5

        up1 = self.up1(F.interpolate(residuals, scale_factor=2, mode="nearest"))
        up1 = torch.cat([up1, d4[:, :, :up1.size(2), :up1.size(3)]], dim=1)

        up2 = self.up2(F.interpolate(up1, scale_factor=2, mode="nearest"))
        up2 = torch.cat([up2, d3[:, :, :up2.size(2), :up2.size(3)]], dim=1)

        up3 = self.up3(F.interpolate(up2, scale_factor=2, mode="nearest"))
        up3 = torch.cat([up3, d2[:, :, :up3.size(2), :up3.size(3)]], dim=1)

        up4 = self.up4(F.interpolate(up3, scale_factor=2, mode="nearest"))
        up4 = torch.cat([up4, d1[:, :, :up4.size(2), :up4.size(3)]], dim=1)

        return self.final_conv(up4)


