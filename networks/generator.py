import torch as th
import torch.nn as nn

from utils import initialize_weights, spectral_normalization


class InputBlock(nn.Module):
    def __init__(self, res, fmap_out):
        super().__init__()
        self.conv = nn.ConvTranspose2d(res, fmap_out * 2, 4, bias=False)
        self.bn = nn.BatchNorm2d(fmap_out * 2)
        self.glu = nn.GLU(dim=1)
        self.fmap_out = fmap_out

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.glu(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, fmap_in, fmap_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(fmap_in, fmap_out * 2, 3,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(fmap_out * 2)
        self.glu = nn.GLU(dim=1)
        self.fmap_out = fmap_out

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.glu(x)
        return x


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(th.zeros(1), requires_grad=True)

    def forward(self, x, noise=None):
        if noise is None:
            b, _, h, w = x.shape
            noise = th.randn(b, 1, h, w).type_as(x)

        return x + self.weight * noise


class UpsamplingCompBlock(nn.Module):
    def __init__(self, fmap_in, fmap_out, noise=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(fmap_in, fmap_out * 2, 3,
                               padding=1, bias=False)
        self.n1 = NoiseInjection() if noise else nn.Identity()
        self.bn1 = nn.BatchNorm2d(fmap_out * 2)
        self.conv2 = nn.Conv2d(fmap_out, fmap_out * 2, 3,
                               padding=1, bias=False)
        self.n2 = NoiseInjection() if noise else nn.Identity()
        self.bn2 = nn.BatchNorm2d(fmap_out * 2)
        self.glu = nn.GLU(dim=1)
        self.fmap_out = fmap_out

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.n1(x)
        x = self.bn1(x)
        x = self.glu(x)
        x = self.conv2(x)
        x = self.n2(x)
        x = self.bn2(x)
        x = self.glu(x)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class SLEBlock(nn.Module):
    def __init__(self, chan_in, chan_out, swish=True):
        super().__init__()
        Activation = Swish if swish else nn.LeakyReLU(0.1)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(chan_in, chan_out, 4, bias=False),
            Activation(),
            nn.Conv2d(chan_out, chan_out, 1, bias=False),
            nn.Sigmoid()
            )

    def forward(self, x_low, x_high):
        return self.net(x_low) * x_high


class OutputInterBlock(nn.Module):
    def __init__(self, fmap):
        super().__init__()
        self.conv = nn.Conv2d(fmap, 3, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x


class OutputFinalBlock(nn.Module):
    def __init__(self, fmap):
        super().__init__()
        self.conv = nn.Conv2d(fmap, 3, 3, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x


class Generator(nn.Module):
    def __init__(self, resolution, latent_dim=256, fmap_max=1024,
                 noise=False, sn=False):
        super().__init__()
        assert resolution in [256, 512, 1024]
        assert fmap_max >= 256
        self.resolution = resolution
        self.latent_dim = latent_dim

        self.input_block = InputBlock(latent_dim, min(1024, fmap_max))

        self.upsample_8 = UpsamplingCompBlock(min(1024, fmap_max),
                                              min(512, fmap_max), noise)
        self.upsample_16 = UpsamplingBlock(min(512, fmap_max), 256)
        self.upsample_32 = UpsamplingCompBlock(256, 128, noise)
        self.upsample_64 = UpsamplingBlock(128, 128)
        self.sle_4_64 = SLEBlock(self.input_block.fmap_out,
                                 self.upsample_64.fmap_out)
        self.upsample_128 = UpsamplingCompBlock(128, 64, noise)
        self.sle_8_128 = SLEBlock(self.upsample_8.fmap_out,
                                  self.upsample_128.fmap_out)
        self.upsample_256 = UpsamplingBlock(64, 32)
        self.sle_16_256 = SLEBlock(self.upsample_16.fmap_out,
                                   self.upsample_256.fmap_out)
        fmap_out = 32
        if resolution > 256:
            self.upsample_512 = UpsamplingCompBlock(32, 16, noise)
            self.sle_32_512 = SLEBlock(self.upsample_32.fmap_out,
                                       self.upsample_512.fmap_out)
            fmap_out = 16
        if resolution > 512:
            self.upsample_1024 = UpsamplingBlock(16, 8)
            fmap_out = 8

        self.out_int_block = OutputInterBlock(self.upsample_128.fmap_out)
        self.out_big_block = OutputFinalBlock(fmap_out)

        self.apply(initialize_weights)
        if sn:
            self.apply(spectral_normalization)

    def forward(self, x):
        x_4 = self.input_block(x)

        x_8 = self.upsample_8(x_4)
        x_16 = self.upsample_16(x_8)
        x_32 = self.upsample_32(x_16)

        x_64 = self.upsample_64(x_32)
        x_sle_64 = self.sle_4_64(x_4, x_64)

        x_128 = self.upsample_128(x_sle_64)
        x_sle_128 = self.sle_8_128(x_8, x_128)

        x_256 = self.upsample_256(x_sle_128)
        x = self.sle_16_256(x_16, x_256)

        if self.resolution > 256:
            x_512 = self.upsample_512(x)
            x = self.sle_32_512(x_32, x_512)

        if self.resolution > 512:
            x = self.upsample_1024(x)

        out_big = self.out_big_block(x)
        out_int = self.out_int_block(x_sle_128)

        return out_big, out_int


if __name__ == "__main__":
    gen = Generator(1024, 256, 1024, True, True)

    z = th.randn(1, 256, 1, 1)
    x_big, x_int = gen(z)
