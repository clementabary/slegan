import random
import torch as th
import torch.nn as nn
import torch.nn.functional as tnf

from utils import initialize_weights, spectral_normalization
from tricks import DiffAugment


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


class InputBigBlock(nn.Module):
    def __init__(self, resolution):
        super().__init__()
        if resolution == 1024:
            self.conv1 = nn.Conv2d(3, 8, 4, 2, 1, bias=False)
            self.conv2 = nn.Conv2d(8, 16, 4, 2, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(16)
        else:
            self.conv1 = nn.Conv2d(3, 16, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        if hasattr(self, "conv2"):
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.lrelu(x)
        return x


class InputSmallBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.block1 = DownsamplingBlock(32, 64)
        self.block2 = DownsamplingBlock(64, 128)
        self.block3 = DownsamplingBlock(128, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class DownsamplingBlock(nn.Module):
    def __init__(self, fmap_in, fmap_out):
        super(DownsamplingBlock, self).__init__()
        self.conv = nn.Conv2d(fmap_in, fmap_out, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(fmap_out)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class DownsamplingCompBlock1(nn.Module):
    def __init__(self, fmap_in, fmap_out):
        super().__init__()
        self.conv1 = nn.Conv2d(fmap_in, fmap_out, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(fmap_out)
        self.conv2 = nn.Conv2d(fmap_out, fmap_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(fmap_out)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        return x


class DownsamplingCompBlock2(nn.Module):
    def __init__(self, fmap_in, fmap_out):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.conv = nn.Conv2d(fmap_in, fmap_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(fmap_out)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x


class DownsamplingCompBlock(nn.Module):
    def __init__(self, fmap_in, fmap_out):
        super().__init__()
        self.down1 = DownsamplingCompBlock1(fmap_in, fmap_out)
        self.down2 = DownsamplingCompBlock2(fmap_in, fmap_out)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x)
        return (x1 + x2)/2


class SimpleDecoderBlock(nn.Module):
    def __init__(self, fmap_in, fmap_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(fmap_in, fmap_out * 2, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(fmap_out * 2)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.glu(x)
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, fmap):
        super().__init__()
        self.fmaps_in = [fmap, 128, 64, 64]
        self.fmaps_out = [128, 64, 64, 32]

        self.blocks = nn.ModuleList()
        for fmap_in, fmap_out in zip(self.fmaps_in, self.fmaps_out):
            self.blocks.append(SimpleDecoderBlock(fmap_in, fmap_out))
        self.conv = nn.Conv2d(self.fmaps_out[-1], 3, 3, 1, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.conv(x)
        x = self.tanh(x)
        return x


class RFOutputBigBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = DownsamplingCompBlock(512, 1024)
        self.conv = nn.Conv2d(1024, 1, 4, bias=False)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class RFOutputSmallBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = DownsamplingBlock(256, 512)
        self.conv = nn.Conv2d(512, 1, 4, bias=False)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, resolution, sn=False, p=0., types=[]):
        super().__init__()
        assert resolution in [256, 512, 1024]
        self.p = p
        self.types = types

        self.input_big_block = InputBigBlock(resolution)
        self.sle_2_16 = SLEBlock(16, 128)
        self.down_4 = DownsamplingCompBlock(16, 32)
        self.sle_4_32 = SLEBlock(32, 256)
        self.down_8 = DownsamplingCompBlock(32, 64)
        self.sle_8_64 = SLEBlock(64, 512)
        self.down_16 = DownsamplingCompBlock(64, 128)
        self.down_32 = DownsamplingCompBlock(128, 256)
        self.down_64 = DownsamplingCompBlock(256, 512)

        self.decoder_w = SimpleDecoder(512)
        if resolution > 256:
            self.decoder_p = SimpleDecoder(256)
        self.decoder_s = SimpleDecoder(256)

        self.output_big_block = RFOutputBigBlock()

        self.input_small_block = InputSmallBlock()
        self.output_small_block = RFOutputSmallBlock()

        self.apply(initialize_weights)
        if sn:
            self.apply(spectral_normalization)

    def augment(self, x):
        if random.random() < self.p:
            x = DiffAugment(x, types=self.types)
        return x

    def random_crop(self, x):
        h = int(th.randint(0, x.size(-2), (1,)))
        w = int(th.randint(0, x.size(-1), (1,)))
        return x[..., h:h+x.size(-2)//2, w:w+x.size(-1)//2]

    def forward(self, x_big, x_small=None, aux=False):
        if x_small is None:
            x_small = tnf.interpolate(x_big, size=128)
        x_big = self.augment(x_big)
        x_small = self.augment(x_small)

        x_2 = self.input_big_block(x_big)
        x_4 = self.down_4(x_2)
        x_8 = self.down_8(x_4)

        x_16 = self.down_16(x_8)
        x_16 = self.sle_2_16(x_2, x_16)

        x_32 = self.down_32(x_16)
        x_32 = self.sle_4_32(x_4, x_32)

        x_64 = self.down_64(x_32)
        x_64 = self.sle_8_64(x_8, x_64)

        out_big = self.output_big_block(x_64)

        x_int = self.input_small_block(x_small)
        out_small = self.output_small_block(x_int)

        out = th.cat([out_big.flatten(1), out_small.flatten(1)], dim=-1)
        if not aux:
            return out
        return out, x_64, x_int, x_32


if __name__ == "__main__":
    dis = Discriminator(1024)

    x = th.randn(1, 3, 1024, 1024)
    out, x_hat_w, x_hat_s, x_hat_p = dis(x, aux=True)
