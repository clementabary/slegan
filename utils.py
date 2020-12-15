import torch as th
import torchvision as tv
import torch.nn as nn

import pytorch_lightning as pl
import os


class EveryNStepsCheckpoint(pl.Callback):
    def __init__(self, path, every_n_step):
        self.path = os.path.join(path, 'checkpoints')
        self.every_n_step = every_n_step
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def on_batch_end(self, trainer, pl_module):
        gs = trainer.global_step
        if gs % self.every_n_step == 0 and gs != 0:
            ckpt_path = f"{self.path}/model_{gs}.ckpt"
            trainer.save_checkpoint(ckpt_path)


class UnNormalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class ToImage():
    def __init__(self):
        self.unnorm = UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.topil = tv.transforms.ToPILImage()

    def __call__(self, x: th.Tensor, norm=True):
        x = x.clone().detach()
        if len(x.size()) == 4 and x.size()[0] == 1:
            x = x[0]
        elif len(x.size()) != 3:
            raise ValueError('Wrongly shaped tensor.')
        if norm:
            x = self.unnorm(x)
        return self.topil(x)


def initialize_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        # nn.init.kaiming_normal_(m.weight)
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm2d)):
        # m.weight.data.fill_(1)
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)


def spectral_normalization(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        m = nn.utils.spectral_norm(m)


def nparams(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad_])
