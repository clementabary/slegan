import torch as th
import torch.nn as nn
import torch.nn.functional as tnf
import torch.autograd as ag
import torchvision
import lpips

from random import random
from einops import rearrange
from math import floor

from networks.discriminator import Discriminator
from networks.generator import Generator


class SelfSupDisReconLoss(nn.Module):
    def __init__(self, net, type="mae"):
        super(SelfSupDisReconLoss, self).__init__()
        assert type in ["mse", "mae", "perceptual"]
        if type == "mse":
            self.loss = nn.MSELoss(reduction="mean")
        elif type == "mae":
            self.loss = nn.L1Loss(reduction="mean")
        # TODO: Check device casting
        elif type == "perceptual":
            self.loss = lpips.LPIPS(net='vgg', verbose=False)
        self.net = net

    def quad_crop(self, f, q):
        return rearrange(f, 'b c (m h) (n w) -> (m n) b c h w', m=2, n=2)[q]

    def forward(self, x, f_w, f_s, f_p):
        x_w = self.net.decoder_w(f_w)
        x_s = self.net.decoder_s(f_s)
        loss_w = self.loss(x_w,tnf.interpolate(x, size=x_w.shape[-2:]))
        loss_s = self.loss(x_s,tnf.interpolate(x, size=x_s.shape[-2:]))

        loss_p = 0
        if f_p.shape[-1] >= 16:
            rand_q = floor(random() * 4)
            f_p = self.quad_crop(f_p, rand_q)
            x_p = self.net.decoder_p(f_p)
            x = self.quad_crop(x, rand_q)
            loss_p = self.loss(x_p,tnf.interpolate(x, size=x_p.shape[-2:]))

        # return loss_w + loss_s + loss_p
        return (loss_w + loss_s + loss_p).mean()


class AdversarialLoss(nn.Module):
    def __init__(self, type, true_label=1., fake_label=0., smooth=False):
        super(AdversarialLoss, self).__init__()
        self.type = type
        # used for JS loss
        self.true_label = th.tensor(true_label)
        self.fake_label = th.tensor(fake_label)
        self.smooth = smooth  # used for Hinge loss
        if type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif type == "lsgan":
            self.loss = nn.MSELoss()
        elif type in ["wgan", "wgangp", "hinge"]:
            self.loss = None
        else:
            raise ValueError(f"{type} not implemented.")

    def forward(self, x, bool, is_disc=False):
        if self.type in ["vanilla", "lsgan"]:
            if bool:
                y = self.true_label.expand_as(x).type_as(x)
            else:
                y = self.fake_label.expand_as(x).type_as(x)
            return 0.5 * self.loss(x, y)
        elif self.type in ["wgan", "wgangp"]:
            return - x.mean() if bool else x.mean()
        elif self.type == "hinge":
            if is_disc:
                l = 0.2 * th.rand_like(x) + 0.8 if self.smooth else 1.
                return tnf.relu(l - x).mean() if bool else tnf.relu(l + x).mean()
            else:
                return - x.mean()


@th.enable_grad()
def compute_gp(critic, real, fake):
    alpha = th.rand((real.size(0), 1, 1, 1)).type_as(real)
    alpha = alpha.expand(real.size())
    alpha.requires_grad_(True)
    interpol = alpha * real + (1 - alpha) * fake
    interpol_critic = critic(interpol)
    gradients = ag.grad(outputs=interpol_critic, inputs=interpol,
                        grad_outputs=th.ones_like(interpol_critic),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    # gradients_norm = th.sqrt(th.sum(gradients ** 2, dim=1) + 1e-12)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
