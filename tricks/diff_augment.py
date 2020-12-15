import random
import torch as th
import torch.nn.functional as tnf


def DiffAugment(x, types=[]):
    for p in types:
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous()


def rand_brightness(x):
    x = x + (th.rand(x.size(0), 1, 1, 1).type_as(x) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (th.rand(x.size(0), 1, 1, 1).type_as(x) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (th.rand(x.size(0), 1, 1, 1).type_as(x) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = th.randint(-shift_x, shift_x + 1,
                               size=[x.size(0), 1, 1], device=x.device)
    translation_y = th.randint(-shift_y, shift_y + 1,
                               size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = th.meshgrid(
        th.arange(x.size(0), dtype=th.long, device=x.device),
        th.arange(x.size(2), dtype=th.long, device=x.device),
        th.arange(x.size(3), dtype=th.long, device=x.device),
    )
    grid_x = th.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = th.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = tnf.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[
        grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = th.randint(0, x.size(
        2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = th.randint(0, x.size(
        3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = th.meshgrid(
        th.arange(x.size(0), dtype=th.long, device=x.device),
        th.arange(cutout_size[0], dtype=th.long, device=x.device),
        th.arange(cutout_size[1], dtype=th.long, device=x.device),
    )
    grid_x = th.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = th.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = th.ones(x.size(0), x.size(2), x.size(3)).type_as(x)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


def rand_hflip(x, prob=0.5):
    if random.random() < prob:
        return x
    return th.flip(x, dims=(3,))


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
    'flip': [rand_hflip]
}
