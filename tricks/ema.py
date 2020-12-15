import torch as th
import pytorch_lightning as pl


class ExponentialMovingAverage:
    # Got from https://github.com/fadel/pytorch_ema
    def __init__(self, parameters, decay, use_num_updates=True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with th.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)


class EMACallback(pl.Callback):
    def __init__(self, decay=0.995, every_n_step=10, not_before_step=15000):
        self.decay = decay
        self.every_n_step = every_n_step
        self.not_before_step = not_before_step

    def on_train_start(self, trainer, pl_module):
        if hasattr(pl_module, "gen_ema"):
            self.ema = ExponentialMovingAverage(pl_module.gen.parameters(),
                                                self.decay)

    def on_batch_end(self, trainer, pl_module):
        if hasattr(pl_module, "gen_ema"):
            gs = trainer.global_step
            if gs % self.every_n_step == 0 and gs > self.not_before_step:
                self.ema.update(pl_module.gen.parameters())

    def on_train_end(self, trainer, pl_module):
        if hasattr(pl_module, "gen_ema"):
            self.ema.copy_to(pl_module.gen_ema.parameters())
