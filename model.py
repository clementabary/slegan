import pytorch_lightning as pl
import omegaconf as om
import hydra.utils as hu
import torch as th

from losses import compute_gp


class SLEModel(pl.LightningModule):
    def __init__(self, hparams: om.DictConfig):
        super().__init__()

        if not isinstance(hparams, om.DictConfig):
            hparams = om.DictConfig(hparams)
        self.hparams = om.OmegaConf.to_container(hparams, resolve=True)

        # Instantiate datasets (Hydra compat)
        self.dataset = hu.instantiate(hparams.dataset)

        # Instantiate network modules
        self.gen = hu.instantiate(hparams.gen)
        self.dis = hu.instantiate(hparams.dis)
        self.type = hparams.var.type
        if hparams.var.ema:
            self.gen_ema = hu.instantiate(hparams.gen)

        # Instantiate optimizers & schedulers
        self.gen_optim = hu.instantiate(hparams.gen_opt, self.gen.parameters())
        self.dis_optim = hu.instantiate(hparams.dis_opt, self.dis.parameters())

        # Instantiate losses
        self.lambda_recon = hparams.var.lambda_recon
        self.lambda_gp = hparams.var.lambda_gp
        self.recon_loss = hu.instantiate(hparams.recon_loss, self.dis)
        self.adv_loss = hu.instantiate(hparams.adv_loss)

    def forward(self, x=None):
        self.gen.eval()
        if x is None:
            x = th.randn(1, self.gen.latent_dim, 1, 1)
        with th.no_grad():
            return self.gen(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch

        if optimizer_idx == 0:
            z = th.randn(x.size(0), self.gen.latent_dim, 1, 1).type_as(x)
            x_hat_b, x_hat_s = self.gen(z)
            adv_loss = self.adv_loss(self.dis(x_hat_b, x_hat_s), True, False)
            gen_loss = adv_loss
            logs = {
                    "gen_loss": gen_loss,
                    "g_adv_loss": adv_loss
                    }
            return {"loss": gen_loss, "log": logs}

        if optimizer_idx == 1:
            z = th.randn(x.size(0), self.gen.latent_dim, 1, 1).type_as(x)
            x_hat_b, x_hat_s = self.gen(z)
            x_hat_b, x_hat_s = x_hat_b.detach(), x_hat_s.detach()
            logits, f_b, f_s, f_p = self.dis(x, aux=True)
            adv_loss = self.adv_loss(self.dis(x_hat_b, x_hat_s), False, True) \
                + self.adv_loss(logits, True, True)
            recon_loss = self.recon_loss(x, f_b, f_s, f_p)
            dis_loss = adv_loss + self.lambda_recon * recon_loss
            if self.type == "wgangp":
                # TODO: adapt to multiple params !
                gp = compute_gp(self.dis, x, x_hat_b)
                dis_loss += self.lambda_gp * gp
            logs = {
                    "dis_loss": dis_loss,
                    "recon_loss": recon_loss,
                    "d_adv_loss": adv_loss
                    }
            if self.type == "wgangp":
                logs["gp"] = gp
            return {"loss": dis_loss, "log": logs}

    def train_dataloader(self):
        return th.utils.data.DataLoader(self.dataset,
                                        **self.hparams.loader["params"])

    def configure_optimizers(self):
        return self.gen_optim, self.dis_optim

    @staticmethod
    def _classname(obj, lower=True):
        if hasattr(obj, '__name__'):
            name = obj.__name__
        else:
            name = obj.__class__.__name__
        return name.lower() if lower else name
