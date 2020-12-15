import hydra
import os
import pytorch_lightning as pl

from model import SLEModel
from utils import EveryNStepsCheckpoint
from tricks import EMACallback


@hydra.main(config_path="configs/main.yaml")
def main(cfg):

    pl.seed_everything(cfg.seed)

    model = SLEModel(cfg)
    ckpt_callback = EveryNStepsCheckpoint(os.getcwd(), 5000)
    ema_callback = EMACallback(0.995, 10, 15000)
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=os.getcwd(), name="")

    trainer = pl.Trainer(**cfg.trainer,
                         checkpoint_callback=False,
                         callbacks=[ckpt_callback, ema_callback],
                         logger=tb_logger)
    trainer.fit(model)


if __name__ == "__main__":
    main()
