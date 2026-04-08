import os
import hydra
import logging

import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from avg_ckpts import ensemble
from datamodule.data_module import DataModule
from lightning import ModelModule


@hydra.main(version_base="1.3", config_path="configs", config_name="config_train")
def main(cfg):
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()

    #模型权重
    checkpoint = ModelCheckpoint(
        monitor="monitoring_step",
        mode="max", #越大越好
        dirpath=os.path.join(cfg.exp_dir, cfg.exp_name) if cfg.exp_dir else None, #checkpoints和logs保存目录
        save_last=True, #保存最后的权重
        filename="{epoch}", #保存文件名格式
        save_top_k=5, #保存最好的5个checkpoints
    )
    #记录每步学习率变化，并绘制到tensorboard
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    #模型
    # Set modules and trainer
    modelmodule = ModelModule(cfg)
    #数据
    datamodule = DataModule(cfg)
    #训练超参调整及日志
    trainer = Trainer(
        **cfg.trainer,
        #logger=WandbLogger(name=cfg.exp_name, project="auto_avsr"),
        callbacks=callbacks,
        strategy=DDPPlugin(find_unused_parameters=False)
    )
    #使用数据训练模型
    trainer.fit(model=modelmodule, datamodule=datamodule)
    trainer.test(model=modelmodule, datamodule=datamodule)
    ensemble(cfg)


if __name__ == "__main__":
    main()
