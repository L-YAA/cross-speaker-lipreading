mport os
import hydra
import logging
# ========== 新增：显存优化 ==========
import torch
torch.backends.cudnn.benchmark = True  # 加速卷积计算，减少显存碎片
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32，降低显存占用
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # 减少显存碎片
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 定位具体OOM的行（可选）
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy
from avg_ckpts import ensemble
from datamodule.data_module import DataModule
from lightning import ModelModule
import torch.distributed as dist

@hydra.main(version_base="1.3", config_path="configs", config_name="config_train")
def main(cfg):
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()

    # ========== 新增ly：打印ps/pd/alpha全局配置值 ==========
    ps_val = cfg.model.visual_backbone.ps
    pd_val = cfg.model.visual_backbone.pd
    alpha_val = cfg.model.visual_backbone.alpha
    print(f"\n【全局配置】训练启动 - ps={ps_val}, pd={pd_val}, alpha={alpha_val}\n")

    checkpoint = ModelCheckpoint(
        monitor=cfg.monitor,
        mode=cfg.mode,  # 越大越好
        dirpath=os.path.join(cfg.exp_dir, cfg.exp_name) if cfg.exp_dir else None,
        save_last=True,
        filename=cfg.filename,
        save_top_k=cfg.save_top_k,
        save_on_train_epoch_end=False
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    modelmodule = ModelModule(cfg)
    print("[model] has beam_search:", hasattr(modelmodule, "beam_search"))
    print("[model] beam_search keys:", sum(k.startswith("beam_search.") for k in modelmodule.state_dict().keys()))
    datamodule = DataModule(cfg)

    trainer = Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        # ✅ 关键修复：允许某些参数在某些 step 不参与反传
        strategy=DDPStrategy(find_unused_parameters=False),
    )

    # trainer.fit(model=modelmodule, datamodule=datamodule)
    ckpt_path = None
    try:
        ckpt_path = cfg.get("ckpt_path", None)  # OmegaConf DictConfig 支持 get
    except Exception:
        pass
    print(f"[resume] ckpt_path={ckpt_path}")

    trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=ckpt_path)
    ensemble(cfg)


if __name__ == "__main__":
    main()