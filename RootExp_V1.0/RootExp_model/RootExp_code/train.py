import os
import sys
import warnings

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from loader import isic_loader
from engine import train_one_epoch, val_one_epoch, test_one_epoch
from utils import get_logger, log_config_info, set_seed, get_optimizer, get_scheduler

from models.rootbin import RootBinBlockMambaEAMASPPNet
from configs.config_setting import setting_config

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(config):
    print("#----------Creating logger----------#")
    sys.path.append(config.work_dir + "/")
    log_dir = os.path.join(config.work_dir, "log")
    checkpoint_dir = os.path.join(config.work_dir, "checkpoints")
    resume_model = os.path.join(checkpoint_dir, "latest.pth")
    outputs = os.path.join(config.work_dir, "outputs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    global logger
    logger = get_logger("train", log_dir)
    log_config_info(config, logger)

    print("#----------GPU init----------#")
    set_seed(config.seed)
    gpu_ids = [0]
    torch.cuda.empty_cache()

    print("#----------Preparing dataset----------#")
    train_dataset = isic_loader(path_Data=config.data_path, train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    val_dataset = isic_loader(path_Data=config.data_path, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=True,
    )

    print("#----------Preparing Model: RootBin_Block_Mamba_EAM_ASPP----------#")
    model_cfg = config.model_config
    model = RootBinBlockMambaEAMASPPNet(
        num_classes=model_cfg["num_classes"],
        input_channels=model_cfg["input_channels"],
        stage_name=model_cfg.get("stage_name", "stage0_unet_scblock_hvss_eam_aspp"),
        split_att=model_cfg.get("split_att", "fc"),
        encoder_channels=model_cfg.get("encoder_channels"),
        use_eam=model_cfg.get("use_eam", True),
        use_aspp=model_cfg.get("use_aspp", True),
        use_hvss=model_cfg.get("use_hvss", True),
        use_bridge=model_cfg.get("use_bridge", True),
        hvss_depth=model_cfg.get("hvss_depth", 1),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
        hvss_order=model_cfg.get("hvss_order", 2),
        hvss_d_state=model_cfg.get("hvss_d_state", 6),
        mlp_ratio=model_cfg.get("mlp_ratio", 1.5),
        use_skip_enhance=model_cfg.get("use_skip_enhance", True),
        skip_fusion_dim=model_cfg.get("skip_fusion_dim", 64),
        aspp_dilations=tuple(model_cfg.get("aspp_dilations", (1, 6, 12, 18))),
    )
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    print("#----------Preparing loss, opt, sch and amp----------#")
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print("#----------Set other params----------#")
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    patience_counter = 0
    early_stop_patience = getattr(config, "early_stop_patience", 200)

    if os.path.exists(resume_model):
        print("#----------Resume Model and Other params----------#")
        checkpoint = torch.load(resume_model, map_location=torch.device("cpu"))
        model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception:
            pass
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception:
            pass
        saved_epoch = checkpoint["epoch"]
        start_epoch += saved_epoch
        min_loss = checkpoint.get("min_loss", min_loss)
        min_epoch = checkpoint.get("min_epoch", min_epoch)
        patience_counter = checkpoint.get("patience_counter", 0)
        loss = checkpoint.get("loss", float("nan"))
        log_info = (
            f"resuming model from {resume_model}. resume_epoch: {saved_epoch}, "
            f"min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}"
        )
        logger.info(log_info)

    print("#----------Training----------#")
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, logger, config, scaler=scaler)
        loss = val_one_epoch(val_loader, model, criterion, epoch, logger, config)

        if loss < min_loss:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, "best.pth"))
            min_loss = loss
            min_epoch = epoch
            patience_counter = 0
        elif early_stop_patience > 0:
            patience_counter += 1

        torch.save(
            {
                "epoch": epoch,
                "min_loss": min_loss,
                "min_epoch": min_epoch,
                "loss": loss,
                "patience_counter": patience_counter,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(checkpoint_dir, "latest.pth"),
        )

        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            logger.info(
                f"Early stopping at epoch {epoch}: no improvement for {early_stop_patience} epochs "
                f"(best epoch={min_epoch}, best loss={min_loss:.4f})"
            )
            print(f"Early stopping at epoch {epoch}")
            break

    if os.path.exists(os.path.join(checkpoint_dir, "best.pth")):
        print("#----------Testing on val set----------#")
        best_weight = torch.load(config.work_dir + "checkpoints/best.pth", map_location=torch.device("cpu"))
        model.module.load_state_dict(best_weight, strict=False)
        _ = test_one_epoch(val_loader, model, criterion, logger, config)
        os.rename(
            os.path.join(checkpoint_dir, "best.pth"),
            os.path.join(checkpoint_dir, f"best-epoch{min_epoch}-loss{min_loss:.4f}.pth"),
        )


if __name__ == "__main__":
    main(setting_config)
