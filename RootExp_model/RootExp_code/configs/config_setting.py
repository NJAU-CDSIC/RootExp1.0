from datetime import datetime
from utils import *


class setting_config:
    """
    RootBin_Block_Mamba + ASPP then EAM fusion at bottleneck before SC_Att_Bridge.
    """

    network = "RootBin_Block_Mamba_EAM_ASPP"
    model_config = {
        "num_classes": 1,
        "input_channels": 3,
        "stage_name": "stage0_unet_scblock_hvss_eam_aspp",
        "split_att": "fc",
        "encoder_channels": [40, 80, 160, 320, 640],
        "use_eam": True,
        "use_aspp": True,
        "use_skip_enhance": False,
        "skip_fusion_dim": 64,
        "hvss_depth": 1,
        "drop_path_rate": 0.05,
        "hvss_order": 2,
        "hvss_d_state": 8,
        "mlp_ratio": 2.0,
    }

    datasets = "ISIC2017"
    if datasets == "ISIC2017":
        data_path = "./dataset/"
    else:
        raise Exception("datasets in not right!")

    criterion = BceDiceLoss()

    num_classes = 1
    input_size_h = 224
    input_size_w = 224
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 4
    seed = 42
    world_size = None
    rank = None
    amp = False
    batch_size = 8
    epochs = 2000
    early_stop_patience = 800

    work_dir = "results/" + network + "_" + datasets + "_" + datetime.now().strftime("%A_%d_%B_%Y_%Hh_%Mm_%Ss") + "/"

    print_interval = 20
    val_interval = 30
    save_interval = 100
    threshold = 0.5

    opt = "AdamW"
    assert opt in ["Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "ASGD", "RMSprop", "Rprop", "SGD"], "Unsupported optimizer!"
    if opt == "AdamW":
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-2
        amsgrad = False

    sch = "WP_CosineLR"
    warm_up_epochs = 20
    if sch == "CosineAnnealingLR":
        T_max = 50
        eta_min = 0.00001
        last_epoch = -1
