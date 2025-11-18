import argparse
from configs.rmamamba_config import get_config as get_rmamamba_config
from configs.mambaunet_config import get_config as get_mambaunet_config
from configs.swinunet_config import get_config as get_swinunet_config
from models.RMAMamba import *
from models.SwinUnet import *
from models.VMUNet import *
from models.VMUNetV2 import *
from models.MambaUNet import *
from yacs.config import CfgNode as CN

def build_RMAMamba_T():
    args = CN()
    args.model = "RMAMamba_T"
    args.pretrained_weight_path = 'pretrained_pth/vssm_ckpt/vssm_tiny_0230_ckpt_epoch_262.pth'
    args.cfg = 'configs/vssm1/vssm_tiny_224.yaml'
    args.opts = None

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str,
    #                     default='RMAMamba_T')
    # parser.add_argument('--pretrained_weight_path', type=str,
    #                     default='pretrained_pth/vssm_ckpt/vssm_tiny_0230_ckpt_epoch_262.pth')
    # parser.add_argument('--cfg', type=str,
    #                     default='configs/vssm1/vssm_tiny_224.yaml')
    # parser.add_argument("--opts",
    #                     help="Modify config options by adding 'KEY VALUE' pairs. ",
    #                     default=None,
    #                     nargs='+')
    # opt = parser.parse_args()

    config = get_rmamamba_config(args)
    model = eval(args.model)(pretrained=args.pretrained_weight_path,
                       patch_size=config.MODEL.VSSM.PATCH_SIZE,
                       in_chans=config.MODEL.VSSM.IN_CHANS,
                       num_classes=config.MODEL.NUM_CLASSES,
                       depths=config.MODEL.VSSM.DEPTHS,
                       dims=config.MODEL.VSSM.EMBED_DIM,
                       # ===================
                       ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
                       ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
                       ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
                       ssm_dt_rank=(
                           "auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
                       ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
                       ssm_conv=config.MODEL.VSSM.SSM_CONV,
                       ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
                       ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
                       ssm_init=config.MODEL.VSSM.SSM_INIT,
                       forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
                       # ===================
                       mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
                       mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
                       mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
                       # ===================
                       drop_path_rate=config.MODEL.DROP_PATH_RATE,
                       patch_norm=config.MODEL.VSSM.PATCH_NORM,
                       norm_layer=config.MODEL.VSSM.NORM_LAYER,
                       downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
                       patchembed_version=config.MODEL.VSSM.PATCHEMBED,
                       gmlp=config.MODEL.VSSM.GMLP,
                       use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                       )

    return model


def build_RMAMamba_S():
    args = CN()
    args.model = "RMAMamba_S"
    args.pretrained_weight_path = 'pretrained_pth/vssm_ckpt/vssm_small_0229_ckpt_epoch_222.pth'
    args.cfg = 'configs/vssm1/vssm_small_224.yaml'
    args.opts = None

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str,
    #                     default='RMAMamba_S')
    # parser.add_argument('--pretrained_weight_path', type=str,
    #                     default='pretrained_pth/vssm_ckpt/vssm_small_0229_ckpt_epoch_222.pth')
    # parser.add_argument('--cfg', type=str,
    #                     default='configs/vssm1/vssm_small_224.yaml')
    # parser.add_argument("--opts",
    #                     help="Modify config options by adding 'KEY VALUE' pairs. ",
    #                     default=None,
    #                     nargs='+')
    # opt = parser.parse_args()

    config = get_rmamamba_config(args)
    model = eval(args.model)(pretrained=args.pretrained_weight_path,
                       patch_size=config.MODEL.VSSM.PATCH_SIZE,
                       in_chans=config.MODEL.VSSM.IN_CHANS,
                       num_classes=config.MODEL.NUM_CLASSES,
                       depths=config.MODEL.VSSM.DEPTHS,
                       dims=config.MODEL.VSSM.EMBED_DIM,
                       # ===================
                       ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
                       ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
                       ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
                       ssm_dt_rank=(
                           "auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
                       ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
                       ssm_conv=config.MODEL.VSSM.SSM_CONV,
                       ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
                       ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
                       ssm_init=config.MODEL.VSSM.SSM_INIT,
                       forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
                       # ===================
                       mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
                       mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
                       mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
                       # ===================
                       drop_path_rate=config.MODEL.DROP_PATH_RATE,
                       patch_norm=config.MODEL.VSSM.PATCH_NORM,
                       norm_layer=config.MODEL.VSSM.NORM_LAYER,
                       downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
                       patchembed_version=config.MODEL.VSSM.PATCHEMBED,
                       gmlp=config.MODEL.VSSM.GMLP,
                       use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                       )

    return model


def build_SwinUnet_T():
    args = CN()
    args.model = "SwinUnet"
    args.num_classes = 1
    args.pretrained_weight_path = 'pretrained_pth/Swin/swin_tiny_patch4_window7_224.pth'
    args.cfg = 'configs/swin/swin_tiny_patch4_window7_224_lite.yaml'
    args.opts = None
    args.use_checkpoint = True

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name', type=str,
    #                     default='SwinUnet')
    # parser.add_argument('--num_classes', type=int,
    #                     default=1, help='output channel of network')
    # parser.add_argument('--pretrained_weight_path', type=str,
    #                     default='pretrained_pth/Swin/swin_tiny_patch4_window7_224.pth')
    # parser.add_argument('--cfg', type=str,
    #                     default='configs/swin/swin_tiny_patch4_window7_224_lite.yaml')
    # parser.add_argument(
    #     "--opts",
    #     help="Modify config options by adding 'KEY VALUE' pairs. ",
    #     default=None,
    #     nargs='+',
    # )
    # parser.add_argument('--use-checkpoint', action='store_true',
    #                     help="whether to use gradient checkpointing to save memory")
    # opt = parser.parse_args()

    config = get_swinunet_config(args)
    model = eval(args.model)(config, pretrained=args.pretrained_weight_path, num_classes=args.num_classes)
    return model


def build_VMUnet():
    model_cfg = {
        'name': 'VMUnet',
        'num_classes': 1,
        'input_channels': 3,
        # ----- VM-UNet ----- #
        'depths': [2,2,2,2],
        'depths_decoder': [2,2,2,1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': 'pretrained_pth/vssm_ckpt/vmamba_small_e238_ema.pth',
    }
    model = eval(model_cfg['name'])(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
    return model


def build_VMUnetV2():
    model_cfg = {
        'name': 'VMUNetV2',
        'num_classes': 1,
        'input_channels': 3,
        # ----- VM-UNet-V2 -----2 9 27  small #
        'depths': [2, 2, 9, 2],
        'depths_decoder': [2, 2, 2, 1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': 'pretrained_pth/vssm_ckpt/vmamba_small_e238_ema.pth',
        'deep_supervision': True,
    }
    model = eval(model_cfg['name'])(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
        deep_supervision=model_cfg['deep_supervision'],
    )
    return model


def build_MambaUnet():
    args = CN()
    args.model = "MambaUnet"
    args.num_classes = 1
    args.cfg = 'configs/vssm1/vmamba_tiny.yaml'
    args.pretrained_weight_path = 'pretrained_pth/vssm_ckpt/vmamba_tiny_e292.pth'
    args.opts = None

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str,
    #                     default='MambaUnet', help='model_name')
    # parser.add_argument('--num_classes', type=int,
    #                     default=1, help='output channel of network')
    # parser.add_argument(
    #     '--cfg', type=str, default="configs/vssm1/vmamba_tiny.yaml", help='path to config file', )
    # parser.add_argument('--pretrained_weight_path', type=str,
    #                     default='pretrained_pth/vssm_ckpt/vmamba_tiny_e292.pth')
    # parser.add_argument(
    #     "--opts",
    #     help="Modify config options by adding 'KEY VALUE' pairs. ",
    #     default=None,
    #     nargs='+',
    # )
    # args = parser.parse_args()

    config = get_mambaunet_config(args)
    model = MambaUnet(config, pretrained=args.pretrained_weight_path, num_classes=args.num_classes).cuda()
    return model