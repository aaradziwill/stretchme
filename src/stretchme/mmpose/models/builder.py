from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn import build_model_from_cfg
from mmcv.utils import Registry

MODELS = Registry("models", build_func=build_model_from_cfg, parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
POSENETS = MODELS
MESH_MODELS = MODELS


def build_posenet(cfg):
    """Build posenet."""
    return POSENETS.build(cfg)
