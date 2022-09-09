from .model import GeneralModel, NLayerDiscriminator
from .tree_render import TreeRenderer
import torch
import os

def build_model(cfg, flut_size=0, bone_feature_dim=0):
    if not cfg.MODEL.USE_MOTION:
        bone_feature_dim = 0
    return GeneralModel(cfg, flut_size=flut_size, use_render_net=cfg.MODEL.USE_RENDER_NET, bone_feature_dim=bone_feature_dim, texture_feature_dim=cfg.MODEL.SH_FEAT_DIM)


def build_discriminator(cfg):
    return NLayerDiscriminator(input_nc=input_nc, ndf=32, n_layers=5, norm_layer='spectral')