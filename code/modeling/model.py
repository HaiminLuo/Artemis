import torch
import torch.nn as nn
from .UNet import UNet, LightUNet
from .tree_render import TreeRenderer

import functools
import torch.nn.utils.spectral_norm as spectral_norm
from torch.cuda.amp import autocast
from utils import generate_transformation_matrices

import time

class GeneralModel(nn.Module):
    def __init__(self, cfg, flut_size=0, use_render_net=True, bone_feature_dim=0, texture_feature_dim=0, sh_dim=9):
        super(GeneralModel, self).__init__()
        self.use_render_net = use_render_net
        self.bone_feature_dim = bone_feature_dim

        if not cfg.MODEL.USE_MOTION:
            self.bone_feature_dim = 0

        if flut_size != 0:
            self.tree_renderer = TreeRenderer(flut_size, background_brightness=0., random_init=cfg.MODEL.RANDOM_INI, feature_dim=texture_feature_dim * sh_dim + 1)


        if self.use_render_net:
            if not cfg.MODEL.USE_LIGHT_RENDERER:
                self.render_net = UNet(rgb_feat_channels=texture_feature_dim + self.bone_feature_dim,
                                        alpha_feat_channels=1, n_classes1=3, n_classes2=1)
            else:
                print('Light')
                self.render_net = LightUNet(rgb_feat_channels=texture_feature_dim + self.bone_feature_dim,
                                            alpha_feat_channels=1, n_classes1=3, n_classes2=1)

    def forward(self, rays, tree=None, joint_features=None, skinning_weights=None, joint_index=None, coords=None, transformation_matrices=None):
        batch_size, h, w = rays.size(0), rays.size(1), rays.size(2)
        joint_num, joint_feature_dim = (joint_features.size(1), joint_features.size(2)) if joint_features is not None else (0, 0)
        if isinstance(tree, list):
            assert len(tree) == batch_size
        else:
            tree = [tree]

        if isinstance(transformation_matrices, list):
            assert len(transformation_matrices) == batch_size
        else:
            transformation_matrices = [transformation_matrices]

        # print(rays.shape)

        rgbas, results = [], []
        rgb_features, alpha_features, motion_features = [], [], []
        features_in = []
        vrts = []

        joint_features = joint_features.reshape(-1, joint_feature_dim).type_as(rays) if joint_features is not None else None
        # print(joint_features.max())
        if self.use_bone_net and joint_features is not None:
            joint_features = self.bone_net(joint_features)

        if joint_features is not None:
            joint_features = joint_features.reshape(batch_size, joint_num, -1)

        s = time.time()
        for i in range(batch_size):
            ray = rays[i:i+1, ...].reshape(-1, rays.size(3))

            torch.cuda.synchronize()
            s = time.time()
            matrices = None
            # print(batch_size, i, transformation_matrices[i].max())
            if transformation_matrices[i] is not None:
                matrices = generate_transformation_matrices(matrices=transformation_matrices[i], skinning_weights=skinning_weights, joint_index=joint_index)
            features = self.tree_renderer(tree[i], ray, matrices)
            features = features.reshape(1, h, w, -1).permute(0, 3, 1, 2)

            motion_feature = None
            if joint_features is not None:
                with autocast(enabled=False):
                    motion_feature = self.tree_renderer.motion_feature_render(tree[i], joint_features[i].float(), skinning_weights,
                                                                              joint_index,
                                                                              ray)
                    motion_feature = motion_feature.reshape(1, h, w, -1).permute(0, 3, 1, 2)
            torch.cuda.synchronize()
            features_in.append(features)
            motion_features.append(motion_feature)

            if coords is not None:
                vrts.append(self.tree_renderer.voxel_regularization(tree[i], coords[i]))

        features_in = torch.cat(features_in, dim=0)
        motion_features = None if motion_features[0] is None else torch.cat(motion_features, dim=0)

        rgbas = torch.cat([features_in[:, :3, ...], features_in[:, -1:, ...]], dim=1)
        torch.cuda.synchronize()
        s = time.time()
        if self.use_render_net:
            results = self.render_net(torch.cat([features[:, :-1, ...], motion_feature], dim=1), features_in[:, -1:, ...])
        else:
            results = rgbas

        if coords is None:
            return rgbas, results, motion_features
        else:
            return rgbas, results, sum(vrts) / float(len(vrts)), motion_features

    def render_volume_feature(self, rays, tree=None, joint_features=None, skinning_weights=None, joint_index=None):
        batch_size, h, w = rays.size(0), rays.size(1), rays.size(2)
        rays = rays.reshape(-1, rays.size(3))

        features = self.tree_renderer(tree, rays)
        features = features.reshape(batch_size, h, w, -1).permute(0, 3, 1, 2)
        rgb_feature_map = features[:, :-1, ...]
        alpha = features[:, -1:, ...]
        rgb = features[:, :3, ...]
        rgba = torch.cat([rgb, alpha], dim=1)

        motion_feature = None
        if joint_features is not None:
            motion_feature = self.tree_renderer.motion_feature_render(tree, joint_features, skinning_weights, joint_index, rays) # joint_features, skinning_weights, joint_index, rays

        return rgba, features, motion_feature



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ] if norm_layer is not 'spectral' else [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ] if norm_layer is not 'spectral' else [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)