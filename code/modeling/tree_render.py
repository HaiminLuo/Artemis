import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

import svox_t
import collections
Sh_Rays = collections.namedtuple('Sh_Rays', 'origins dirs viewdirs')


class TreeRenderer(nn.Module):

    def __init__(self, flut_size, background_brightness=0., step_size=1e-3, random_init=False, feature_dim=0):
        super(TreeRenderer, self).__init__()
        self.background_brightness = background_brightness
        self.step_size = step_size
        print(flut_size)

        features = torch.empty((flut_size, feature_dim))
        nn.init.normal_(features)
        self.register_parameter("features", nn.Parameter(features))

    def forward(self, tree, rays, transformation_matrices=None, fast_rendering=False):
        t = tree.to(rays.device)

        r = svox_t.VolumeRenderer(t, background_brightness=self.background_brightness, step_size=self.step_size)
        dirs = rays[..., :3].contiguous()
        origins = rays[..., 3:].contiguous()

        sh_rays = Sh_Rays(origins, dirs, dirs)

        res = r(self.features, sh_rays, transformation_matrices=transformation_matrices, fast=fast_rendering)

        return res

    def motion_render(self, tree, rays):
        t = tree.to(rays.device)
        dirs = rays[..., :3].contiguous()
        origins = rays[..., 3:].contiguous()
        sh_rays = Sh_Rays(origins, dirs, dirs)
        r = svox_t.VolumeRenderer(t, background_brightness=self.background_brightness, step_size=self.step_size)
        motion_feature, depth, hit_point, data_idx = r.motion_render(self.features, sh_rays)

        return motion_feature, depth, hit_point, data_idx

    def motion_feature_render(self, tree, joint_features, skinning_weights, joint_index, rays):
        t = tree.to(rays.device)
        dirs = rays[..., :3].contiguous()
        origins = rays[..., 3:].contiguous()
        sh_rays = Sh_Rays(origins, dirs, dirs)
        r = svox_t.VolumeRenderer(t, background_brightness=self.background_brightness, step_size=self.step_size)
        motion_feature= r.motion_feature_render(self.features, joint_features, skinning_weights, joint_index, sh_rays)

        return motion_feature

    def voxel_regularization(self, tree, coordinate):
        tree = tree.to(coordinate.device)
        query_features = tree(self.features, coordinate, want_node_ids=False)
        vrt = nn.L1Loss()(query_features[..., :-1].detach(), self.features[..., :-1])
        # vrt = nn.MSELoss()(query_features.detach(), self.features)
        return vrt

    @staticmethod
    def MotionRender(tree, features, rays):
        t = tree.to(rays.device)
        dirs = rays[..., :3].contiguous()
        origins = rays[..., 3:].contiguous()
        sh_rays = Sh_Rays(origins, dirs, dirs)
        r = svox_t.VolumeRenderer(t, background_brightness=0., step_size=1e-3)
        motion_feature, depth, hit_point, data_idx = r.motion_render(features, sh_rays)

        return motion_feature, depth, hit_point, data_idx

    @staticmethod
    def MotionFeatureRender(tree, features, joint_features, skinning_weights, joint_index, rays):
        t = tree.to(rays.device)
        dirs = rays[..., :3].contiguous()
        origins = rays[..., 3:].contiguous()
        sh_rays = Sh_Rays(origins, dirs, dirs)
        r = svox_t.VolumeRenderer(t, background_brightness=0., step_size=1e-3)
        motion_feature = r.motion_feature_render(features, joint_features, skinning_weights, joint_index, sh_rays)

        return motion_feature