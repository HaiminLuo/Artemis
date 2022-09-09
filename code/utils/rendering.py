from utils import ray_sampling
import collections
import time
import torch
import svox_t
from utils import generate_transformation_matrices
import numpy as np
import math

Sh_Rays = collections.namedtuple('Sh_Rays', 'origins dirs viewdirs')


def render_image(cfg, model, K, T, img_size=(450, 800), tree=None, matrices=None, joint_features=None,
                 bg=None, skinning_weights=None, joint_index=None):
    torch.cuda.synchronize()
    s = time.time()
    h, w = img_size[0], img_size[1]
    rays, _, _ = ray_sampling(K.unsqueeze(0).cuda(), T.unsqueeze(0).cuda(), img_size)

    with torch.no_grad():
        joint_features = None if not cfg.MODEL.USE_MOTION else joint_features
        matrices = generate_transformation_matrices(matrices=matrices, skinning_weights=skinning_weights,
                                                    joint_index=joint_index)

        with torch.cuda.amp.autocast(enabled=False):
            features = model.tree_renderer(tree, rays, matrices).reshape(1, h, w, -1).permute(0, 3, 1, 2)

        if cfg.MODEL.USE_MOTION:
            motion_feature = model.tree_renderer.motion_feature_render(tree, joint_features, skinning_weights,
                                                                       joint_index,
                                                                       rays)


            motion_feature = motion_feature.reshape(1, h, w, -1).permute(0, 3, 1, 2)
        else:
            motion_feature = features[:, :9, ...]

        with torch.cuda.amp.autocast(enabled=True):
            features_in = features[:, :-1, ...]
            if cfg.MODEL.USE_MOTION:
                features_in = torch.cat([features[:, :-1, ...], motion_feature], dim=1)
            rgba_out = model.render_net(features_in, features[:, -1:, ...])
            
        rgba_volume = torch.cat([features[:, :3, ...], features[:, -1:, ...]], dim=1)

        rgb = rgba_out[0, :-1, ...]
        alpha = rgba_out[0, -1:, ...]
        img_volume = rgba_volume[0, :3, ...].permute(1, 2, 0)

        if model.use_render_net:
            rgb = torch.nn.Hardtanh()(rgb)
            rgb = (rgb + 1) / 2

            alpha = torch.nn.Hardtanh()(alpha)
            alpha = (alpha + 1) / 2
            alpha = torch.clamp(alpha, min=0, max=1.)

    if bg is not None:
        if bg.max() > 1:
            bg = bg / 255
        comp_img = rgb * alpha + (1 - alpha) * bg
    else:
        comp_img = rgb * alpha + (1 - alpha)

    img_unet = comp_img.permute(1, 2, 0).float().cpu().numpy()

    return img_unet, alpha.squeeze().float().detach().cpu().numpy(), img_volume.float().detach().cpu().numpy()


def build_octree(coordinates, radius=None, center=None, skeleton=None, data_dim=91,
                 max_depth=10, data_format='SH9'):
    coordinates = coordinates.contiguous().cuda().detach()

    maxs_ = torch.max(coordinates, dim=0).values
    mins_ = torch.min(coordinates, dim=0).values
    radius_ = (maxs_ - mins_) / 2
    center_ = mins_ + radius_

    if radius is None:
        radius = radius_

    if center is None:
        center = center_
        
    t = svox_t.N3Tree(N=2,
                      data_dim=data_dim,
                      depth_limit=max_depth,
                      init_reserve=30000,
                      init_refine=0,
                      #                 geom_resize_fact=1.5,
                      radius=list(radius),
                      center=list(center),
                      data_format=data_format,
                      extra_data=skeleton.contiguous(),
                      map_location=coordinates.device)

    for i in range(max_depth):
        t[coordinates].refine()

    t.shrink_to_fit()
    t.construct_tree(coordinates)

    return t


def warp_build_octree(coords, transformation_matrices, skeleton=None, vb_weights=None, vb_indices=None, radius=None,
                      max_depth=8):
    torch.cuda.synchronize()
    s = time.time()
    vs, ms = svox_t.warp_vertices(transformation_matrices, coords, vb_weights, vb_indices)
    torch.cuda.synchronize()
    wtime = time.time() - s

    torch.cuda.synchronize()
    s = time.time()
    t = build_octree(vs, radius=radius, max_depth=max_depth, skeleton=skeleton)
    torch.cuda.synchronize()
    btime = time.time() - s

    return t, wtime, btime

def rodrigues_rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
