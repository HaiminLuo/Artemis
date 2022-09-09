import os
import sys
import torch

import svox_t

def build_octree(coordinates, radius=None, center=None, skeleton=None, data_dim=91, max_depth=10, data_format='SH9'):
    coordinates = coordinates.contiguous().cuda().detach()

    maxs_ = torch.max(coordinates, dim=0).values
    mins_ = torch.min(coordinates, dim=0).values
    radius_ = (maxs_ - mins_) / 2
    center_ = mins_ + radius_

    if radius is None:
        radius = radius_

    if center is None:
        center = center_
    # print(center, radius)
    t = svox_t.N3Tree(N=2,
                      data_dim=data_dim,
                      depth_limit=max_depth,
                      init_reserve=5000,
                      init_refine=0,
                      #                 geom_resize_fact=1.5,
                      radius=list(radius),
                      center=list(center),
                      data_format=data_format,
                      extra_data=skeleton, ).cuda()

    for i in range(max_depth):
        t[coordinates].refine()

    t.shrink_to_fit()
    t.construct_tree(coordinates)

    return t.cpu()

def load_octree(tree_path):
    return svox_t.N3Tree.load(tree_path)

def generate_transformation_matrices(matrices, skinning_weights, joint_index):
    return svox_t.blend_transformation_matrix(matrices, skinning_weights, joint_index)