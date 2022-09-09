import os
import torch
import numpy as np
from pytorch3d.transforms import so3_log_map, so3_rotation_angle, so3_relative_angle, matrix_to_euler_angles, euler_angles_to_matrix


def get_local_transformation(transformations, parents):
    parents_t = torch.cat([torch.eye(4).unsqueeze(0), transformations[parents[1:]]])
    local_t = torch.matmul(torch.inverse(parents_t), transformations)

    return local_t


def get_local_eular(transformations, parents, axis='XYZ'):
    local_t = get_local_transformation(transformations, parents)
    local_t = torch.clamp(local_t[:, :3, :3], max=1.)
    local_eular = matrix_to_euler_angles(local_t, axis)

    return local_eular


def rotation_matrix_to_eular(rotations, axis='XYZ'):
    return matrix_to_euler_angles(rotations[:, :3, :3], axis)


def eular_to_rotation_matrix(eular, axis='XYZ'):
    return euler_angles_to_matrix(eular, axis)

