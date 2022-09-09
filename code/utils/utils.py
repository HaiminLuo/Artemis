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


def campose_to_extrinsic(camposes):
    if camposes.shape[1] != 12:
        raise Exception(" wrong campose data structure!")
        return

    res = np.zeros((camposes.shape[0], 4, 4))

    res[:, 0:3, 2] = camposes[:, 0:3]
    res[:, 0:3, 0] = camposes[:, 3:6]
    res[:, 0:3, 1] = camposes[:, 6:9]
    res[:, 0:3, 3] = camposes[:, 9:12]
    res[:, 3, 3] = 1.0

    return res


def read_intrinsics(fn_instrinsic):
    fo = open(fn_instrinsic)
    data = fo.readlines()
    i = 0
    Ks = []
    while i < len(data):
        if len(data[i]) > 6:
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            a = np.array(tmp)
            i = i + 1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            b = np.array(tmp)
            i = i + 1
            tmp = data[i].split()
            tmp = [float(i) for i in tmp]
            c = np.array(tmp)
            res = np.vstack([a, b, c])
            Ks.append(res)

        i = i + 1
    Ks = np.stack(Ks)
    fo.close()

    return Ks
