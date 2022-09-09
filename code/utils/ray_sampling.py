import torch
import math

'''
Sample rays from views (and images) with/without masks

--------------------------
INPUT Tensors
Ks: intrinsics of cameras (M,3,3)
Ts: extrinsic of cameras (M,4,4)
image_size: the size of image [H,W]
images: (M,C,H,W)
mask_threshold: a float threshold to mask rays
masks:(M,H,W)
depth:(M,H,W)
-------------------
OUPUT:
list of rays:  (N,6)  dirs(3) + pos(3)
RGB:  (N,C)
'''


def ray_sampling(Ks, Ts, image_size, masks=None, mask_threshold=0.5, images=None, depth=None,
                 far_depth=None, fine_depth=None):
    h = image_size[0]
    w = image_size[1]
    M = Ks.size(0)

    x = torch.linspace(0, h - 1, steps=h, device=Ks.device)
    y = torch.linspace(0, w - 1, steps=w, device=Ks.device)

    grid_x, grid_y = torch.meshgrid(x, y)
    coordinates = torch.stack([grid_y, grid_x]).unsqueeze(0).repeat(M, 1, 1, 1)  # (M,2,H,W)
    coordinates = torch.cat([coordinates, torch.ones(coordinates.size(0), 1, coordinates.size(2),
                                                     coordinates.size(3), device=Ks.device)], dim=1).permute(0, 2, 3,
                                                                                                             1).unsqueeze(
        -1)

    inv_Ks = torch.inverse(Ks)

    dirs = torch.matmul(inv_Ks, coordinates)  # (M,H,W,3,1)
    dirs = dirs / torch.norm(dirs, dim=3, keepdim=True)

    # zs to depth
    td = dirs.view(M, h, w, -1)
    if depth is not None:
        depth = depth.to(Ks.device)
        depth = depth / td[:, :, :, 2]
    if far_depth is not None:
        far_depth = far_depth.to(Ks.device)
        far_depth = far_depth / td[:, :, :, 2]
    else:
        far_depth = depth
    if fine_depth is not None:
        fine_depth = fine_depth.to(Ks.device)
        fine_depth = fine_depth / td[:, :, :, 2]
    else:
        fine_depth = depth

    dirs = torch.cat([dirs, torch.zeros(dirs.size(0), coordinates.size(1),
                                        coordinates.size(2), 1, 1, device=Ks.device)], dim=3)  # (M,H,W,4,1)

    dirs = torch.matmul(Ts, dirs)  # (M,H,W,4,1)
    dirs = dirs[:, :, :, 0:3, 0]  # (M,H,W,3)

    pos = Ts[:, 0:3, 3]  # (M,3)
    pos = pos.unsqueeze(1).unsqueeze(1).repeat(1, h, w, 1)

    rays = torch.cat([dirs, pos], dim=3)  # (M,H,W,6)
    if depth is not None:
        depth = torch.stack([depth, far_depth, fine_depth], dim=3)  # (M,H,W,2)
    ds = None

    if images is not None:
        images = images.to(Ks.device)
        rgbs = images.permute(0, 2, 3, 1)  # (M,H,W,C)
    else:
        rgbs = None

    if masks is not None:
        rays = rays[masks > mask_threshold, :]
        if rgbs is not None:
            rgbs = rgbs[masks > mask_threshold, :]
        if depth is not None:
            ds = depth[masks > mask_threshold, :]
    else:
        rays = rays.reshape((-1, rays.size(3)))
        if rgbs is not None:
            rgbs = rgbs.reshape((-1, rgbs.size(3)))
        if depth is not None:
            ds = depth.reshape((-1, depth.size(3)))

    return rays, rgbs, ds


def patch_sampling(Ks, Ts, image_size, patch_size=32, images=None, depth=None, far_depth=None, fine_depth=None,
                   alpha=None, keep_bg=False):
    h = image_size[0]
    w = image_size[1]
    M = Ks.size(0)

    h_num, w_num = math.ceil(h / patch_size), math.ceil(w / patch_size)
    ray_patches, rgb_patches, depth_patches, alpha_patches = [], [], [], []

    x = torch.linspace(0, h - 1, steps=h, device=Ks.device)
    y = torch.linspace(0, w - 1, steps=w, device=Ks.device)

    grid_x, grid_y = torch.meshgrid(x, y)
    coordinates = torch.stack([grid_y, grid_x]).unsqueeze(0).repeat(M, 1, 1, 1)  # (M,2,H,W)
    coordinates = torch.cat([coordinates, torch.ones(coordinates.size(0), 1, coordinates.size(2),
                                                     coordinates.size(3), device=Ks.device)], dim=1).permute(0, 2, 3,
                                                                                                             1).unsqueeze(
        -1)

    inv_Ks = torch.inverse(Ks)

    dirs = torch.matmul(inv_Ks, coordinates)  # (M,H,W,3,1)
    dirs = dirs / torch.norm(dirs, dim=3, keepdim=True)

    # zs to depth
    td = dirs.view(M, h, w, -1)
    if depth is not None:
        depth = depth / td[:, :, :, 2]
    if far_depth is not None:
        far_depth = far_depth / td[:, :, :, 2]
    else:
        far_depth = depth
    if fine_depth is not None:
        fine_depth = fine_depth / td[:, :, :, 2]
    else:
        fine_depth = depth

    dirs = torch.cat([dirs, torch.zeros(dirs.size(0), coordinates.size(1),
                                        coordinates.size(2), 1, 1, device=Ks.device)], dim=3)  # (M,H,W,4,1)

    dirs = torch.matmul(Ts, dirs)  # (M,H,W,4,1)
    dirs = dirs[:, :, :, 0:3, 0]  # (M,H,W,3)

    pos = Ts[:, 0:3, 3]  # (M,3)
    pos = pos.unsqueeze(1).unsqueeze(1).repeat(1, h, w, 1)

    rays = torch.cat([dirs, pos], dim=3)  # (M,H,W,6)

    if images is not None:
        rgbs = images.permute(0, 2, 3, 1)  # (M,H,W,C)
    else:
        rgbs = None

    depth = torch.stack([depth, far_depth, fine_depth], dim=3)  # (M,H,W,2)

    padded_h, padded_w = h_num * patch_size, w_num * patch_size
    ray_padded, rgb_padded, depth_padded, alpha_padded = torch.zeros([rays.size(0), padded_h, padded_w, rays.size(3)],
                                                                     device=rays.device), \
                                                         torch.zeros([rgbs.size(0), padded_h, padded_w, rgbs.size(3)],
                                                                     device=rgbs.device), \
                                                         torch.zeros([depth.size(0), padded_h, padded_w, depth.size(3)],
                                                                     device=depth.device), \
                                                         torch.zeros([alpha.size(0), padded_h, padded_w],
                                                                     device=alpha.device)
    ray_padded[:, :h, :w, :] = rays
    rgb_padded[:, :h, :w, :] = rgbs
    depth_padded[:, :h, :w, :] = depth
    alpha_padded[:, :h, :w] = alpha

    rays = ray_padded.view(int(padded_h / patch_size), patch_size, int(padded_w / patch_size), patch_size, -1).permute(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, ray_padded.shape[-1])
    rgbs = rgb_padded.view(int(padded_h / patch_size), patch_size, int(padded_w / patch_size), patch_size, -1).permute(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, rgb_padded.shape[-1])
    ds = depth_padded.view(int(padded_h / patch_size), patch_size, int(padded_w / patch_size), patch_size, -1).permute(0, 2, 1, 3, 4).reshape(-1, patch_size, patch_size, depth_padded.shape[-1])
    als = alpha_padded.view(int(padded_h / patch_size), patch_size, int(padded_w / patch_size), patch_size).permute(0, 2, 1, 3).reshape(-1, patch_size, patch_size)

    mask = ds.reshape(ds.shape[0], -1).sum(1) > 0

    return rays[mask], rgbs[mask], ds[mask], als[mask]
