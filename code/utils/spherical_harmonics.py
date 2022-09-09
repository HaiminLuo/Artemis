import torch


def ComputeSH(dirs):
    '''
        dirs: b*3
    '''
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = dirs[..., 2]
    xx = dirs[..., 0] ** 2
    yy = dirs[..., 1] ** 2
    zz = dirs[..., 2] ** 2

    xy = dirs[..., 0] * dirs[..., 1]
    yz = dirs[..., 1] * dirs[..., 2]
    xz = dirs[..., 0] * dirs[..., 2]

    sh = torch.zeros((dirs.shape[0], 25)).to(dirs.device)

    sh[:, 0] = 0.282095

    sh[:, 1] = -0.4886025119029199 * y
    sh[:, 2] = 0.4886025119029199 * z
    sh[:, 3] = -0.4886025119029199 * x

    sh[:, 4] = 1.0925484305920792 * xy
    sh[:, 5] = -1.0925484305920792 * yz
    sh[:, 6] = 0.31539156525252005 * (2.0 * zz - xx - yy)
    sh[:, 7] = -1.0925484305920792 * xz
    # sh2p2
    sh[:, 8] = 0.5462742152960396 * (xx - yy)

    sh[:, 9] = -0.5900435899266435 * y * (3 * xx - yy)
    sh[:, 10] = 2.890611442640554 * xy * z
    sh[:, 11] = -0.4570457994644658 * y * (4 * zz - xx - yy)
    sh[:, 12] = 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy)
    sh[:, 13] = -0.4570457994644658 * x * (4 * zz - xx - yy)
    sh[:, 14] = 1.445305721320277 * z * (xx - yy)
    sh[:, 15] = -0.5900435899266435 * x * (xx - 3 * yy)

    sh[:, 16] = 2.5033429417967046 * xy * (xx - yy)
    sh[:, 17] = -1.7701307697799304 * yz * (3 * xx - yy)
    sh[:, 18] = 0.9461746957575601 * xy * (7 * zz - 1.0)
    sh[:, 19] = -0.6690465435572892 * yz * (7 * zz - 3.0)
    sh[:, 20] = 0.10578554691520431 * (zz * (35 * zz - 30) + 3)
    sh[:, 21] = -0.6690465435572892 * xz * (7 * zz - 3)
    sh[:, 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1.0)
    sh[:, 23] = -1.7701307697799304 * xz * (xx - 3 * yy)
    sh[:, 24] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

    return sh


def computeRGB(dirs, coeff):
    '''
    dirs: n * 3
    coeff: n * n_components * f
    '''
    n_components = coeff.shape[1]

    return (ComputeSH(dirs)[..., :n_components].unsqueeze(-1) * coeff).sum(1)