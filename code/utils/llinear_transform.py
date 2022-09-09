import torch
import time

def compute_skinning_weights(pos, tmp_vertices, tmp_weights, n_influence=3, n_binding=10, chunks=10000, debug=False):
    if pos.shape[0] > chunks:
        coords = pos.split(chunks, dim=0)
    else:
        coords = [pos]

    weights = []

    for coord in coords:
        dis = coord[:, None, :] - tmp_vertices
        dis = dis.pow(2).sum(-1).sqrt()
        dis_min, indice = torch.topk(dis, n_influence, largest=False, dim=1)
        w = dis_min.max(-1)[0][..., None] - dis_min
        w = torch.softmax(w, dim=-1)
        weight = torch.sum(tmp_weights[indice] * w[..., None], dim=1)
        weights.append(weight)
    weights = torch.cat(weights, dim=0)

    weights, indices = torch.topk(weights, n_binding, largest=True, dim=1)
    weights = weights / weights.sum(-1).unsqueeze(-1)

    return weights, indices



def compute_transformation(src_pose, dst_pose, weights, indices, chunks=10000):
    if weights is None or indices is None:
        print('Please provide skinning weights.')
        return None

    transformation_matrices = []
    if weights.shape[0] > chunks:
        weights, indices = weights.split(chunks, dim=0), indices.split(chunks, dim=0)
    else:
        weights, indices = [weights], [indices]

    for i in range(len(weights)):
        weight, indice = weights[i], indices[i]
        src_matrices, dst_matricrs = src_pose[indice], dst_pose[indice]
        transformation_matrix = torch.matmul(dst_matricrs, src_matrices)
        transformation_matrix = torch.sum(transformation_matrix * weight[..., None, None], dim=1)

        transformation_matrices.append(transformation_matrix)

    transformation_matrices = torch.cat(transformation_matrices, dim=0)

    return transformation_matrices.float()


def transform_coords(coords, matrices, chunks=10000):
    if coords.shape[0] > chunks:
        coords, matrices = coords.split(chunks, dim=0), matrices.split(chunks, dim=0)
    else:
        coords, matrices = [coords], [matrices]

    transformed_coords = []

    for i in range(len(coords)):
        coord, ms = coords[i], matrices[i]
        coord = coord.unsqueeze(-1)
        coord = torch.cat([coord, torch.ones((coord.size(0), 1, coord.size(2)), device=coord.device)], dim=1)
        coord = torch.matmul(ms, coord)

        transformed_coords.append(coord)

    transformed_coords = torch.cat(transformed_coords, dim=0).squeeze()[:, :3]

    return transformed_coords


