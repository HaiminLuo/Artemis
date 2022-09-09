import os 
import sys 
import torch
import argparse
from config import cfg
from utils import *
from modeling import build_model
import cv2
from tqdm import tqdm

from imageio_ffmpeg import write_frames

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config", default=None, type=str, help="config file path."
)
parser.add_argument(
    "--dataset", default=None, type=str, help="dataset path."
)
parser.add_argument(
    "--model", default=None, type=str, help="checkpoint model path."
)
parser.add_argument(
    "--output_path", default='./out', type=str, help="image / videos output path."
)
parser.add_argument(
    "--render_video", action='store_true', default=False, help="render around view videos."
)
parser.add_argument(
    "--camera_path", default=None, type=str, help="path to cameras sequence for rendering around-view videos."
)



args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

config_path = args.config
dataset_path = args.dataset
model_path = args.model

# load config
cfg.merge_from_file(config_path)
tree_depth = cfg.MODEL.TREE_DEPTH
img_size = cfg.INPUT.SIZE_TEST

# load canonical volume
coords = torch.load(os.path.join(dataset_path, 'volumes/coords_init.pth'), map_location='cpu').float().cuda()
skeleton_init = torch.from_numpy(
        campose_to_extrinsic(np.loadtxt(os.path.join(dataset_path, 'bones/Bones_%04d.inf' % 0)))).float()
bone_parents = torch.from_numpy(np.load(os.path.join(dataset_path, 'bones/bone_parents.npy'))).long()
pose_init = get_local_eular(skeleton_init, bone_parents)

skinning_weights = torch.load(os.path.join(dataset_path, 'volumes/volume_weights.pth'),
                                     map_location='cpu').float().cuda()
joint_index = torch.load(os.path.join(dataset_path, 'volumes/volume_indices.pth'),
                                map_location='cpu').int().cuda()
volume_radius = torch.from_numpy(np.loadtxt(os.path.join(dataset_path, 'volumes/radius.txt'))).float()

# load cameras
camposes = np.loadtxt(os.path.join(dataset_path, 'CamPose.inf'))
Ts = torch.Tensor(campose_to_extrinsic(camposes))
center = torch.mean(coords, dim=0).cpu()
Ks = torch.Tensor(read_intrinsics(os.path.join(dataset_path, 'Intrinsic.inf')))
K = Ks[0]
K[:2, :3] /= 2

# load motions
sequences = []
seq_nums = []
with open(os.path.join(dataset_path, 'sequences'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        seq, num = line.strip().split(' ')
        sequences.append(seq)
        seq_nums.append(int(num))

pose_features = []
bone_matrices = []
skeletons = []
seq_trees = []

pose_feature_pad = torch.cat([skeleton_init[:, :3, 3], pose_init], dim=1)

for seq_id, seq in enumerate(sequences):
    seq_num = seq_nums[seq_id]
    skeleton, pose_feature_s, bone_matrice = [], [], []
    seq_trees.append([None] * seq_num)
    for i in range(seq_num):
        pose = torch.from_numpy(
                campose_to_extrinsic(np.loadtxt(
                os.path.join(dataset_path, 'bones/%s/Bones_%04d.inf' % (seq, i))))).float()
        matrices = torch.matmul(skeleton_init, torch.inverse(pose))
        skeleton.append(pose.unsqueeze(0))
        bone_matrice.append(matrices.unsqueeze(0))

        pose = get_local_eular(pose, bone_parents)
        delta_pose = pose - pose_init
        pose_feature = torch.cat([pose_feature_pad, delta_pose], dim=1)
        if torch.any(torch.isnan(pose_feature)):
            pose_feature = torch.where(torch.isnan(pose_feature), torch.zeros_like(pose_feature).float(),
                                           pose_feature)

        pose_feature_s.append(pose_feature.unsqueeze(0))

    skeleton = torch.cat(skeleton, dim=0).float()
    pose_feature_s = torch.cat(pose_feature_s, dim=0).float()
    bone_matrice = torch.cat(bone_matrice, dim=0).float()

    skeletons.append(skeleton.cuda())
    pose_features.append(pose_feature_s.cuda())
    bone_matrices.append(bone_matrice.cuda())

# load models
model = build_model(cfg, flut_size=coords.shape[0], bone_feature_dim=pose_features[0].shape[2]).cuda()
checkpoint = torch.load(model_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()

# render
if not args.render_video:
    seq_id = 0
    frame_id = 0
    cam_id = 0
    cur_T = Ts[cam_id]
    if seq_trees[seq_id][frame_id] is None:
        transformation_matrices = torch.inverse(bone_matrices[seq_id][frame_id])
        skeleton = skeletons[seq_id][frame_id][:, :3, 3]
        t, _, _ = warp_build_octree(coords, transformation_matrices=transformation_matrices,
                                            skeleton=skeleton, vb_weights=skinning_weights,
                                            vb_indices=joint_index, radius=volume_radius,
                                            max_depth=tree_depth)
        seq_trees[seq_id][frame_id] = t

    rgb, mask, feature = render_image(cfg, model, K, cur_T, (img_size[1], img_size[0]),
                                    tree=seq_trees[seq_id][frame_id],
                                    matrices=bone_matrices[seq_id][frame_id],
                                    joint_features=pose_features[seq_id][frame_id],
                                    bg=None, skinning_weights=skinning_weights,
                                    joint_index=joint_index)

    cv2.imwrite('rgb.jpg', cv2.cvtColor(rgb * 255, cv2.COLOR_BGR2RGB))
    cv2.imwrite('feature.jpg', feature * 255)
else:
    assert args.camera_path is not None, 'Please provide cameras trajectory.'
    camposes = np.loadtxt(os.path.join(args.camera_path, 'CamPose_spiral.inf'))
    Ts = torch.Tensor(campose_to_extrinsic(camposes))
    center = torch.mean(coords, dim=0).cpu()
    Ks = torch.Tensor(read_intrinsics(os.path.join(args.camera_path, 'Intrinsic_spiral.inf')))

    for seq_id in range(len(sequences)):
        
        writer_raw_rgb = write_frames(os.path.join(args.output_path, '%s_rgb.mp4' % sequences[seq_id]), img_size, fps=30, macro_block_size=8, quality=6)  # size is (width, height)
        writer_raw_alpha = write_frames(os.path.join(args.output_path, '%s_alpha.mp4' % sequences[seq_id]), img_size, fps=30, macro_block_size=8, quality=6)  # size is (width, height)
        writer_raw_feature = write_frames(os.path.join(args.output_path, '%s_feature.mp4' % sequences[seq_id]), img_size, fps=30, macro_block_size=8, quality=6)  # size is (width, height)
        writer_raw_rgb.send(None)
        writer_raw_alpha.send(None)
        writer_raw_feature.send(None)

        for cam_id in tqdm(range(Ts.shape[0]), unit=" frame", desc=f"Rendering video"):
            frame_id = cam_id % seq_nums[seq_id]
            
            if seq_trees[seq_id][frame_id] is None:
                transformation_matrices = torch.inverse(bone_matrices[seq_id][frame_id])
                skeleton = skeletons[seq_id][frame_id][:, :3, 3]
                t, wtime, btime = warp_build_octree(coords, transformation_matrices=transformation_matrices,
                                                    skeleton=skeleton, vb_weights=skinning_weights,
                                                    vb_indices=joint_index, radius=volume_radius,
                                                    max_depth=tree_depth)
                seq_trees[seq_id][frame_id] = t

            rgb, mask, feature = render_image(cfg, model, Ks[cam_id], Ts[cam_id], (img_size[1], img_size[0]),
                                            tree=seq_trees[seq_id][frame_id],
                                            matrices=bone_matrices[seq_id][frame_id],
                                            joint_features=pose_features[seq_id][frame_id],
                                            bg=None, skinning_weights=skinning_weights,
                                            joint_index=joint_index)

            img = rgb * 255
            feature = feature * 255
            alpha = torch.from_numpy(mask).unsqueeze(-1).repeat(1, 1, 3).numpy() * 255
            img = img.copy(order='C')
            writer_raw_rgb.send(img.astype(np.uint8))
            writer_raw_alpha.send(alpha.astype(np.uint8))
            writer_raw_feature.send(feature.astype(np.uint8))

            cv2.imwrite(os.path.join(args.output_path, '%s_rgb_%04d.jpg' % (sequences[seq_id], cam_id)),
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(args.output_path, '%s_alpha_%04d.jpg' % (sequences[seq_id], cam_id)), alpha)
            cv2.imwrite(os.path.join(args.output_path, '%s_feature_%04d.jpg' % (sequences[seq_id], cam_id)), feature)

        writer_raw_rgb.close()
        writer_raw_alpha.close()
        writer_raw_feature.close()
