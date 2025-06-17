# Code for computing the absolute pose errors for TUM dataset

from .wrappers import Camera, Pose
import torch
from .epipolar import angle_error_mat
    

def projection_error(kpts0_world, kpts1, cam1: Camera, T_0to1: Pose):
    # Computes distance between world kpts projected to image and cam kpts

    # Transform world points from world to camera frame
    kpts0_cam_3d = T_0to1.transform(kpts0_world)
    # Project into image plane
    kpts0_cam_2d, valid = cam1.cam2image(kpts0_cam_3d)
    dist0_1 = ((kpts0_cam_2d - kpts1) ** 2).sum(-1).sqrt()

    return dist0_1


def dist_error_vec(v1, v2, eps=1e-10):
    return (v1-v2).norm(dim=-1)


def absolute_pose_error(T_0to1, R, t, offset, ignore_gt_t_thr=0.0, eps=1e-10):
    if isinstance(T_0to1, torch.Tensor):
        R_gt, t_gt = T_0to1[:3, :3], T_0to1[:3, 3]
    else:
        R_gt, t_gt = T_0to1.R, T_0to1.t
    R_gt, t_gt = torch.squeeze(R_gt), torch.squeeze(t_gt)

    # Apply offset from pose estimation to GT
    t_gt = t_gt + torch.mv(R_gt, offset)

    # distance between transformation vectors
    t_err = dist_error_vec(t, t_gt, eps)

    # angle error between 2 rotation matrices
    r_err = angle_error_mat(R, R_gt)

    return t_err, r_err