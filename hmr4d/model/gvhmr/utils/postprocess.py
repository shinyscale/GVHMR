import torch
from torch.cuda.amp import autocast
from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)

import hmr4d.utils.matrix as matrix
from hmr4d.utils.ik.ccd_ik import CCD_IK
from hmr4d.utils.geo_transform import get_sequence_cammat, transform_mat, apply_T_on_points
from hmr4d.utils.net_utils import gaussian_smooth
from hmr4d.model.gvhmr.utils.endecoder import EnDecoder

from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines


def _as_identity(R):
    is_I = matrix_to_axis_angle(R).norm(dim=-1) < 1e-5
    R = R.clone()
    R[is_I] = torch.eye(3)[None].expand(is_I.sum(), -1, -1).to(R)
    return R


def _yaw_from_matrix(R):
    forward = R[..., :, 2]
    return torch.atan2(forward[..., 0], forward[..., 2])


def _yaw_to_matrix(yaw):
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    row0 = torch.stack([cos_yaw, zeros, sin_yaw], dim=-1)
    row1 = torch.stack([zeros, ones, zeros], dim=-1)
    row2 = torch.stack([-sin_yaw, zeros, cos_yaw], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _smooth_heading(heading, sigma):
    sin_h = gaussian_smooth(torch.sin(heading), sigma=sigma, dim=-1)
    cos_h = gaussian_smooth(torch.cos(heading), sigma=sigma, dim=-1)
    return torch.atan2(sin_h, cos_h)


@autocast(enabled=False)
def get_stable_global_orient(global_orient_gv, global_orient_c, cam_angvel):
    """Rebuild world yaw from a fused body-anchored and VO camera heading."""
    B, L = cam_angvel.shape[:2]
    R_gv = axis_angle_to_matrix(global_orient_gv)
    R_c = axis_angle_to_matrix(global_orient_c)

    # Absolute camera heading derived from the network's GV/body relationship.
    R_c2gv = R_gv @ R_c.mT
    heading_body = _yaw_from_matrix(R_c2gv)

    # Relative camera motion from VO anchored to the first-frame camera heading.
    R_t_to_tp1 = _as_identity(rotation_6d_to_matrix(cam_angvel.float()))
    cam_to_world = [R_c2gv[:, 0]]
    for i in range(1, L):
        cam_to_world.append(cam_to_world[-1] @ R_t_to_tp1[:, i - 1].mT)
    cam_to_world = torch.stack(cam_to_world, dim=1)
    heading_vo = _yaw_from_matrix(cam_to_world)

    # Blend a heavily smoothed absolute heading with a lighter VO heading so
    # we keep real pans while rejecting long-horizon drift.
    sigma_body = max(5, min(31, L // 48 or 5))
    sigma_vo = max(3, min(15, L // 96 or 3))
    heading_body = _smooth_heading(heading_body, sigma=sigma_body)
    heading_vo = _smooth_heading(heading_vo, sigma=sigma_vo)
    sin_heading = 0.75 * torch.sin(heading_body) + 0.25 * torch.sin(heading_vo)
    cos_heading = 0.75 * torch.cos(heading_body) + 0.25 * torch.cos(heading_vo)
    heading = torch.atan2(sin_heading, cos_heading)

    global_orient = matrix_to_axis_angle(_yaw_to_matrix(heading) @ R_c)
    return global_orient


@autocast(enabled=False)
def pp_static_joint(outputs, endecoder: EnDecoder):
    # Global FK
    pred_w_j3d = endecoder.fk_v2(**outputs["pred_smpl_params_global"])
    L = pred_w_j3d.shape[1]
    joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    pred_j3d_static = pred_w_j3d.clone()[:, :, joint_ids]  # (B, L, J, 3)

    ######## update overall movement with static info, and make displacement ~[0,0,0]
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J, 3)

    static_conf_logits = outputs["static_conf_logits"][:, :-1].clone()
    static_label_ = static_conf_logits > 0  # (B, L-1, J) # avoid non-contact frame
    static_conf_logits = static_conf_logits.float() - (~static_label_ * 1e6)  # fp16 cannot go through softmax
    is_static = static_label_.sum(dim=-1) > 0  # (B, L-1)

    pred_disp = pred_j_disp * static_conf_logits[..., None].softmax(dim=-2)  # (B, L-1, J, 3)
    pred_disp = pred_disp * is_static[..., None, None]  # (B, L-1, J, 3)
    pred_disp = pred_disp.sum(-2)  # (B, L-1, 3)
    ####################

    # Overwrite results:
    if False:  # for-loop
        post_w_transl = outputs["pred_smpl_params_global"]["transl"].clone()  # (B, L, 3)
        for i in range(1, L):
            post_w_transl[:, i:] -= pred_disp[:, i - 1 : i]
    else:  # vectorized
        pred_w_transl = outputs["pred_smpl_params_global"]["transl"].clone()  # (B, L, 3)
        pred_w_disp = pred_w_transl[:, 1:] - pred_w_transl[:, :-1]  # (B, L-1, 3)
        pred_w_disp_new = pred_w_disp - pred_disp
        post_w_transl = torch.cumsum(torch.cat([pred_w_transl[:, :1], pred_w_disp_new], dim=1), dim=1)
        post_w_transl[..., 0] = gaussian_smooth(post_w_transl[..., 0], dim=-1)
        post_w_transl[..., 2] = gaussian_smooth(post_w_transl[..., 2], dim=-1)

    # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
    post_w_j3d = pred_w_j3d - pred_w_transl.unsqueeze(-2) + post_w_transl.unsqueeze(-2)
    ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]  # (B,)  Minimum y value
    post_w_transl[..., 1] -= ground_y

    return post_w_transl


@autocast(enabled=False)
def pp_static_joint_cam(outputs, endecoder: EnDecoder):
    """Use static joint and static camera assumption to postprocess the global transl"""
    # input
    pred_smpl_params_incam = outputs["pred_smpl_params_incam"].copy()
    pred_smpl_params_global = outputs["pred_smpl_params_global"]
    static_conf_logits = outputs["static_conf_logits"].clone()[:, :-1]  # (B, L-1, J)
    joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    B, L = pred_smpl_params_incam["transl"].shape[:2]
    assert B == 1

    # FK
    pred_w_j3d = endecoder.fk_v2(**pred_smpl_params_global)  # (B, L, J, 3)
    # smooth incam results, as this could be noisy
    pred_smpl_params_incam["transl"] = gaussian_smooth(pred_smpl_params_incam["transl"], sigma=5, dim=-2)
    pred_c_j3d = endecoder.fk_v2(**pred_smpl_params_incam)  # (B, L, J, 3)

    # compute T_c2w (static) from first frame
    R_gv = axis_angle_to_matrix(pred_smpl_params_global["global_orient"][:, 0])  # (B, 3, 3)
    R_c = axis_angle_to_matrix(pred_smpl_params_incam["global_orient"][:, 0])  # (B, 3, 3)
    R_c2w = R_gv @ R_c.mT  # (B, 3, 3)
    t_c2w = pred_w_j3d[:, 0, 0] - torch.einsum("bij,bj->bi", R_c2w, pred_c_j3d[:, 0, 0])  # (B, 3)
    T_c2w = transform_mat(R_c2w, t_c2w)  # (B, 4, 4)
    pred_c_j3d_in_w = apply_T_on_points(pred_c_j3d, T_c2w[:, None])

    # 1. Make transl similar to incam
    post_w_transl = pred_smpl_params_global["transl"].clone()  # (B, L, 3)
    post_w_j3d = pred_w_j3d.clone()  # (B, L, J, 3)
    cp_thr = torch.tensor([0.25, 0.25, 0.25]).to(post_w_j3d)  # Only update very bad pred
    for i in range(1, L):
        cp_diff = post_w_j3d[:, i, 0] - pred_c_j3d_in_w[:, i, 0]  # (B, 3)
        cp_diff = cp_diff * ~((cp_diff > -cp_thr) * (cp_diff < cp_thr))
        cp_diff = torch.clamp(cp_diff, -0.02, 0.02)
        post_w_transl[:, i:] -= cp_diff
        post_w_j3d[:, i:] -= (cp_diff)[:, None, None]

    # 1. Make stationary joint stay stationary
    # pred_j3d_static = pred_w_j3d.clone()[:, :, joint_ids]  # (B, L, J, 3)
    pred_j3d_static = post_w_j3d[:, :, joint_ids]  # (B, L, J, 3)
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J, 3)

    static_label = static_conf_logits.sigmoid() > 0.8  # (B, L-1, J)
    static_label_sumJ = static_label.sum(-1, keepdim=True)  # (B, L-1, 1)
    static_label_sumJ = torch.clamp_min(static_label_sumJ, 1)  # replace 0 with 1
    pred_disp_sumJ = (pred_j_disp * static_label[..., None]).sum(-2)  # (B, L-1, 3)
    pred_disp = pred_disp_sumJ / static_label_sumJ  # (B, L-1, 3)
    pred_disp[:, :, 1] = 0  # do not modify y

    # Overwrite results (for-loop)
    for i in range(1, L):
        post_w_transl[:, i:] -= pred_disp[:, [i - 1]]
        post_w_j3d[:, i:] -= pred_disp[:, [i - 1], None]

    # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
    ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]  # (B,)  Minimum y value
    post_w_transl[..., 1] -= ground_y

    return post_w_transl


@autocast(enabled=False)
def process_ik(outputs, endecoder):
    static_conf = outputs["static_conf_logits"].sigmoid()  # (B, L, J)
    post_w_j3d, local_mat, post_w_mat = endecoder.fk_v2(**outputs["pred_smpl_params_global"], get_intermediate=True)

    # sebas rollout merge
    joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    post_target_j3d = post_w_j3d.clone()
    for i in range(1, post_w_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, joint_ids]
        this = post_w_j3d[:, i, joint_ids]
        c_prev = static_conf[:, i - 1, :, None]
        post_target_j3d[:, i, joint_ids] = prev * c_prev + this * (1 - c_prev)

    # ik
    global_rot = matrix.get_rotation(post_w_mat)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    left_leg_chain = [0, 1, 4, 7, 10]
    right_leg_chain = [0, 2, 5, 8, 11]
    left_hand_chain = [9, 13, 16, 18, 20]
    right_hand_chain = [9, 14, 17, 19, 21]

    def ik(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        IK_solver = CCD_IK(
            local_mat,
            parents,
            target_ind,
            target_pos,
            target_rot,
            kinematic_chain=chain,
            max_iter=2,
        )

        chain_local_mat = IK_solver.solve()
        chain_rotmat = matrix.get_rotation(chain_local_mat)
        local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]  # (B, L, J, 3, 3)
        return local_mat

    local_mat = ik(local_mat, post_target_j3d[:, :, [7, 10]], global_rot[:, :, [7, 10]], [3, 4], left_leg_chain)
    local_mat = ik(local_mat, post_target_j3d[:, :, [8, 11]], global_rot[:, :, [8, 11]], [3, 4], right_leg_chain)
    local_mat = ik(local_mat, post_target_j3d[:, :, [20]], global_rot[:, :, [20]], [4], left_hand_chain)
    local_mat = ik(local_mat, post_target_j3d[:, :, [21]], global_rot[:, :, [21]], [4], right_hand_chain)

    body_pose = matrix_to_axis_angle(matrix.get_rotation(local_mat[:, :, 1:]))  # (B, L, J-1, 3, 3)
    body_pose = body_pose.flatten(2)  # (B, L, (J-1)*3)

    return body_pose
