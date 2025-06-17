# Code from gluefactory (https://github.com/cvg/glue-factory)
# -----------------------------------------------------------
# Modified by Simone Gaisbauer: 
# - Added methods eval_matches_projection, eval_absolute_pose_robust, eval_poses for evaluation of TUM dataset
# - Removed unused methods
# -----------------------------------------------------------

import numpy as np
import torch
from kornia.geometry.homography import find_homography_dlt

from ..geometry.epipolar import generalized_epi_dist, relative_pose_error
from ..geometry.poses import projection_error, absolute_pose_error
from ..geometry.homography import homography_corner_error, sym_homography_error
from ..robust_estimators import load_estimator
from ..utils.tensor import index_batch
from ..utils.tools import AUCMetric


def check_keys_recursive(d, pattern):
    if isinstance(pattern, dict):
        {check_keys_recursive(d[k], v) for k, v in pattern.items()}
    else:
        for k in pattern:
            assert k in d.keys()


def get_matches_scores(kpts0, kpts1, matches0, mscores0):
    m0 = matches0 > -1
    m1 = matches0[m0]
    pts0 = kpts0[m0]
    pts1 = kpts1[m1]
    scores = mscores0[m0]
    return pts0, pts1, scores


def eval_per_batch_item(data: dict, pred: dict, eval_f, *args, **kwargs):
    # Batched data
    results = [
        eval_f(data_i, pred_i, *args, **kwargs)
        for data_i, pred_i in zip(index_batch(data), index_batch(pred))
    ]
    # Return a dictionary of lists with the evaluation of each item
    return {k: [r[k] for r in results] for k in results[0].keys()}


def eval_matches_projection(data: dict, pred: dict) -> dict:
    check_keys_recursive(data, ["view1", "T_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    kp0, kp1, kp0_world = pred["keypoints0"], pred["keypoints1"], pred["keypoints0_world"]
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]

    # Case where no matches
    # All values in matches are <=-1
    if torch.all(m0 <= -1):
        return {
            "prec@5": 0.0,
            "prec@10": 0.0,
            "prec@30": 0.0,
            "num_matches": 0,
            "num_keypoints": (kp0.shape[0] + kp1.shape[0]) / 2.0,
        }	    
    
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
    pts0_world, _, _ = get_matches_scores(kp0_world, kp1, m0, scores0)
    results = {}

    # match metrics
    pix_err = projection_error(
        pts0_world[None], 
        pts1[None], 
        data["view1"]["camera"],
        data["T_0to1"])[0]
    # print if any pixel error is nan
    if torch.isnan(pix_err).any():
        print("nan pixel error")

    results["prec@5"] = (pix_err < 5).float().mean()
    results["prec@10"] = (pix_err < 10).float().mean()
    results["prec@30"] = (pix_err < 30).float().mean()

    results["num_matches"] = pts0.shape[0]
    results["num_keypoints"] = (kp0.shape[0] + kp1.shape[0]) / 2.0

    return results


def eval_matches_epipolar(data: dict, pred: dict) -> dict:
    check_keys_recursive(data, ["view0", "view1", "T_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)

    results = {}

    # match metrics
    n_epi_err = generalized_epi_dist(
        pts0[None],
        pts1[None],
        data["view0"]["camera"],
        data["view1"]["camera"],
        data["T_0to1"],
        False,
        essential=True,
    )[0]
    results["epi_prec@1e-4"] = (n_epi_err < 1e-4).float().mean()
    results["epi_prec@5e-4"] = (n_epi_err < 5e-4).float().mean()
    results["epi_prec@1e-3"] = (n_epi_err < 1e-3).float().mean()

    results["num_matches"] = pts0.shape[0]
    results["num_keypoints"] = (kp0.shape[0] + kp1.shape[0]) / 2.0

    return results


def eval_matches_homography(data: dict, pred: dict) -> dict:
    check_keys_recursive(data, ["H_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    H_gt = data["H_0to1"]
    if H_gt.ndim > 2:
        return eval_per_batch_item(data, pred, eval_matches_homography)

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
    err = sym_homography_error(pts0, pts1, H_gt)
    results = {}
    results["prec@1px"] = (err < 1).float().mean().nan_to_num().item()
    results["prec@3px"] = (err < 3).float().mean().nan_to_num().item()
    results["num_matches"] = pts0.shape[0]
    results["num_keypoints"] = (kp0.shape[0] + kp1.shape[0]) / 2.0
    return results


def eval_relative_pose_robust(data, pred, conf):
    check_keys_recursive(data, ["view0", "view1", "T_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    T_gt = data["T_0to1"]
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)

    results = {}

    estimator = load_estimator("relative_pose", conf["estimator"])(conf)
    data_ = {
        "m_kpts0": pts0,
        "m_kpts1": pts1,
        "camera0": data["view0"]["camera"][0],
        "camera1": data["view1"]["camera"][0],
    }
    est = estimator(data_)

    if not est["success"]:
        results["rel_pose_error"] = float("inf")
        results["ransac_inl"] = 0
        results["ransac_inl%"] = 0
    else:
        # R, t, inl = ret
        M = est["M_0to1"]
        inl = est["inliers"].numpy()
        t_error, r_error = relative_pose_error(T_gt, M.R, M.t)
        results["rel_pose_error"] = max(r_error, t_error)
        results["ransac_inl"] = np.sum(inl)
        results["ransac_inl%"] = np.mean(inl)

    return results


def eval_absolute_pose_robust(data, pred, conf):
    check_keys_recursive(data, ["view1", "T_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    T_gt = data["T_0to1"]
    kp0, kp1, kp0_world = pred["keypoints0"], pred["keypoints1"], pred["keypoints0_world"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
    pts0_world, _, _ = get_matches_scores(kp0_world, kp1, m0, scores0)
    results = {}

    estimator = load_estimator("absolute_pose", conf["estimator"])(conf)
    data_ = {
        "m_kpts0": pts0,
        "m_kpts1": pts1,
        "m_kpts0_w": pts0_world,
        "camera1": data["view1"]["camera"][0],
    }
    est = estimator(data_)

    if not est["success"]:
        results["abs_pose_error_t"] = float("inf")
        results["abs_pose_error_R"] = float("inf")
        results["ransac_inl"] = 0
        results["ransac_inl%"] = 0
    else:
        # R, t, inl = ret
        M = est["M_0to1"]
        inl = est["inliers"].numpy()
        t_error, r_error = absolute_pose_error(T_gt, M.R, M.t, est["offset"])
        results["abs_pose_error_t"] = t_error
        results["abs_pose_error_R"] = r_error
        results["ransac_inl"] = np.sum(inl)
        results["ransac_inl%"] = np.mean(inl)

    return results


def eval_homography_robust(data, pred, conf):
    H_gt = data["H_0to1"]
    if H_gt.ndim > 2:
        return eval_per_batch_item(data, pred, eval_relative_pose_robust, conf)

    estimator = load_estimator("homography", conf["estimator"])(conf)

    data_ = {}
    if "keypoints0" in pred:
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0, scores0 = pred["matches0"], pred["matching_scores0"]
        pts0, pts1, _ = get_matches_scores(kp0, kp1, m0, scores0)
        data_["m_kpts0"] = pts0
        data_["m_kpts1"] = pts1
    if "lines0" in pred:
        if "orig_lines0" in pred:
            lines0 = pred["orig_lines0"]
            lines1 = pred["orig_lines1"]
        else:
            lines0 = pred["lines0"]
            lines1 = pred["lines1"]
        m_lines0, m_lines1, _ = get_matches_scores(
            lines0, lines1, pred["line_matches0"], pred["line_matching_scores0"]
        )
        data_["m_lines0"] = m_lines0
        data_["m_lines1"] = m_lines1

    est = estimator(data_)
    if est["success"]:
        M = est["M_0to1"]
        error_r = homography_corner_error(M, H_gt, data["view0"]["image_size"]).item()
    else:
        error_r = float("inf")

    results = {}
    results["H_error_ransac"] = error_r
    if "inliers" in est:
        inl = est["inliers"]
        results["ransac_inl"] = inl.float().sum().item()
        results["ransac_inl%"] = inl.float().sum().item() / max(len(inl), 1)

    return results


def eval_homography_dlt(data, pred):
    H_gt = data["H_0to1"]
    H_inf = torch.ones_like(H_gt) * float("inf")

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
    scores = scores.to(pts0)
    results = {}
    try:
        if H_gt.ndim == 2:
            pts0, pts1, scores = pts0[None], pts1[None], scores[None]
        h_dlt = find_homography_dlt(pts0, pts1, scores)
        if H_gt.ndim == 2:
            h_dlt = h_dlt[0]
    except AssertionError:
        h_dlt = H_inf

    error_dlt = homography_corner_error(h_dlt, H_gt, data["view0"]["image_size"])
    results["H_error_dlt"] = error_dlt.item()
    return results


def eval_poses(pose_results, auc_ths, key, unit="°"):
    pose_aucs = {}
    best_th = -1
    for th, results_i in pose_results.items():
        pose_aucs[th] = AUCMetric(auc_ths, results_i[key]).compute()
    mAAs = {k: np.mean(v) for k, v in pose_aucs.items()}
    best_th = max(mAAs, key=mAAs.get)

    if len(pose_aucs) > -1:
        print("Tested ransac setup with following results:")
        print("AUC", pose_aucs)
        print("mAA", mAAs)
        print("best threshold =", best_th)

    summaries = {}

    for i, ath in enumerate(auc_ths):
        summaries[f"{key}@{ath}{unit}"] = pose_aucs[best_th][i]
    summaries[f"{key}_mAA"] = mAAs[best_th]

    for k, v in pose_results[best_th].items():
        arr = np.array(v)
        if not np.issubdtype(np.array(v).dtype, np.number):
            continue
        if k.startswith("abs_pose_error"):
            # Calculate mean only of the estimations with inliers
            nz = np.nonzero(np.array(pose_results[best_th]["ransac_inl"]))
            summaries[f"m{k}"] = round(np.mean(arr[nz]), 3)
            # Percentage of nonzero elements in the inliers
            summaries[f"ransac_success_perc"] = round(nz[0].shape[0] / arr.shape[0], 3)
        else:
            summaries[f"m{k}"] = round(np.median(arr), 3)
        
    return summaries, best_th
