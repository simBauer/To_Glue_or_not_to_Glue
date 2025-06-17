# Code for evaluation of TUM dataset
# - Based on robust_estimators/relative_pose/opencv.py from gluefactory (https://github.com/cvg/glue-factory)

import cv2
import torch
import numpy as np
from ...geometry.wrappers import Pose
from ..base_estimator import BaseEstimator


class OpenCVAbsolutePoseEstimator(BaseEstimator):
    default_conf = {"ransac_th": 10.0}

    required_data_keys = ["m_kpts0", "m_kpts0_w", "m_kpts1", "camera1"]

    def _init(self, conf):
        self.solver = "ransac"

    def _forward(self, data):
        kpts0_w, kpts1 = data["m_kpts0_w"], data["m_kpts1"]
        camera1 = data["camera1"]
        M, inliers = None, np.zeros_like(kpts1[:, 0])
        offset = None
        th = self.conf["ransac_th"]
        
        if len(kpts1) >= 6:

            kpts0_w = kpts0_w.cpu().detach().numpy()
            pts1 = kpts1.cpu().detach().numpy()
            K = camera1.calibration_matrix().cpu().numpy()

            # Normalize coordinates for estimation
            offset = np.mean(kpts0_w, axis=0).reshape((1, 3))
            pts0 = kpts0_w - offset

            # TODO: Include distortion coefficients
            retval, rvec, tvec, inl_idx = cv2.solvePnPRansac(
                objectPoints=pts0,
                imagePoints=pts1,
                cameraMatrix=K,
                distCoeffs=None,
                reprojectionError=th
            )
            inliers = np.zeros((pts1.shape[0]))
            inliers[inl_idx] = 1

            if retval:    
                rvec = torch.from_numpy(rvec.flatten()).to(device=kpts1.device).float()
                tvec = torch.from_numpy(tvec.flatten()).to(device=kpts1.device).float()
                M = Pose.from_aa(rvec, tvec)

            offset = torch.from_numpy(offset.flatten()).to(device=kpts1.device).float()

        estimation = {
            "success": M is not None,
            "M_0to1": M if M is not None else Pose.from_4x4mat(torch.eye(4).to(device=kpts1.device)),
            "inliers": torch.from_numpy(inliers).to(device=kpts1.device),
            "offset": offset
        }

        return estimation
