# Additional classical methods
# - Base class for opencv feature detection and description 
# - Code based on models/extractors/sift.py extractor from gluefactory (https://github.com/cvg/glue-factory)

import numpy as np
import torch
from kornia.color import rgb_to_grayscale

from ..base_model import BaseModel
from .sift import run_opencv_sift as run_opencv


class BaseOpencvExtractor(BaseModel):
    default_conf = {
        "max_num_keypoints": 4096,
        "backend": "opencv",
    }

    required_data_keys = ["image", "mask"]

    def extract_single_image(self, image: torch.Tensor, mask=None):
        image_np = image.cpu().numpy().squeeze(0)
        if mask is not None:
            mask = mask.cpu().numpy().squeeze(0).astype(np.uint8)
        keypoints, scores, scales, angles, descriptors = run_opencv(
            self.extractor, (image_np * 255.0).astype(np.uint8), mask=mask
        )
        pred = {
            "keypoints": keypoints,
            "scales": scales,
            "oris": angles,
            "descriptors": descriptors,
        }
        if scores is not None:
            pred["keypoint_scores"] = scores

        pred = {k: torch.from_numpy(v) for k, v in pred.items()}
            
        return pred
    
    
    def _forward(self, data: dict) -> dict:
        image = data["image"]
        mask = data["mask"]
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)
        device = image.device
        image = image.cpu()
        pred = []
        for k in range(len(image)):
            img = image[k]
            if "image_size" in data.keys():
                # avoid extracting points in padded areas
                w, h = data["image_size"][k]
                img = img[:, :h, :w]
            p = self.extract_single_image(img, mask=mask)
            pred.append(p)
        pred = {k: torch.stack([p[k] for p in pred], 0).to(device) for k in pred[0]}

        return pred

    def loss(self, pred, data):
        raise NotImplementedError
