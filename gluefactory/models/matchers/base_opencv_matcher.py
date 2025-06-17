# Additional classical methods

import torch
from ..base_model import BaseModel


def ratio_test(all_matches, th):
    # Apply ratio test for determining good matches
    # as proposed by Lowe 2004
    good_matches = [m[0] for m in all_matches if
                    ((len(m) == 2) and (m[0].distance < th*m[1].distance) or
                     (len(m) == 1))]
    return good_matches


class BaseOpencvMatcher(BaseModel):
    default_conf = {
        "ratio_test": True,
        "ratio_thresh": 0.7,
        "mutual_check": False,
        "checks": 50,
        "feature": None,
    }

    required_data_keys = ["descriptors0", "descriptors1"]

    def raise_feature_error(self):
        features = ["orb", "sift", "akaze"]
        raise ValueError(
            "Type of features must be specified as one of: "
            + f"{','.join(features)}."
        )

    def _forward(self, data):

        desc0 = data["descriptors0"].detach().cpu().numpy().squeeze(0)
        desc1 = data["descriptors1"].detach().cpu().numpy().squeeze(0)

        # Ratio test on two best matches
        if self.conf.ratio_test and not self.conf.mutual_check:
            all_matches = self.matcher.knnMatch(desc0, desc1, k=2)
            good_matches = ratio_test(all_matches, self.conf.ratio_thresh)
        # Best match
        elif self.conf.mutual_check or not self.conf.ratio_test:
            good_matches = self.matcher.match(desc0, desc1)
        else:
            raise ValueError(
                f"Matcher settings ratio test: {self.conf.ratio_test} and "
                + f"mutual check: {self.conf.mutual_check} are contradicting."
            )

        matches0 = -1 * torch.ones(
            data["descriptors0"].shape[:-1], dtype=torch.int)
        matches1 = -1 * torch.ones(
            data["descriptors1"].shape[:-1], dtype=torch.int)
        for m in good_matches:
            matches0[..., m.queryIdx] = m.trainIdx
            matches1[..., m.trainIdx] = m.queryIdx

        return {
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": (matches0 > -1).float(),
            "matching_scores1": (matches1 > -1).float()
        }

    def loss(self, pred, data):
        raise NotImplementedError
