# Additional classical methods

import cv2 as cv
from .base_opencv_matcher import BaseOpencvMatcher


class FLANN(BaseOpencvMatcher):
    default_conf = {
        "checks": 50,
    }

    def _init(self, conf):

        # Initialize parameters for sift and orb according to
        # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
        if self.conf.feature == "sift":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(
                algorithm=FLANN_INDEX_KDTREE,
                trees=5)
        elif self.conf.feature == "orb":
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1)
        elif self.conf.feature == "akaze":
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH)
        else:
            self.raise_feature_error()

        self.matcher = cv.FlannBasedMatcher(
            index_params,
            dict(checks=self.conf.checks))
