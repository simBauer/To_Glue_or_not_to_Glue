# Additional classical methods

import cv2 as cv
from .base_opencv_matcher import BaseOpencvMatcher


class NN(BaseOpencvMatcher):

    def _init(self, conf):
        if self.conf.feature in ["orb", "akaze"]:
            DISTANCE_METRIC = cv.NORM_HAMMING
        elif self.conf.feature == "sift":
            DISTANCE_METRIC = cv.NORM_L2
        else:
            self.raise_feature_error()

        if self.conf.ratio_test:
            cross_check = False
        else:
            cross_check = True

        self.matcher = cv.BFMatcher(
            normType=DISTANCE_METRIC,
            crossCheck=cross_check)
