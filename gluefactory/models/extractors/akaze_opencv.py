# Additional classical methods

from .base_opencv_extractor import BaseOpencvExtractor
import cv2


class AKAZE(BaseOpencvExtractor):

    def _init(self, conf):
        self.extractor = cv2.AKAZE_create(
            max_points=self.conf.max_num_keypoints)
