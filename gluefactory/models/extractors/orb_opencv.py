# Additional classical methods

from .base_opencv_extractor import BaseOpencvExtractor
import cv2


class ORB(BaseOpencvExtractor):

    def _init(self, conf):
        self.extractor = cv2.ORB_create(
            nfeatures=self.conf.max_num_keypoints)
