import cv2
import numpy as np

from .ann_label import Label
from .ann_box import Box


class SegmentationMask:


    def __init__(self, mask: np.ndarray):
        self.mask = mask


    def xx(self):
        binary_mask = cv2.threshold(self.mask, 127, 255, cv2.THRESH_BINARY)[1]
        r, region_map = cv2.connectedComponents(self.mask)