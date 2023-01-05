import cv2
import numpy as np
from .op_base import *

class ImageOperationClip(ImageOperation):

    def __init__(self, min_value: int = None, max_value: int = None):
        super().__init__(OperationTypes.OP_TYPE_NP_IMAGE, OperationTypes.OP_TYPE_NP_IMAGE)
        self.min_value = min_value
        self.max_value = max_value

    def execute(self, op_input: np.ndarray, min_value: int = None, max_value: int = None) -> np.ndarray:
        if min_value is None: runtime_min_value = self.min_value
        else: runtime_min_value = min_value

        if max_value is None: runtime_max_value = self.max_value
        else: runtime_max_value = max_value

        if runtime_min_value is None or runtime_max_value is None:
            return op_input

        output = self.apply(op_input, runtime_min_value, runtime_max_value)
        return output

    @staticmethod
    def apply(image: np.ndarray, min_value: int, max_value: int):
        return np.clip(image, min_value, max_value)

class ImageOperationResize(ImageOperation):

    def __init__(self, width: int = None, height: int = None, interpolation: int = cv2.INTER_LINEAR):
        super().__init__(OperationTypes.OP_TYPE_NP_IMAGE, OperationTypes.OP_TYPE_NP_IMAGE)
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def execute(self, op_input: np.ndarray, width: int = None, height: int = None):
        if width is None: runtime_width = self.width
        else: runtime_width = width

        if height is None: runtime_height = self.height
        else: runtime_height = height

        if runtime_width is None or runtime_height is None:
            return op_input

        output = self.apply(op_input, runtime_width, runtime_height)
        return output

    @staticmethod
    def apply(image: np.ndarray, width: int, height: int, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        output = cv2.resize(image, (width, height), interpolation=interpolation)
        return output

class ImageOperationResizePad(ImageOperation):

    def __init__(self, width: int = None, height: int = None, interpolation: int = cv2.INTER_LINEAR, border_mode: int =0,
                 value: int = 0, eps: float = 1e-3):
        super().__init__(OperationTypes.OP_TYPE_NP_IMAGE, OperationTypes.OP_TYPE_NP_IMAGE)
        self.width = width
        self.height = height
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.eps = eps

    def execute(self, op_input: np.ndarray, width: int = None, height: int = None, interpolation: int = None,
                border_mode: int = None, value: int = None, eps: float = None):
        if width is not None: runtime_width = width
        else: runtime_width = self.width

        if height is not None: runtime_height = height
        else: runtime_height = self.height

        if interpolation is not None: runtime_interpolation = interpolation
        else: runtime_interpolation = self.interpolation

        if border_mode is not None: runtime_border_mode = border_mode
        else: runtime_border_mode = self.border_mode

        if value is not None: runtime_value = value
        else: runtime_value = self.value

        if eps is not None: runtime_eps = eps
        else: runtime_eps = self.eps

        if runtime_width is None or runtime_height is None:
            return op_input

        self.apply(op_input, runtime_width, runtime_height, runtime_interpolation, runtime_border_mode, runtime_value, runtime_eps)


    @staticmethod
    def apply(image: np.ndarray, width: int, height: int, interpolation: int = cv2.INTER_LINEAR, border_mode: int =0,
                 value: int = 0, eps: float = 1e-3) -> np.ndarray:
        im_height, im_width = image.shape[:2]
        im_h2w = im_height / im_width
        target_h2w = height / width
        if abs(im_h2w - target_h2w) < eps:
            return ImageOperationResize.apply(image, width=width, height=height, interpolation=interpolation)
        if im_h2w > target_h2w:
            h = height
            w = int(h / im_h2w)
        else:
            w = width
            h = int(w * im_h2w)
        output = ImageOperationResize.apply(image, width=w, height=h, interpolation=interpolation)
        output = ti_pad(output, min_height=height, min_width=width, border_mode=border_mode, value=value)
        return output

class ImageOperationGaussianBlur(ImageOperation):

    def __init__(self, kernel_size: int = 3, sigma: int = 0):
        super().__init__(OperationTypes.OP_TYPE_NP_IMAGE, OperationTypes.OP_TYPE_NP_IMAGE)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def execute(self, op_input: np.ndarray, kernel_size: int = None, sigma: int = None):
        if kernel_size is not None: runtime_kernel_size = kernel_size
        else: runtime_kernel_size = self.kernel_size

        if sigma is not None: runtime_sigma = sigma
        else: runtime_sigma = self.sigma

        output = self.apply(op_input, runtime_kernel_size, runtime_sigma)
        return output

    @staticmethod
    def apply(image: np.ndarray, kernel_size: int = 3, sigma: int = 0) -> np.ndarray:
        min_value, max_value = image.min(), image.max()
        output = cv2.GaussianBlur(image, ksize=(kernel_size, kernel_size), sigmaX=sigma)
        output = np.clip(output, min_value, max_value)
        return output

class ImageOperationGrayscale(ImageOperation):

    def __init__(self):
        super().__init__(OperationTypes.OP_TYPE_NP_IMAGE, OperationTypes.OP_TYPE_NP_IMAGE)

    def execute(self, op_input: np.ndarray):
        output = self.apply(op_input)
        return output

    @staticmethod
    def apply(image: np.ndarray):
        if len(image.shape) == 3: return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else: return image

class ImageOperationGray2Color(ImageOperation):

    def __init__(self):
        super().__init__(OperationTypes.OP_TYPE_NP_IMAGE, OperationTypes.OP_TYPE_NP_IMAGE)

    def execute(self, op_input: np.ndarray):
        output = self.apply(op_input)
        return output

    @staticmethod
    def apply(image: np.ndarray):
        """
        Takes an image of shape either (WxH) or (WxHx1) or (WxHx3) and returns an image of shape (WxHx3).
        Raises Value error if the image shape is not compatible.
        :param image: input image as a numpy array
        :return: a 3-channel image
        """
        if image.ndim <= 1 or image.ndim > 3:
            raise ValueError(f'image must be of shape WxH or WxHx1. {image.shape} was given.')
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if image.shape[-1] == 3:
            return image
        if image.shape[-1] == 4:
            return image[..., :3]
        if image.shape[-1] != 1:
            raise ValueError(f'image must be of shape WxH, WxHx1, WxHx3 or WxHx4. {image.shape} was given.')
        return np.tile(image, (1, 1, 3))


