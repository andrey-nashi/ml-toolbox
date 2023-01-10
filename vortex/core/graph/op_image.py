import cv2
import numpy as np
import scipy
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

        output = self.apply(op_input, runtime_width, runtime_height, runtime_interpolation, runtime_border_mode, runtime_value, runtime_eps)
        return output

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

class ImageOperationNormalize(ImageOperation):

    def __init__(self, mean: list = None, std: list = None, max_pixel_value: int = None, axis: list = [0,1]):
        super().__init__(OperationTypes.OP_TYPE_NP_IMAGE, OperationTypes.OP_TYPE_NP_IMAGE)
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value
        self.axis = axis

    def execute(self, op_input: np.ndarray, mean: list = None, std: list = None, max_pixel_value: int = None, axis: list = [0,1]):
        if mean is not None: runtime_mean = mean
        else: runtime_mean = self.mean

        if std is not None: runtime_std = std
        else: runtime_std = self.std

        if max_pixel_value is not None: runtime_max_pixel_value = max_pixel_value
        else: runtime_max_pixel_value = self.max_pixel_value

        if axis is not None: runtime_axis = axis
        else: runtime_axis = self.axis

        output = self.apply(op_input, runtime_mean, runtime_std, runtime_max_pixel_value, runtime_axis)
        return output


    @staticmethod
    def apply(image: np.ndarray, mean: list = None, std: list = None, max_pixel_value: int = None, axis: list = [0,1]):
        mean = np.array(mean, dtype=np.float32) if mean is not None else np.mean(image, axis=axis)
        std = np.array(std, dtype=np.float32) if std is not None else np.std(image, axis=axis)
        """
        Normalize the input image using the mean and standard deviation
        parameters and scale it to have the max value equal to the one
        given as the parameter.
        :param img: input image as a numpy array
        :param mean: mean vector (same number of elements as in image channels)
        :param std: standard deviation vector (same number of elements as in image channels)
        :param max_pixel_value: additional scaling
        :return: a normalized image, each element in range [?,?] with mean at 0
        """

        if not np.isscalar(std):
            std[std == 0] = np.finfo(float).eps
        elif std == 0:
            std = np.finfo(float).eps

        if max_pixel_value is not None:
            if max_pixel_value == -1:
                if image.dtype in (np.uint8, np.uint16):
                    max_pixel_value = np.iinfo(image.dtype).max
                else:
                    raise ValueError(f'max_pixel_value can be -1 only for uint8/uint16 images. <{image.dtype}> was given.')
            mean *= max_pixel_value
            std *= max_pixel_value

        with np.errstate(all='ignore'):
            denominator = np.reciprocal(std, dtype=np.float32)
            img = image.astype(np.float32)
            img -= mean
            img *= denominator
        return img

class ImageOperationNormalize01(ImageOperation):

    def __init__(self, min_value: int = None, max_value: int = None, dtype: int = None):
        super().__init__(OperationTypes.OP_TYPE_NP_IMAGE, OperationTypes.OP_TYPE_NP_IMAGE)
        self.min_value = min_value
        self.max_value = max_value
        self.dtype = dtype

    def execute(self, op_input: np.ndarray, min_value: int = None, max_value: int = None, dtype: int = None):
        if min_value is not None: runtime_min_value = min_value
        else: runtime_min_value = self.min_value

        if max_value is not None: runtime_max_value = max_value
        else: runtime_max_value = self.max_value

        if dtype is not None: runtime_dtype = dtype
        else: runtime_dtype = self.dtype

        output = self.apply(op_input, runtime_min_value, runtime_max_value, runtime_dtype)
        return output

    @staticmethod
    def apply(image: np.ndarray, min_value: int = None, max_value: int = None, dtype: int = None):
        """
        Normalize the input image so that each pixel will be in range [0,1].
        By default min and max are calculated from the given image.
        Set min=0, and max=255 for standard transformation of images to [0,1]
        :param img: input image as a numpy array
        :param min: min value of the image pixels
        :param max: max value of the image pixels
        :return: a normalized image, each element in range [0,1]
        """
        if min_value is None: min_value = np.min(image)
        if max_value is None: max_value = np.max(image)
        if max_value == min_value: return np.clip(image, 0, 1).astype(image.dtype)
        out = (image - min_value) / (max_value - min_value)
        if dtype is not None: out = out.astype(dtype)
        return out

class ImageOperationNormalizeRange(ImageOperation):

    def __init__(self, min_value_old: int = None, max_value_old: int = None,  min_value_new: int = None, max_value_new: int = None):
        super().__init__(OperationTypes.OP_TYPE_NP_IMAGE, OperationTypes.OP_TYPE_NP_IMAGE)
        self.min_value_old = min_value_old
        self.max_value_old = max_value_old
        self.min_value_new = min_value_new
        self.max_value_new = max_value_new

    def execute(self, op_input: np.ndarray, min_value_old: int = None, max_value_old: int = None,  min_value_new: int = None, max_value_new: int = None):
        if min_value_old is not None: runtime_min_value_old = min_value_old
        else: runtime_min_value_old = self.min_value_old

        if max_value_old is not None: runtime_max_value_old = max_value_old
        else: runtime_max_value_old = self.max_value_old

        if min_value_new is not None: runtime_min_value_new = min_value_new
        else: runtime_min_value_new = self.min_value_new

        if max_value_new is not None: runtime_max_value_new = max_value_new
        else: runtime_max_value_new = self.max_value_new

        output = self.apply(op_input, runtime_min_value_old, runtime_max_value_old, runtime_min_value_new, runtime_max_value_new)
        return output

    @staticmethod
    def apply(image: np.ndarray, min_value_old: int = None, max_value_old: int = None,  min_value_new: int = None, max_value_new: int = None):
        if np.isnan(min_value_new): min_value_new  = 0
        assert min_value_new <= max_value_new, (min_value_new, max_value_new)
        if min_value_old is None: min_value_old = np.min(image)
        if max_value_old is None: max_value_old = np.max(image)
        assert min_value_old <= max_value_old, (min_value_old, max_value_old)
        if max_value_old == min_value_old:
            if max_value_old <= 0: return (np.ones_like(image) * min_value_new).astype(image.dtype)
            else: return (np.ones_like(image) * max_value_new).astype(image.dtype)

        # ---- Make range [0, 1]
        image_01 = (image - min_value_old) / (max_value_old - min_value_old)
        # ---- Make range [new_min, new_max]
        output = (image_01 * (max_value_new - min_value_new)) + min_value_new

        return output

class ImageOperationRotate(ImageOperation):

    def __init__(self, angle: float, interpolation: int = cv2.INTER_LINEAR, reshape: bool = False):
        super().__init__(OperationTypes.OP_TYPE_NP_IMAGE, OperationTypes.OP_TYPE_NP_IMAGE)
        self.angle = angle
        self.interpolation = interpolation
        self.reshape = reshape


    def execute(self, op_input: np.ndarray, angle: float, interpolation: int = cv2.INTER_LINEAR, reshape: bool = False):
        if angle is not None: runtime_angle = angle
        else: runtime_angle = self.angle

        if interpolation is not None: runtime_interpolation = interpolation
        else: runtime_interpolation = self.interpolation

        if reshape is not None: runtime_reshape = reshape
        else: runtime_reshape = self.reshape

        output = self.apply(op_input, runtime_angle, runtime_interpolation, runtime_reshape)
        return output

    @staticmethod
    def apply(image: np.ndarray, angle: float, interpolation: int = cv2.INTER_LINEAR, reshape: bool = False):
        output = scipy.ndimage.rotate(image, angle, mode="constant", cval=np.min(image), reshape=reshape)
        return output

class ImageOperationShift(ImageOperation):

    def __init__(self):
        super().__init__(OperationTypes.OP_TYPE_NP_IMAGE, OperationTypes.OP_TYPE_NP_IMAGE)