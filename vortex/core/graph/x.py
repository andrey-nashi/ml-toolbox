

def ti_shift(img: np.ndarray, shift: Sequence[numeric], interpolation: int = cv2.INTER_LINEAR,
             mode: str = 'constant_min', *args, **kwargs) -> np.ndarray:
    if mode == 'constant_min':
        kwargs['cval'] = np.min(img)
        mode = 'constant'
    shift = shift[:img.ndim]
    assert 2 <= len(shift) <= 3, (len(shift), shift)
    if len(shift) != img.ndim:
        if img.ndim < 2 or img.ndim > 3:
            raise ValueError(f'img must have 2 or 3 dimensions (WxH or WxHxCh). Shape <{img.shape}> was passed.')
        if len(shift) < img.ndim:
            shift = [*shift, 0]
    if -1 < shift[0] < 1:
        shift = [int(shift[i] * img.shape[i]) for i in range(img.ndim)]
    img = ndimage.shift(img, shift, order=interpolation, mode=mode, *args, **kwargs)
    return img


def ti_flipv(img: np.ndarray) -> np.ndarray:
    """
    Flip the input image vertically
    :param img: input image as a numpy array
    :return: flipped image
    """
    return np.flipud(img)


def ti_fliph(img: np.ndarray) -> np.ndarray:
    """
    Flip the input image horizontally
    :param img: input image as a numpy array
    :return: flipped image
    """
    return np.fliplr(img)


def ti_scale(img: np.ndarray, scale: float, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Scale the input image by the given parameter
    :param img: input image as a numpy array
    :param scale: scale factor
    :param interpolation: interpolation method (cv2.INTER_LINEAR by default)
    :return: scaled image as a numpyt array
    """
    height, width = img.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    return ti_resize(img, new_width, new_height, interpolation)


def ti_crop(img: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int) -> np.ndarray:
    """
    Crop the input image using the provided coordinates
    :param img: input image as a numpy array
    :param x_min: min width of crop
    :param y_min: min height of crop
    :param x_max: max width of crop
    :param y_max: max height of crop
    :return: cropped image as a numpy array
    """
    if img is None:
        return img
    assert 0 <= x_min < x_max and 0 <= y_min < y_max, ((x_min, x_max), (y_min, y_max))
    return img[y_min: y_max, x_min: x_max].copy()


def ti_point_crop(img: np.ndarray, height: int, width: int,
                  crop_width, crop_height, verbose: bool = False) -> np.ndarray:
    """
    Crops an image based on a center point and the crop dimensions.
    Args:
        img: image to be cropped
        height: height coordinate of the center of the output crop
        width: width coordinate of the center of the output crop
        crop_width: width of the output crop
        crop_height: height of the output crop
        verbose: if True it will print a warning in case the calculated crop limits are out of the image boarders.
    Returns:
        Cropped image as a numpy array
    """
    img_height, img_width = img.shape[:2]
    y1 = height - (crop_height // 2)
    x1 = width - (crop_width // 2)

    y1_ = min(max(0, y1), img_height - crop_height)
    x1_ = min(max(0, x1), img_width - crop_width)

    if verbose and ((y1 != y1_) or (x1 != x1_)):
        print('[Warning] (ti_point_crop): Calculated points changed '
              f'from {x1}, {y1} to {x1_}, {y1_} due to boundary limits')
    x1, y1 = x1_, y1_

    y2 = y1 + crop_height
    x2 = x1 + crop_width
    return img[y1:y2, x1:x2]


def ti_center_crop(img: np.ndarray, crop_width, crop_height) -> np.ndarray:
    """
    Crop the center portion of the image with the specified size.
    :param img: input image as a numpy array
    :param crop_width: width of the output crop
    :param crop_height: height of the output crop
    :return: cropped image as a numpy array
    """
    height, width = img.shape[:2]
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    img = img[y1:y2, x1:x2].copy()
    return img


def get_random_crop_coords(height: int, width: int, crop_height: int, crop_width: int, h_start: float, w_start: float):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def ti_random_crop(img: np.ndarray, crop_width, crop_height, h_start=0, w_start=0) -> np.ndarray:
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    img = img[y1:y2, x1:x2]
    return img


def ti_brightness_contrast_adjust(img, alpha=1, beta=0):
    min, max = img.min(), img.max()
    dtype = img.dtype
    img = img.astype("float32")
    if alpha != 1:
        img *= alpha
    if beta != 0:
        img += beta * np.mean(img)
    img = np.clip(img, min, max)
    return img.astype(dtype)


def ti_random_gamma(img: np.ndarray, gamma=1.0):
    dtype = img.dtype
    min_val = img.min()
    max_val = max(MAX_VALUES_BY_DTYPE.get(dtype, 1.0), img.max())

    if img.dtype == np.uint8:
        table = (np.arange(0, 256.0 / 255, 1.0 / 255) ** gamma) * 255
        img = cv2.LUT(img, table.astype(np.uint8))
    elif img.min() >= 0 and img.max() <= 1:
        img = np.power(img, gamma)
    else:
        img = ti_normalize_01(img)
        img = np.power(img, gamma)
        img = ti_normalize_range(img, min=0, max=1, new_min=min_val, new_max=max_val, dtype=dtype)
    return img


def ti_solarize(img: np.ndarray, threshold=128) -> np.ndarray:
    """
    Invert all pixel values above a threshold.
    :param img: input image as numpy array
    :param threshold: all pixels above this greyscale level are inverted.
    :return: solarized image
    """

    dtype = img.dtype
    max_val = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)

    if dtype == np.dtype("uint8"):
        lut = [(i if i < threshold else max_val - i) for i in range(max_val + 1)]

        prev_shape = img.shape
        img = cv2.LUT(img, np.array(lut, dtype=dtype))

        if len(prev_shape) != len(img.shape):
            img = np.expand_dims(img, -1)
        return img

    result_img = img.copy()
    cond = img >= threshold
    result_img[cond] = max_val - result_img[cond]
    return result_img


def ti_fancy_pca(img: np.ndarray, alpha: numeric = 0.1) -> np.ndarray:
    """
    Perform 'Fancy PCA' augmentation from:
    http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    :param img: input image as numpy array with ranges [0,255] for each pixel
    :param alpha:  how much to perturb/scale the eigen vecs and vals the paper used std=0.1
    :return: numpy image-like array as float range(0, 1)
    """
    if (len(img.shape) == 2) or (len(img.shape) == 3 and img.shape[-1] == 1) or img.dtype != np.uint8:
        raise TypeError("Image must be RGB image in uint8 format.")

    orig_img = img.astype(float).copy()

    img = img / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # img_rs shape (640000, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    # alpha = np.random.normal(0, alpha_std)

    # broad cast to speed things up
    m2[:, 0] = alpha * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.asarray(m1) * np.asarray(m2)

    for idx in range(3):  # RGB
        orig_img[..., idx] += add_vect[idx] * 255

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_img /= 255.0
    orig_img = np.clip(orig_img, 0.0, 255.0)

    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)

    return orig_img


def ti_equalize(img, n_bins: int = 512, renormalize: bool = False) -> np.ndarray:
    dtype, min, max = img.dtype, img.min(), img.max()
    from skimage import exposure
    """
    Use histogram equalization on the given image
    :param img: input image as numpy array
    :param n_bins: number of bins to be used for equalization
    :return: equalized image
    """
    try:
        equalized = cv2.equalizeHist(img)
    except:
        if img.ndim == 2:
            equalized = exposure.equalize_hist(img, nbins=n_bins)
        elif img.ndim == 3:
            equalized = np.stack([exposure.equalize_hist(img[:, :, i], nbins=n_bins)
                                  for i in range(img.shape[-1])], axis=-1)
        else:
            raise ValueError(f"Image dimension must be 2 or 3. Input dimension = {img.ndim}.")
    if renormalize:
        equalized = ti_normalize_range(equalized, min, max, dtype=dtype)
    return equalized


def ti_rescale_intensity(img, low_perc: int = 2, high_perc: int = 98):
    from skimage import exposure
    """
    Return image after stretching or shrinking its intensity levels
    Args:
        img: input image as numpy array
        low_perc: low percentile of values to be used as min. (default 2%)
        high_perc: high percentile of values to be used as max. (default 98%)

    Returns: intesity rescaled image (close to original range)
    """
    low, high = np.percentile(img, (low_perc, high_perc))
    return exposure.rescale_intensity(img, in_range=(low, high))


def ti_clahe(img: np.ndarray, clip_limit=2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    # if img.dtype != np.uint8:
    #     raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if img.ndim == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img


@to_fixed
def ti_pad(img: np.ndarray, min_height, min_width, border_mode=0, value=None) -> np.ndarray:
    """
    Pad the given image with constant values or reflections.
    :param img: input image as a numpy array
    :param min_height: height of the new image
    :param min_width: width of the new image
    :param border_mode: method for padding (by default is cv2.BORDER_CONSTANT)
    :param value: value to pad the borders with
    :return: a padded image as a numpy array
    """
    if border_mode == 0:
        border_mode = cv2.BORDER_CONSTANT
        value = value if value is not None else np.asscalar(np.min(img))
    if border_mode == 1:
        border_mode = cv2.BORDER_REFLECT_101

    height, width, *ch = img.shape[:3]
    ch = ch[0] if len(ch) > 0 else 0
    if not isinstance(value, (list, tuple)) and ch > 0:
        value = [value] * ch

    if height < min_height:
        h_pad_top = int((min_height - height) / 2.0)
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < min_width:
        w_pad_left = int((min_width - width) / 2.0)
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0
    img = cv2.copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode, value=value)

    if img.shape[:2] != (max(min_height, height), max(min_width, width)):
        raise RuntimeError(f'Invalid result shape. Got: {img.shape[:2]}. Expected: '
                           f'{(max(min_height, height), max(min_width, width))}')

    if ch == 0:
        return img
    if img.ndim == 2:
        img = np.stack([img] * ch, -1)
    assert img.shape[2] == ch, (img.shape, ch)
    return img


def ti_unpad(img: np.ndarray, height, width) -> np.ndarray:
    img_height, img_width = img.shape[:2]
    assert img_width >= width and img_height >= height, ((img_width, width), (img_height, height))
    # height
    h_diff = img_height - height
    h_min = int(h_diff / 2)
    h_max = height + h_min
    # width
    w_diff = img_width - width
    w_min = int(w_diff / 2)
    w_max = width + w_min
    return img[h_min: h_max, w_min: w_max]


def ti_bgr2rgb(img):
    return img[..., ::-1].copy()


def ti_rgb2label(img: np.ndarray, color_table=None, one_hot_encoding=False) -> Tuple[np.ndarray, Union[dict, Any]]:
    """
    Transform and RGB image into a map using the provided color table.
    In other words each [R,G,B] pixel will be swapped to an integer number
    :param img: an opencv RGB/BGR image
    :param color_table: a table of colors that represents classes [R,G,B] - int
    :param one_hot_encoding: is carry out one hot encoding
    :return: map as a numpy array
    """

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # ---- Generate table for color encoding if it is not given
    if color_table is None:
        color_table = {val: i for i, val in enumerate(set(tuple(v) for m2d in image for v in m2d))}
    n_labels = len(color_table)

    # ---- Output map, where each pixel is the label of the class
    output = np.ndarray(shape=image.shape[:2], dtype=int)
    output[:, :] = 0

    for rgb, idx in color_table.items():
        output[(image == rgb).all(2)] = idx

    # ---- Execute one hot encoding if the flag is set
    if one_hot_encoding:
        one_hot_labels = np.zeros((image.shape[0], image.shape[1], n_labels))
        for c in range(n_labels):
            one_hot_labels[:, :, c] = (output == c).astype(int)
        output = one_hot_labels

    return output, color_table


def ti_grid_distortion(img: np.ndarray, num_steps: int, xsteps: Sequence[numeric], ysteps: Sequence[numeric],
                       interpolation: int = cv2.INTER_LINEAR, border_mode: int = cv2.BORDER_REFLECT_101,
                       value: int = None) -> np.ndarray:
    """
    Distort the image using the new grid coordinates.
    Reference:  http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    :param img: input image as a numpy array
    :param num_steps: number of steps in the grid
    :param xsteps: x coordinates of the grid as a 1-D array, each element [0,1]
    :param ysteps: y coordinates of the grid as a 1-D array, each element [0,1]
    :param interpolation: interpolation method (cv2.INTER_LINEAR by default)
    :param border_mode: border flag, by default borders will be reflected (cv2.BORDER_REFLECT_101 by default)
    :param value: for the cv2.BORDER_CONSTANT the value of the constant
    """
    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        x = idx * x_step
        start = int(x)
        end = int(x) + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx in range(num_steps + 1):
        y = idx * y_step
        start = int(y)
        end = int(y) + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    remap_fn = cv2.remap(img, map1=map_x, map2=map_y, interpolation=interpolation,
                         borderMode=border_mode, borderValue=value)
    return remap_fn


def ti_noise_gauss(img: np.ndarray, var_min: numeric = 10, var_max: numeric = 50, mean: numeric = 0):
    """
    Apply gaussian noise to the given image
    :param img: input image as numpy array
    :param var_min: minimum value of the gaussian noise
    :param var_max: max value of the gaussian noise
    :param mean: mean value of the gaussian noise
    :return: noisy image as a numpy array
    """
    dtype = img.dtype
    min, max = img.min(), img.max()
    var = random.uniform(var_min, var_max)
    sigma = var ** 0.5
    random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

    gauss = random_state.normal(mean, sigma, img.shape)
    img = np.clip((img + gauss), min, max).astype(dtype)
    return img


def ti_noise_iso(img: np.ndarray, color_shift: numeric = 0.05, intensity: numeric = 0.5) -> np.ndarray:
    """
    Apply poisson noise to image to simulate camera sensor noise.
    :param img: input image as numpy array (only RGB is supported)
    :param color_shift:
    :param intensity: multiplication factor for noise values. Values of ~0.5 are produce noticeable,
                      yet acceptable level of noise.
    :return: noisy image as a numpy array
    """

    assert img.dtype in (np.uint8, np.uint16), img.dtype
    assert img.ndim == 3, img.shape
    max_num = np.iinfo(img.dtype).max
    random_state = np.random.RandomState(42)

    one_over_max = float(1.0 / max_num)
    image = np.multiply(img, one_over_max, dtype=np.float32)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    _, stddev = cv2.meanStdDev(hls)

    luminance_noise = random_state.poisson(stddev[1] * intensity * max_num, size=hls.shape[:2])
    color_noise = random_state.normal(0, color_shift * 360 * intensity, size=hls.shape[:2])

    hue = hls[..., 0]
    hue += color_noise
    hue[hue < 0] += 360
    hue[hue > 360] -= 360

    luminance = hls[..., 1]
    luminance += (luminance_noise / max_num) * (1.0 - luminance)

    image = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB) * max_num

    return image.astype(img.dtype)


def ti_permute(img: np.ndarray, w_parts: int, h_parts: int,
               w_permutation: Sequence[int], h_permutation: Sequence[int]):
    def permute_axis(img, parts, permutation, axis):
        splits = np.array_split(img, parts, axis=axis)
        splits = [splits[p] for p in permutation]
        permuted = np.concatenate(splits, axis=axis)
        return permuted

    h_permuted = permute_axis(img, h_parts, h_permutation, axis=0)
    hw_permuted = permute_axis(h_permuted, w_parts, w_permutation, axis=1)
    return hw_permuted


def ti_rec_replace(img: np.ndarray, value, w_min, h_min, w_max, h_max):
    if 0 <= w_min <= 1:
        assert all(0 <= d <= 1 for d in (h_min, w_max, h_max)), (w_min, h_min, w_max, h_max)
        h, w = img.shape[:2]
        h_min, h_max = int(h_min * h), int(h_max * h)
        w_min, w_max = int(w_min * w), int(w_max * w)
    dtype = img.dtype
    img = img.astype(type(value))
    img = img.copy()
    img[h_min: h_max, w_min: w_max] = value
    img = img.astype(dtype) if np.all(img == img.astype(dtype)) else img
    return img
