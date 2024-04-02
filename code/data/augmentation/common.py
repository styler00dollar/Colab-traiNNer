"""
31-Dez-21
https://github.com/victorca25/augmennt/blob/master/augmennt/common.py
"""

from glob import glob
from os.path import join as pjoin

import cv2
import numpy as np
from functools import wraps

try:
    from PIL import Image

    pil_available = True
except ImportError:
    pil_available = False

# try:
#     import cv2
#     cv2_available =  True
# except ImportError:
#     cv2_available = False

# BORDERS_MODE
_cv2_str2pad = {
    "constant": 0,  # BORDER_CONSTANT
    "edge": 1,  # BORDER_REPLICATE
    "replicate": 1,  # BORDER_REPLICATE
    "mirror": 2,  # BORDER_REFLECT
    "symmetric": 2,  # BORDER_REFLECT
    "wrap": 3,  # BORDER_WRAP
    "reflect": 4,  # BORDER_REFLECT_101
    "reflect101": 4,  # BORDER_REFLECT_101, BORDER_REFLECT101
    "default": 4,  # BORDER_DEFAULT
    "reflect1": 4,  # BORDER_DEFAULT
    "transparent": 5,  # BORDER_TRANSPARENT
    "isolated": 16,  # BORDER_ISOLATED
}

# INTER_MODE
_cv2_str2interpolation = {
    "nearest": cv2.INTER_NEAREST,
    "NEAREST": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "BILINEAR": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "AREA": cv2.INTER_AREA,
    "bicubic": cv2.INTER_CUBIC,
    "BICUBIC": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "LANCZOS": cv2.INTER_LANCZOS4,
}
_cv2_interpolation2str = {v: k for k, v in _cv2_str2interpolation.items()}


# much faster than iinfo and finfo
MAX_VALUES_BY_DTYPE = {
    np.dtype("int8"): 127,
    np.dtype("uint8"): 255,
    np.dtype("int16"): 32767,
    np.dtype("uint16"): 65535,
    np.dtype("int32"): 2147483647,
    np.dtype("uint32"): 4294967295,
    np.dtype("int64"): 9223372036854775807,
    np.dtype("uint64"): 18446744073709551615,
    np.dtype("float32"): 1.0,
    np.dtype("float64"): 1.0,
}

# TODO: needed?
# MIN_VALUES_BY_DTYPE = {
#     np.dtype("int8"): -128,
#     np.dtype("uint8"): 0,
#     np.dtype("int16"): 32768,
#     np.dtype("uint16"): 0,
#     np.dtype("int32"): 2147483648,
#     np.dtype("uint32"): 0,
#     np.dtype("int64"): 9223372036854775808,
#     np.dtype("uint64"): 0,
#     np.dtype("float32"): -1.0,  # depends on normalization
#     np.dtype("float64"): -1.0,  # depends on normalization
# }


def pil2cv(pil_image):
    open_cv_image = np.array(pil_image)
    if len(open_cv_image.shape) == 2:
        open_cv_image = np.expand_dims(open_cv_image, axis=-1)
    # Convert RGB to BGR
    return open_cv_image[:, :, ::-1].copy()


def cv2pil(open_cv_image):
    if pil_available:
        shape = open_cv_image.shape
        if len(shape) == 3 and shape[-1] == 1:  # len(shape) == 2:
            open_cv_image = np.squeeze(open_cv_image, axis=-1)
        if len(shape) == 3 and shape[-1] == 3:
            # Convert BGR to RGB
            open_cv_image = cv2.cvtColor(open_cv_image.copy(), cv2.COLOR_BGR2RGB)
        return Image.fromarray(open_cv_image)
    raise Exception("PIL not available")


def wrap_cv2_function(func):
    """
    Ensure the image input to the function is a cv2 image
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        if isinstance(img, np.ndarray):
            result = func(img, *args, **kwargs)
        elif pil_available and isinstance(img, Image.Image):
            result = cv2pil(func(pil2cv(img), *args, **kwargs))
        else:
            raise TypeError("Image type not recognized")
        return result

    return wrapped_function


def wrap_pil_function(func):
    """
    Ensure the image input to the function is a pil image
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        if isinstance(img, np.ndarray):
            result = pil2cv(func(cv2pil(img), *args, **kwargs))
        elif pil_available and isinstance(img, Image.Image):
            result = func(img, *args, **kwargs)
        else:
            raise TypeError("Image type not recognized")
        return result

    return wrapped_function


def srgb2linear(srgb, gamma=2.4, th=0.04045):
    """Convert SRGB images to linear RGB color space.
        To use the formulat, values have to be in the 0 to 1 range,
        for that reason srgb must be in range [0,255], uint8 and:
        signal = input / 255 is applied.
    Parameters:
        gamma (float): gamma correction. The default is 2.4, but 2.2
            approximately matches the power law sensitivity of human vision
        th (float): threshold value for formula. The default is 0.04045,
            which is an approximate, exact value is 0.0404482362771082
    """

    a = 0.055
    att = 12.92
    linear = np.float32(srgb) / 255.0

    return np.where(linear <= th, linear / att, np.power((linear + a) / (1 + a), gamma))


def linear2srgb(linear, gamma=2.4, th=0.0031308):
    """Convert linear RGB images to SRGB color space.
    linear must be in range [0,1], float32"""
    a = 0.055
    att = 12.92
    srgb = np.clip(linear.copy(), 0.0, 1.0)

    srgb = np.where(srgb <= th, srgb * att, (1 + a) * np.power(srgb, 1.0 / gamma) - a)

    # return srgb * 255.0
    return np.clip(srgb * 255.0, 0.0, 255).astype(np.uint8)


def wrap_linear_function(func):
    """
    Ensures that arithmetic operations on sRGB images happen
    in linear space
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        input_dtype = img.dtype
        if input_dtype == np.uint8:
            linear = srgb2linear(img)
            result = func(linear, *args, **kwargs)
            result = linear2srgb(result)
        elif input_dtype == np.float32:
            result = func(img, *args, **kwargs)
        return result

    return wrapped_function


def preserve_shape(func):
    """
    Wrapper to preserve shape of the image
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        # numpy reshape:
        result = result.reshape(shape)
        return result

    return wrapped_function


def preserve_type(func):
    """
    Wrapper to preserve type of the image
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        result = func(img, *args, **kwargs)
        return result.astype(dtype)

    return wrapped_function


def preserve_channel_dim(func):
    """
    Preserve dummy channel dimension for grayscale images
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == 3 and shape[-1] == 1 and len(result.shape) == 2:
            result = np.expand_dims(result, axis=-1)
            # result = result[:, :, np.newaxis]
        return result

    return wrapped_function


def get_num_channels(img):
    return img.shape[2] if len(img.shape) == 3 else 1


def _maybe_process_in_chunks(process_fn, **kwargs):
    """
    Wrap OpenCV function to enable processing images with more
    than 4 channels.
    Limitations:
        This wrapper requires image to be the first argument and
        rest must be sent via named arguments.
    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.
    Returns:
        numpy.ndarray: Transformed image.
    """

    @wraps(process_fn)
    def __process_fn(img):
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with
                    # 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


def clip(img, dtype, maxval, minval=0):
    return np.clip(img, minval, maxval).astype(dtype)


def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype, 1.0)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function


def preserve_range_float(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        # if isinstance(img, np.ndarray):
        if type(img).__module__ == np.__name__:
            t_dtype = np.dtype("float32")
            dtype = img.dtype
            if dtype == t_dtype:
                return func(img, *args, **kwargs)

            t_maxval = MAX_VALUES_BY_DTYPE.get(t_dtype)
            maxval = MAX_VALUES_BY_DTYPE.get(dtype)
            if not maxval:
                if np.issubdtype(dtype, np.integer):
                    info = np.iinfo
                elif np.issubdtype(dtype, np.floating):
                    info = np.finfo
                maxval = info(dtype).max
            img = img.astype(t_dtype) * t_maxval / maxval
            return (func(img, *args, **kwargs) * maxval).astype(dtype)
        return func(img, *args, **kwargs)

    return wrapped_function


def add_img_channels(img):
    """fix image with only 2 dimensions (add "channel" dimension (1))"""
    if img.ndim == 2:
        img = np.tile(np.expand_dims(img, axis=2), (1, 1, 3))
    return img


@preserve_shape
def convolve(
    img: np.ndarray,
    kernel: np.ndarray,
    per_channel: bool = False,
    flip_k: bool = False,
    mode: str = "default",
) -> np.ndarray:
    border = _cv2_str2pad[mode]
    if flip_k:
        # cross-correlation to convolution
        kernel = cv2.flip(kernel, -1)
    if per_channel:

        def channel_conv(img, kernel):
            if len(img.shape) < 3:
                img = add_img_channels(img)
            output = []
            for channel_num in range(img.shape[2]):
                output.append(
                    cv2.filter2D(img[:, :, channel_num], ddepth=-1, kernel=kernel),
                    borderType=border,
                )
            return np.squeeze(np.stack(output, -1))

        return channel_conv(img, kernel)
    # else:
    conv_fn = _maybe_process_in_chunks(
        cv2.filter2D, ddepth=-1, kernel=kernel, borderType=border
    )
    return conv_fn(img)


def norm_kernel(kernel: np.ndarray) -> np.ndarray:
    """normalize kernel, so it sums up to 1."""
    if np.sum(kernel) == 0.0:
        # if kernel is empty, return identity kernel (delta impulse)
        if len(kernel.shape) == 3:
            kernel[kernel.shape[0] // 2, kernel.shape[1] // 2, kernel.shape[2] // 2] = 1
        elif len(kernel.shape) == 2:
            kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = 1
        elif len(kernel.shape) == 1:
            kernel[kernel.shape[0] // 2] = 1
        return kernel
    return kernel.astype(np.float32) / np.sum(kernel)


def pad_kernel(kernel: np.ndarray, kernel_size: int, pad_to: int = 0) -> np.ndarray:
    """pad kernel size to desired size, "pad_to" must be odd or zero."""
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel


def fetch_kernels(kernels_path, pattern: str = "", scale=None, kformat: str = "npy"):
    if pattern == "kernelgan":
        # using the modified kernelGAN file structure.
        kernels = glob(pjoin(kernels_path, "*/kernel_x{}.{}".format(scale, kformat)))
        if not kernels:
            # try using the original kernelGAN file structure.
            kernels = glob(
                pjoin(kernels_path, "*/*_kernel_x{}.{}".format(scale, kformat))
            )
        # assert kernels, "No kernels found for scale {} in path {}.".format(scale, kernels_path)
    elif pattern == "matmotion":
        kernels = glob(pjoin(kernels_path, "m_??.{}".format(kformat)))
    else:
        kernels = glob(pjoin(kernels_path, "*.{}".format(kformat)))
    return kernels


def sample(
    img: np.ndarray,
    scale=2,
    sampling: str = "down",
    center: bool = False,
    orig: bool = False,
) -> np.ndarray:
    """S-fold sampler, both for upsampling and downsampling.
    When used for downsample, equivalent to the simplest nearest
    neighbor downsampling. Upsamples images spatial size by
    filling the new pixels with zeros.
    Args:
        img: image to sample
        scale (int or tuple): downsampling scale. If a tuple is
            used, can sample each dimension at different scales.
        sampling: type of sampling, either 'up' or 'down'.
        center: select if the first pixel to sample from each distinct
            scale x scale patch will be the upper left one or the center
            one. The remaining pixels are discarded.
        orig: flag to use as the original downsampling behavior with
            linspace (expects scales =< 1 to downsample), else expects
            scale values >= 1 for either direction.
    """
    input_shape = img.shape
    # by default, if scale-factor is a scalar assume 2d resizing
    # and duplicate it
    if isinstance(scale, (int, float)):
        scale_factor = [scale, scale]
    else:
        scale_factor = scale

    if orig:
        scale_factor = [1 / factor for factor in scale_factor]

    # if needed extend the size of scale-factor list to the size of
    # the input by assigning 1 to all the unspecified scales
    if len(scale_factor) != len(input_shape):
        scale_factor = list(scale_factor)
        scale_factor.extend([1] * (len(input_shape) - len(scale_factor)))

    # dealing with missing output-shape. calculating according to scale-factor
    output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))
    if sampling == "up":
        img_out = np.zeros(
            (output_shape[0], output_shape[1], input_shape[2]), dtype=img.dtype
        )

    # can select specific starting points to subsample
    if center:
        st = [(sf - 1) // 2 for sf in scale_factor]
    else:
        st = [0 for sf in scale_factor]

    if sampling == "up":
        # then upsample and return
        img_out[st[0] :: scale_factor[0], st[1] :: scale_factor[1], ...] = img
        return img_out

    # then subsample and return
    if orig:
        return img[
            np.round(
                np.linspace(st[0], img.shape[0] - 1 / scale_factor[0], output_shape[0])
            ).astype(int)[:, None],
            np.round(
                np.linspace(st[1], img.shape[1] - 1 / scale_factor[1], output_shape[1])
            ).astype(int),
            :,
        ]

    return img[st[0] :: scale_factor[0], st[1] :: scale_factor[1], ...]


def split_channels(image: np.ndarray) -> list:
    c = image.shape[2]
    return [image[..., ch] for ch in range(c)]


def merge_channels(channels: list, axis: int = -1) -> np.ndarray:
    return np.stack(channels, axis=axis)


def polar2z(r: np.ndarray, θ: np.ndarray) -> np.ndarray:
    """
    Converts a list of radii and angles (radians) and
    into a corresponding list of complex numbers x + yi.
    Arguments:
        r (np.ndarray): radius
        θ (np.ndarray): angle
    Returns:
        [np.ndarray]: list of complex numbers r e^(i theta) as x + iy
    """
    return r * np.exp(1j * θ)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, (list, tuple)):
        param = tuple(param)
    else:
        raise TypeError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)


# TODO: change preserve_range_float() and others to use these
def to_float(img, maxval=None, t_dtype=np.dtype("float32")):
    if maxval is None:
        dtype = img.dtype
        maxval = MAX_VALUES_BY_DTYPE.get(dtype)
        if not maxval:
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo
            elif np.issubdtype(dtype, np.floating):
                info = np.finfo
            maxval = info(dtype).max
    return img.astype(t_dtype) / maxval


def from_float(img, dtype=np.dtype("uint8"), maxval=None):
    if maxval is None:
        try:
            maxval = MAX_VALUES_BY_DTYPE[dtype]
        except KeyError:
            raise RuntimeError(
                "Can't infer the maximum value for dtype {}. You need to specify the maximum value manually by "
                "passing the maxval argument".format(dtype)
            )
    return (img * maxval).astype(dtype)
