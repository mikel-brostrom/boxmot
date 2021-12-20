from typing import List, Tuple
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable


def resize(img: np.ndarray, size: Tuple[int, int], interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.

    Args:
        img ( np.ndarray Image): Image to be resized.
        size (tuple of int): Desired output size. The size is a sequence with the structure
            (h, w). The output size will be matched to this. `
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    # Resizing needs "PIL.Image" objects, as scipy method was deprecated
    # and numpy does not support these sort of image transformations
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(size[::-1], interpolation)
    return np.array(pil_img)


def ndarray_to_tensor(pic: np.ndarray):
    """Translates the ndarray to a torch Tensor
    :param pic: PIL Image: Resized image
    :return: transformed torch.Tensor, required by the "normalize()" method
    """

    # Define constants
    RGB_VALUES = 255

    # handle numpy array
    if pic.ndim == 2:
        pic = pic[:, :, None]
    tensor = torch.from_numpy(pic.transpose((2, 0, 1)) / RGB_VALUES)
    return tensor


def normalize(tensor: torch.Tensor, mean: List[float], std: List[float], inplace=False):
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """

    if not inplace:
        tensor = tensor.clone()
    # Issues leaving a tensor as torch.uint8, which is the exit of the method ndarray_to_tensor(),
    # as tensor.sub_(mean[:, None, None]).div_(std[:, None, None]) presented an issue of division by 0.
    # Output of torchvision.transforms.compose() is a torch.float32
    tensor = tensor.type(torch.float32)
    dtype = tensor.dtype
    # Transform mean and std in torch tensors
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    # Normalization is implemented
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    # unsqueeze_() adds another dimension to the tensor, with shape  (:,:,:).
    # The input of our net requires a (1,:,:,:) tensor.
    tensor.unsqueeze_(0)
    # "autograd.Variable()" creates tensors that support gradient calculations.
    # Might be redundant as no backpropagation is computed in "test" mode.
    return Variable(tensor)
