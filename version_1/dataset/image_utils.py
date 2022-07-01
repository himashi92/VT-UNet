"""functions to correctly pad or crop non uniform sized MRI (before batching in the dataloader).
"""
import random

import numpy as np


def pad_or_crop_image(image, seg=None, target_size=(128, 144, 144)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    if seg is not None:
        seg = seg[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    if seg is not None:
        seg = np.pad(seg, padlist)
        return image, seg
    return image


def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right


def get_crop_slice(target_size, dim):
    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)


def normalize(image):
    """Basic min max scaler.
    """
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale
    return image


def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Main pre-processing function used for the challenge (seems to work the best).

    Remove outliers voxels first, then min-max scale.

    Warnings
    --------
    This will not do it channel wise!!
    """

    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    image = normalize(image)
    return image


def zscore_normalise(img: np.ndarray) -> np.ndarray:
    slices = (img != 0)
    img[slices] = (img[slices] - np.mean(img[slices])) / np.std(img[slices])
    return img


def remove_unwanted_background(image, threshold=1e-5):
    """Use to crop zero_value pixel from MRI image.
    """
    dim = len(image.shape)
    non_zero_idx = np.nonzero(image > threshold)
    min_idx = [np.min(idx) for idx in non_zero_idx]
    # +1 because slicing is like range: not inclusive!!
    max_idx = [np.max(idx) + 1 for idx in non_zero_idx]
    bbox = tuple(slice(_min, _max) for _min, _max in zip(min_idx, max_idx))
    return image[bbox]


def random_crop2d(*images, min_perc=0.5, max_perc=1.):
    """Crop randomly but identically all images given.

    Could be used to pass both mask and image at the same time. Anything else will
    throw.

    Warnings
    --------
    Only works for channel first images. (No channel image will not work).
    """
    if len(set(tuple(image.shape) for image in images)) > 1:
        raise ValueError("Image shapes do not match")
    shape = images[0].shape
    new_sizes = [int(dim * random.uniform(min_perc, max_perc)) for dim in shape]
    min_idx = [random.randint(0, ax_size - size) for ax_size, size in zip(shape, new_sizes)]
    max_idx = [min_id + size for min_id, size in zip(min_idx, new_sizes)]
    bbox = list(slice(min_, max(max_, 1)) for min_, max_ in zip(min_idx, max_idx))
    # DO not crop channel axis...
    bbox[0] = slice(0, shape[0])
    # prevent warning
    bbox = tuple(bbox)
    cropped_images = [image[bbox] for image in images]
    if len(cropped_images) == 1:
        return cropped_images[0]
    else:
        return cropped_images


def random_crop3d(*images, min_perc=0.5, max_perc=1.):
    """Crop randomly but identically all images given.

    Could be used to pass both mask and image at the same time. Anything else will
    throw.

    Warnings
    --------
    Only works for channel first images. (No channel image will not work).
    """
    return random_crop2d(min_perc, max_perc, *images)