import numpy as np
from scipy.ndimage import rotate, zoom, gaussian_filter, map_coordinates


def random_flip(image, label, axes=(0, 1, 2)):
    """Random flip along each axis.
    Args: image (C, D, H, W), label (D, H, W)
    Returns: (C, D, H, W), (D, H, W)"""
    for ax in axes:
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=ax + 1)  # +1 for channel dim
            label = np.flip(label, axis=ax)
    return np.ascontiguousarray(image), np.ascontiguousarray(label)  # (C,D,H,W), (D,H,W)


def random_rotate(image, label, max_angle=30, p=0.2):
    """Random rotation in the H-W plane.
    Args: image (C, D, H, W), label (D, H, W)
    Returns: (C, D, H, W), (D, H, W)"""
    if np.random.rand() > p:
        return image, label
    angle = np.random.uniform(-max_angle, max_angle)
    rotated_channels = []
    for c in range(image.shape[0]):
        rotated_channels.append(
            rotate(image[c], angle, axes=(1, 2), reshape=False, order=1, mode="constant", cval=0)
        )
    image = np.stack(rotated_channels, axis=0)  # (C, D, H, W)
    label = rotate(label, angle, axes=(1, 2), reshape=False, order=0, mode="constant", cval=0)  # (D, H, W)
    return image, label


def random_scale(image, label, scale_range=(0.7, 1.4), p=0.2):
    """Random scaling with center crop/pad back to original size.
    Args: image (C, D, H, W), label (D, H, W)
    Returns: (C, D, H, W), (D, H, W)"""
    if np.random.rand() > p:
        return image, label
    scale = np.random.uniform(*scale_range)
    original_shape = image.shape[1:]  # (D, H, W)

    scaled_channels = []
    for c in range(image.shape[0]):
        scaled_channels.append(zoom(image[c], scale, order=1, mode="constant", cval=0))  # (D', H', W')
    image_scaled = np.stack(scaled_channels, axis=0)  # (C, D', H', W')
    label_scaled = zoom(label, scale, order=0, mode="constant", cval=0)  # (D', H', W')

    image = _center_crop_or_pad_4d(image_scaled, original_shape)  # (C, D, H, W)
    label = _center_crop_or_pad_3d(label_scaled, original_shape)  # (D, H, W)
    return image, label


def random_elastic(image, label, alpha=(0, 900), sigma=(9, 13), p=0.2):
    """Elastic deformation.
    Args: image (C, D, H, W), label (D, H, W)
    Returns: (C, D, H, W), (D, H, W)"""
    if np.random.rand() > p:
        return image, label
    shape = image.shape[1:]  # (D, H, W)
    a = np.random.uniform(*alpha)
    s = np.random.uniform(*sigma)
    dx = gaussian_filter(np.random.randn(*shape), s) * a  # (D, H, W)
    dy = gaussian_filter(np.random.randn(*shape), s) * a  # (D, H, W)
    dz = gaussian_filter(np.random.randn(*shape), s) * a  # (D, H, W)

    z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    coords = [z + dz, y + dy, x + dx]  # each (D, H, W)

    for c in range(image.shape[0]):
        image[c] = map_coordinates(image[c], coords, order=1, mode="constant", cval=0)  # (D, H, W)
    label = map_coordinates(label, coords, order=0, mode="constant", cval=0)  # (D, H, W)
    return image, label


def random_gaussian_noise(image, p=0.1):
    """Additive Gaussian noise.
    Args: image (C, D, H, W)
    Returns: (C, D, H, W)"""
    if np.random.rand() > p:
        return image
    noise = np.random.normal(0, 0.1, image.shape).astype(np.float32)  # (C, D, H, W)
    return image + noise  # (C, D, H, W)


def random_gaussian_blur(image, sigma_range=(0.5, 1.0), p=0.2):
    """Gaussian blur, applied per channel with 0.5 probability.
    Args: image (C, D, H, W)
    Returns: (C, D, H, W)"""
    if np.random.rand() > p:
        return image
    for c in range(image.shape[0]):
        if np.random.rand() < 0.5:
            sigma = np.random.uniform(*sigma_range)
            image[c] = gaussian_filter(image[c], sigma)  # (D, H, W)
    return image  # (C, D, H, W)


def random_brightness_multiplicative(image, factor_range=(0.75, 1.25), p=0.15):
    """Multiplicative brightness perturbation.
    Args: image (C, D, H, W)
    Returns: (C, D, H, W)"""
    if np.random.rand() > p:
        return image
    factor = np.random.uniform(*factor_range)
    return image * factor  # (C, D, H, W)


def random_gamma(image, gamma_range=(0.7, 1.5), p=0.3):
    """Gamma transform.
    Args: image (C, D, H, W)
    Returns: (C, D, H, W)"""
    if np.random.rand() > p:
        return image
    gamma = np.random.uniform(*gamma_range)
    for c in range(image.shape[0]):
        mn, mx = image[c].min(), image[c].max()
        rng = mx - mn + 1e-8
        image[c] = ((image[c] - mn) / rng) ** gamma * rng + mn  # (D, H, W)
    return image  # (C, D, H, W)


def _center_crop_or_pad_3d(vol, target_shape):
    """Args: vol (D, H, W), target_shape (D', H', W')
    Returns: (D', H', W')"""
    result = np.zeros(target_shape, dtype=vol.dtype)  # (D', H', W')
    slices_src, slices_dst = [], []
    for i in range(3):
        s, t = vol.shape[i], target_shape[i]
        if s >= t:
            start = (s - t) // 2
            slices_src.append(slice(start, start + t))
            slices_dst.append(slice(0, t))
        else:
            start = (t - s) // 2
            slices_src.append(slice(0, s))
            slices_dst.append(slice(start, start + s))
    result[tuple(slices_dst)] = vol[tuple(slices_src)]
    return result  # (D', H', W')


def _center_crop_or_pad_4d(vol, target_shape):
    """Args: vol (C, D, H, W), target_shape (D', H', W')
    Returns: (C, D', H', W')"""
    channels = []
    for c in range(vol.shape[0]):
        channels.append(_center_crop_or_pad_3d(vol[c], target_shape))  # (D', H', W')
    return np.stack(channels, axis=0)  # (C, D', H', W')


class PicaiAugmentation:
    """nnUNet-style augmentation pipeline, parameters aligned with PI-CAI baseline."""

    def __call__(self, image, label):
        """Args: image (C, D, H, W), label (D, H, W)
        Returns: (C, D, H, W), (D, H, W)"""
        image = image.copy()  # (C, D, H, W)
        label = label.copy()  # (D, H, W)

        # Spatial augmentations
        image, label = random_flip(image, label)
        image, label = random_rotate(image, label, max_angle=30, p=0.2)
        image, label = random_scale(image, label, scale_range=(0.7, 1.4), p=0.2)
        image, label = random_elastic(image, label, p=0.2)

        # Intensity augmentations
        image = random_gaussian_noise(image, p=0.1)
        image = random_gaussian_blur(image, p=0.2)
        image = random_brightness_multiplicative(image, p=0.15)
        image = random_gamma(image, p=0.3)

        return image, label  # (C, D, H, W), (D, H, W)
