"""
Follow: pytorch transforms.py for the custom TSS Transform because pytorch transforms cannot support
>3D data, which in TSS we have 4D data (x,y,z, imagetype)



In here, we assume the data comes in has a dimension of z,x,y,c
"""

import os
import sys
import numbers
import numpy as np
from scipy.ndimage import rotate, zoom
import torch
from data_loader.affine_transforms import Affine
import torch.nn.functional as F
from typing import Tuple

import torchio as tio

dirname = os.path.dirname(__file__)
sys.path.insert(0, dirname)


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomVerticalFlip(object):
    """   
    Vertically flip the input randomly with a given probability    
    """

    def __init__(self, p=0.5):
        """
        p (float): probability of the image being flipped. Default value is 0.5
        """
        self.p = p

    def __call__(self, matrix):
        """
        :param matrix: dim = z,x,y,c
        """
        if np.random.random_sample() < self.p:
            return np.flip(matrix, axis=1)
        return matrix

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomHorizontalFlip(object):
    """    
    horizontal flip the input randomly with a given probability        
    """

    def __init__(self, p=0.5):
        """
        p (float): probability of the image being flipped. Default value is 0.5
        """
        self.p = p

    def __call__(self, matrix):
        """
        :param matrix: dim = z,x,y,c
        """
        if np.random.random_sample() < self.p:
            return np.flip(matrix, axis=2)
        return matrix

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomRotation(object):
    """    
    Randomly rotate the image
    """

    def __init__(self, degrees):
        self.degrees = (0.5 - np.random.random_sample()) * 2 * degrees  # random degree (positive or negative)

    def __call__(self, matrix):
        """
        :param matrix: dim = z,x,y,c
        """
        # order = 1(linear spline), 2(quadratic splines), 3 (cubic splines; common)
        matrix = rotate(matrix, self.degrees, reshape=False, order=3)
        return matrix

    def __repr__(self):
        return self.__class__.__name__ + '(degrees={0}'.format(self.degrees)


class RandomShear(object):
    """    
    Randomly affine transform - shear
    
    def __call__(self, matrix):
        degree = np.random.random_sample() 
        matrix = (matrix - np.mean(matrix, axis=1))/np.std(matrix, axis=1)
        matrix = (matrix - np.mean(matrix, axis=0))/np.std(matrix, axis=0)
        if degree >=0.1 and degree <0.5:
            affine_tf = sktf.AffineTransform(shear = degree)
            matrix = sktf.warp(matrix, inverse_map = affine_tf)
        return matrix
    """
    pass


class RandomAffine(object):

    def __init__(self, rotation=None, zoom=None, shear=None,
                 translation_xy=None, translation_z=None):
        """
        Perform random affine transformation. Modification of
        https://pytorch.org/docs/stable/torchvision/transforms.html.
        
        :param rotation: rotation angle, +-180 degree, suggest: 15
        :param zoom: zoom, suggest: (0.9, 1.1)
        :param shear: shear angle value, -180-180, suggest: 0.26180 for 15 degree
        :param translation_xy: translation pixel in x and y direction, 
                                suggest: (0.14286, 0.21429) for (16,12) pixels
        :param translation_z: translation pxiel in z direction, suggest: 0.21429 for 6 slices
        """
        self.rotation = rotation
        self.zoom = zoom
        self.shear = shear
        self.translation_xy = translation_xy
        self.translation_z = translation_z

    def __call__(self, matrix):
        """
        :param matrix: dim = z,x,y,c
        """
        # ===== transform
        if self.rotation or self.zoom or self.shear or self.translation_xy:
            affine = Affine(rotation_range=self.rotation,
                            zoom_range=self.zoom,
                            shear_range=self.shear,  # 15 degree
                            translation_range=self.translation_xy  # 16,12 pixels
                            )

            matrix = self._matrix_affine(matrix, affine)

        # ===== transform in z-axis (slice level of the brain),
        # only focus on translation, the others simply just
        # ignore

        if self.translation_z:
            # single image
            # affine
            affine = Affine(translation_range=(self.translation_z, 0))  # 2 slices;
            matrix = self._matrix_affine(matrix, affine)

        return matrix

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def _matrix_affine(self, matrix, affine):
        """
        Perform affine transomation on a matrix.

        :param matrix: input numpy
        :param affine: affine operator
        """
        matrix_affine = affine(matrix)

        return matrix_affine


class RandomNoise(object):
    def __init__(self, low=-0.1, high=0.1):
        """
        randomly add noise from uniform distribution to the matrix
        :param low: the lower bound
        :param high: the upper bound
        """
        self.low = low
        self.high = high

    def __call__(self, matrix):
        noise = np.random.uniform(self.low, self.high)
        return matrix + noise

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensor(object):
    """
    convert to Pytorch tensor
    """

    def __call__(self, matrix):
        """
        :param matrix: dim = x,y,z
        """
        return torch.from_numpy(matrix.copy())

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomResize(object):
    """
    Resize the input. 
    """

    def __init__(self, zoom_factor):
        """
        :param zoom_factor: how much you want to zoom in each axis, a sequence
        """
        self.zoom_factor = zoom_factor
        self.zx = np.random.uniform(self.zoom_factor[0], self.zoom_factor[1])
        self.zy = np.random.uniform(self.zoom_factor[0], self.zoom_factor[1])

    def __call__(self, matrix):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return zoom(matrix, (1, 1, self.zx, self.zy))  # spline interpolation
        # return sktf.resize(matrix,self.output_shape)
        # return cv2.resize(matrix, self.output_shape, interpolation=cv2.INTER_LINEAR)

    def __repr__(self):
        return self.__class__.__name__ + 'zoom_factor={0}'.format(self.zoom_factor)


class CenterCrop(object):
    """Crops the image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size=35):
        if isinstance(size, numbers.Number):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, matrix):
        w = matrix.shape[2]
        h = matrix.shape[1]
        th, tw = self.size
        x1 = int(np.round((w - tw) / 2.))
        y1 = int(np.round((h - th) / 2.))
        return matrix[x1:x1 + tw, y1:y1 + th, :]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}'.format(self.size)


class Normalize_Standardize(object):
    '''
     Normalize x and y dim
    '''

    def __init__(self, method):
        self.method = method

    def __call__(self, matrix):
        if self.method == 'Standardization':
            matrix = (matrix - np.mean(matrix, axis=(1, 2))) / (np.std(matrix, axis=(1, 2)) + +sys.float_info.epsilon)
        if self.method == 'Normalization':
            matrix = (matrix - np.min(matrix, axis=(1, 2), keepdims=True)) / (
                    np.max(matrix, axis=(1, 2), keepdims=True) - np.min(matrix, keepdims=True))
        return matrix


class RandomCrop:
    """Random cropping on subject."""

    def __init__(self, roi_size: Tuple):
        """Init.

        Args:
            roi_size: cropping size.
        """
        self.c_z = roi_size[0]
        self.c_x = roi_size[1]
        self.c_y = roi_size[2]

    def __call__(self, subject: tio.Subject) -> tio.Subject:
        """Use patch sampler to crop.

        Args:
            matrix: original image

        Returns:
            cropped subject
        """

        z = subject.shape[1]
        x = subject.shape[2]
        y = subject.shape[3]

        z_s = np.random.randint(0, z - 1 - self.c_z)
        y_s = np.random.randint(0, y - self.c_y)
        x_s = np.random.randint(0, x - self.c_x)
        z_e = z_s + self.c_z
        y_e = y_s + self.c_y
        x_e = x_s + self.c_x

        return subject[:, z_s:z_e, y_s:y_e, x_s:x_e]

