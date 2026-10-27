from typing import Tuple, List, Union

import math

import numpy as np
import torch
from scipy.ndimage import fourier_gaussian
from torch.nn.functional import grid_sample, interpolate

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.cropping import crop_tensor


class SpatialTransformRatioCrop(BasicTransform):
    def __init__(self,
                 patch_size: Tuple[int, ...],
                 patch_center_dist_from_border: Union[int, List[int], Tuple[int, ...]],
                 random_crop: bool,
                 crop_scale: Tuple[float, float] = (1.0, 1.0),  # NEW: (min_ratio, max_ratio)
                 p_elastic_deform: float = 0,
                 elastic_deform_scale: RandomScalar = (0, 0.2),
                 elastic_deform_magnitude: RandomScalar = (0, 0.2),
                 p_synchronize_def_scale_across_axes: float = 0,
                 p_rotation: float = 0,
                 rotation: RandomScalar = (0, 2 * np.pi),
                 p_scaling: float = 0,
                 scaling: RandomScalar = (0.7, 1.3),
                 p_synchronize_scaling_across_axes: float = 0,
                 bg_style_seg_sampling: bool = True,
                 mode_seg: str = 'bilinear',
                 border_mode_seg: str = "zeros",
                 center_deformation: bool = True,
                 padding_mode_image: str = "zeros",
                 keep_crop_inside: bool = False,
                 ):
        """
        NEW PARAMETER:
        crop_scale: Tuple of (min_ratio, max_ratio) for sampling the ratio between
                           crop volume and original image volume. Default (1.0, 1.0) means
                           no random volume scaling (same as original behavior).
                           Example: (0.5, 1.0) means crop between 50% and 100% of original volume
        
        The workflow is:
        1. Sample a volume ratio from crop_scale
        2. Calculate intermediate crop size based on this ratio
        3. Crop the image at the intermediate size
        4. Apply transformations (rotation, deformation, scaling)
        5. Resize to patch_size as the final output
        
        Other parameters same as original SpatialTransform.
        """
        super().__init__()
        self.patch_size = patch_size
        if not isinstance(patch_center_dist_from_border, (tuple, list)):
            patch_center_dist_from_border = [patch_center_dist_from_border] * len(patch_size)
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.random_crop = random_crop
        self.crop_scale = crop_scale
        self.p_elastic_deform = p_elastic_deform
        self.elastic_deform_scale = elastic_deform_scale
        self.elastic_deform_magnitude = elastic_deform_magnitude
        self.p_rotation = p_rotation
        self.rotation = rotation
        self.p_scaling = p_scaling
        self.scaling = scaling
        self.p_synchronize_scaling_across_axes = p_synchronize_scaling_across_axes
        self.p_synchronize_def_scale_across_axes = p_synchronize_def_scale_across_axes
        self.bg_style_seg_sampling = bg_style_seg_sampling
        self.mode_seg = mode_seg
        self.border_mode_seg = border_mode_seg
        self.center_deformation = center_deformation
        self.padding_mode_image = padding_mode_image
        # keep the crop centre at least half a crop away from the border (otherwise ~56% of a global crop is padding)
        self.keep_crop_inside = keep_crop_inside

    def _calculate_crop_size(self, image_shape: Tuple[int, ...], volume_ratio: float) -> Tuple[int, ...]:
        if not (0 <= volume_ratio <= 1):
            raise ValueError("Ratio must be in [0, 1]")
        x, y, z = image_shape
        if volume_ratio == 0:
            return (0, 0, 0)
        a = np.random.randint(math.ceil(x * volume_ratio), x + 1)
        min_b = math.ceil(x * y * volume_ratio / a)
        max_b = y
        b = np.random.randint(min_b, max_b + 1)
        c = round(x * y * z * volume_ratio / (a * b))
        return (a, b, c)

    def get_parameters(self, **data_dict) -> dict:
        dim = data_dict['image'].ndim - 1
        image_shape = data_dict['image'].shape[1:]

        # NEW: Sample volume ratio and calculate intermediate crop size
        volume_ratio = np.random.uniform(self.crop_scale[0], self.crop_scale[1])
        intermediate_crop_size = self._calculate_crop_size(image_shape, volume_ratio)

        do_rotation = np.random.uniform() < self.p_rotation
        do_scale = np.random.uniform() < self.p_scaling
        do_deform = np.random.uniform() < self.p_elastic_deform

        if do_rotation:
            angles = [sample_scalar(self.rotation, image=data_dict['image'], dim=i) for i in range(0, 3)]
        else:
            angles = [0] * dim
        if do_scale:
            if np.random.uniform() <= self.p_synchronize_scaling_across_axes:
                scales = [sample_scalar(self.scaling, image=data_dict['image'], dim=None)] * dim
            else:
                scales = [sample_scalar(self.scaling, image=data_dict['image'], dim=i) for i in range(0, 3)]
        else:
            scales = [1] * dim

        # affine matrix
        if do_scale or do_rotation:
            if dim == 3:
                affine = create_affine_matrix_3d(angles, scales)
            elif dim == 2:
                affine = create_affine_matrix_2d(angles[-1], scales)
            else:
                raise RuntimeError(f'Unsupported dimension: {dim}')
        else:
            affine = None

        # elastic deformation - now applied to intermediate_crop_size
        if do_deform:
            if np.random.uniform() <= self.p_synchronize_def_scale_across_axes:
                deformation_scales = [
                    sample_scalar(self.elastic_deform_scale, image=data_dict['image'], dim=None, 
                                patch_size=intermediate_crop_size)
                    ] * dim
            else:
                deformation_scales = [
                    sample_scalar(self.elastic_deform_scale, image=data_dict['image'], dim=i, 
                                patch_size=intermediate_crop_size)
                    for i in range(dim)
                    ]

            sigmas = [i * j for i, j in zip(deformation_scales, intermediate_crop_size)]

            magnitude = [
                sample_scalar(self.elastic_deform_magnitude, image=data_dict['image'], 
                            patch_size=intermediate_crop_size, dim=i, deformation_scale=deformation_scales[i])
                for i in range(dim)]
            
            offsets = torch.normal(mean=0, std=1, size=(dim, *intermediate_crop_size))

            for d in range(dim):
                tmp = np.fft.fftn(offsets[d].numpy())
                tmp = fourier_gaussian(tmp, sigmas[d])
                offsets[d] = torch.from_numpy(np.fft.ifftn(tmp).real)

                mx = torch.max(torch.abs(offsets[d]))
                offsets[d] /= (mx / np.clip(magnitude[d], a_min=1e-8, a_max=np.inf))
            
            spatial_dims = tuple(list(range(1, dim + 1)))
            offsets = torch.permute(offsets, (*spatial_dims, 0))
        else:
            offsets = None

        # Calculate center location for the intermediate crop
        if not self.random_crop:
            center_location_in_pixels = [i / 2 for i in image_shape]
        else:
            center_location_in_pixels = []
            for d in range(dim):
                mn = self.patch_center_dist_from_border[d]
                mx = image_shape[d] - self.patch_center_dist_from_border[d]
                if self.keep_crop_inside:
                    mn = max(mn, intermediate_crop_size[d] / 2)
                    mx = min(mx, image_shape[d] - intermediate_crop_size[d] / 2)
                if mx < mn:
                    center_location_in_pixels.append(image_shape[d] / 2)
                else:
                    center_location_in_pixels.append(np.random.uniform(mn, mx))
        
        return {
            'affine': affine,
            'elastic_offsets': offsets,
            'center_location_in_pixels': center_location_in_pixels,
            'intermediate_crop_size': intermediate_crop_size,  # NEW
            'volume_ratio': volume_ratio  # NEW: for debugging/logging
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        intermediate_crop_size = params['intermediate_crop_size']
        
        if params['affine'] is None and params['elastic_offsets'] is None:
            # No spatial transformation, just crop and resize
            if self.padding_mode_image == 'reflection':
                pad_mode = 'reflect'
                pad_kwargs = {}
            elif self.padding_mode_image == 'zeros':
                pad_mode = 'constant'
                pad_kwargs = {'value': 0}
            elif self.padding_mode_image == 'border':
                pad_mode = 'replicate'
                pad_kwargs = {}
            else:
                raise RuntimeError('Unknown pad mode')

            # Crop to intermediate size
            img = crop_tensor(img, [math.floor(i) for i in params['center_location_in_pixels']], 
                            intermediate_crop_size, pad_mode=pad_mode, pad_kwargs=pad_kwargs)
            
            # Resize to final patch_size
            img = interpolate(img[None], size=self.patch_size, mode='trilinear' if img.ndim == 4 else 'bilinear',
                              align_corners=False)[0]
            return img
        elif params['elastic_offsets'] is None:
            # Affine only: sample the output grid directly (one grid_sample at patch_size covering the
            # intermediate crop extent) instead of a grid at the intermediate size followed by a resize.
            # Same mapping, 1.3-4x faster on CPU and far less memory.
            space = [(torch.arange(p, dtype=torch.float32) + 0.5) * (s / p) - s / 2
                     for p, s in zip(self.patch_size, intermediate_crop_size)]
            grid = torch.stack(torch.meshgrid(space, indexing="ij"), -1)
            grid = torch.matmul(grid, torch.from_numpy(params['affine']).float())
            new_center = torch.Tensor([c - s / 2 for c, s in zip(params['center_location_in_pixels'], img.shape[1:])])
            grid += new_center
            return grid_sample(img[None], _convert_my_grid_to_grid_sample_grid(grid, img.shape[1:])[None],
                               mode='bilinear', padding_mode=self.padding_mode_image, align_corners=False)[0]
        else:
            # Create grid for intermediate crop size
            grid = _create_centered_identity_grid2(intermediate_crop_size)

            # Apply deformation and affine transforms
            if params['elastic_offsets'] is not None:
                grid += params['elastic_offsets']
            if params['affine'] is not None:
                grid = torch.matmul(grid, torch.from_numpy(params['affine']).float())

            # Center the grid
            if self.center_deformation and params['elastic_offsets'] is not None:
                mn = grid.mean(dim=list(range(img.ndim - 1)))
            else:
                mn = 0

            new_center = torch.Tensor([c - s / 2 for c, s in zip(params['center_location_in_pixels'], img.shape[1:])])
            grid += (new_center - mn)
            
            # Apply grid sampling to get intermediate crop
            img_cropped = grid_sample(img[None], _convert_my_grid_to_grid_sample_grid(grid, img.shape[1:])[None],
                                     mode='bilinear', padding_mode=self.padding_mode_image, align_corners=False)[0]
            
            # Resize to final patch_size
            img_final = interpolate(img_cropped[None], size=self.patch_size, 
                                     mode='trilinear' if img.ndim == 4 else 'bilinear',
                                     align_corners=False)[0]
            
            return img_final


def create_affine_matrix_3d(rotation_angles, scaling_factors):
    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
                   [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]])

    Ry = np.array([[np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]])

    Rz = np.array([[np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
                   [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0],
                   [0, 0, 1]])

    # Scaling matrix
    S = np.diag(scaling_factors)

    # Combine rotation and scaling
    RS = Rz @ Ry @ Rx @ S
    return RS

def create_affine_matrix_2d(rotation_angle, scaling_factors):
    # Rotation matrix
    R = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                  [np.sin(rotation_angle), np.cos(rotation_angle)]])

    # Scaling matrix
    S = np.diag(scaling_factors)

    # Combine rotation and scaling
    RS = R @ S
    return RS

def _create_centered_identity_grid2(size: Union[Tuple[int, ...], List[int]]) -> torch.Tensor:
    space = [torch.linspace((1 - s) / 2, (s - 1) / 2, s) for s in size]
    grid = torch.meshgrid(space, indexing="ij")
    grid = torch.stack(grid, -1)
    return grid

def _convert_my_grid_to_grid_sample_grid(my_grid: torch.Tensor, original_shape: Union[Tuple[int, ...], List[int]]):
    # rescale
    for d in range(len(original_shape)):
        s = original_shape[d]
        my_grid[..., d] /= (s / 2)
    my_grid = torch.flip(my_grid, (len(my_grid.shape) - 1, ))
    # my_grid = my_grid.flip((len(my_grid.shape) - 1,))
    return my_grid
