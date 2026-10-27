import torch
import math
from scipy.stats import truncnorm
import numpy as np
from monai.transforms import (
    Compose, 
    ScaleIntensity, 
    CropForeground, RandScaleCrop, Resize, RandRotate, RandFlip,
    Transform, SpatialPad, RandRotate90, RandGaussianNoise,
    RandGaussianSmooth, RandAdjustContrast,
    RandSimulateLowResolution, RandScaleIntensityFixedMean,
    RandomizableTransform, ApplyPending, CenterSpatialCrop
)
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform, BrightnessAdditiveTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform

from SpatialTransformRatioCrop import SpatialTransformRatioCrop

class RandRotatev2(RandRotate):
    def __init__(
        self,
        range_x = 0,
        range_y = 0,
        range_z = 0,
        mean = 0,
        std = 1,
        prob = 0.1,
        keep_size = True,
        mode = 'bilinear',
        padding_mode = 'border',
        align_corners = False,
        dtype = np.float32,
        lazy = False
    ):
        super().__init__(range_x, range_y, range_z, prob, keep_size, mode, padding_mode, align_corners, dtype, lazy)
        self.mean = mean
        self.std = std

    def randomize(self, data=None):
        super().randomize(None)
        if not self._do_transform:
            return None
        self.x = self.sample_truncnorm(self.range_x[0], self.range_x[1], self.mean, self.std)
        self.y = self.sample_truncnorm(self.range_y[0], self.range_y[1], self.mean, self.std)
        self.z = self.sample_truncnorm(self.range_z[0], self.range_z[1], self.mean, self.std)

    def sample_truncnorm(self, low, high, mean, std):
        a, b = (low - mean) / std, (high - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, random_state=self.R)

class PadOrResize(RandomizableTransform):
    def __init__(
        self,
        spatial_size,
        prob: float,
        interpolation_mode: str = "trilinear",
        align_corners: bool = None,
        anti_aliasing: bool = True,
    ):
        super().__init__(prob=prob)
        if isinstance(spatial_size, int):
            self.spatial_size = (spatial_size, spatial_size, spatial_size)
        else:
            self.spatial_size = tuple(spatial_size)
        self.prob = prob
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners
        self.anti_aliasing = anti_aliasing
        self.resize = Resize(
            spatial_size=self.spatial_size,
            mode=interpolation_mode,
            align_corners=align_corners,
            anti_aliasing=anti_aliasing
        )
        self.pad = SpatialPad(spatial_size=self.spatial_size)
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        assert img.ndim == 4
        current_size = img.shape[1:]
        target_size = self.spatial_size
        downsample_size = [tgt if curr > tgt else -1 for curr, tgt in zip(current_size, target_size)]
        downsample = Resize(
            spatial_size=downsample_size,
            mode=self.interpolation_mode,
            align_corners=self.align_corners,
            anti_aliasing=self.anti_aliasing,
        )
        downsampled = downsample(img)
        self.randomize()
        if self._do_transform:
            return self.pad(downsampled)
        else:
            tmp_target_size = [math.ceil(tgt/2) if 2*curr < tgt else -1 for curr, tgt in zip(current_size, target_size)]
            padded = SpatialPad(spatial_size=tmp_target_size)(img)
            return self.resize(padded)
    
class SampleOrCropDepth(Transform):
    def __init__(self, spatial_size):
        self.spatial_size = spatial_size
        self.crop = CenterSpatialCrop((spatial_size, np.iinfo(np.int16).max, np.iinfo(np.int16).max))

    def __call__(self, img):
        if img.shape[1] >= 2* self.spatial_size:
            img = img[::2,:,:]
            if img.shape[1] <= self.spatial_size:
                return img
        return self.crop(img)

class DataAugmentationCTiBOT(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number):
        minmax_scaling_and_crop_foreground = Compose([
            ScaleIntensity(minv=0, maxv=1),
            CropForeground(),
        ])
        flips_and_rotate = Compose([
            RandFlip(spatial_axis=0, prob=1/15),
            RandFlip(spatial_axis=1, prob=1/15),
            RandFlip(spatial_axis=2, prob=1/15),
            RandRotate90(prob=1/15, spatial_axes=(0, 1)),
            RandRotate90(prob=1/15, spatial_axes=(0, 2)),
            RandRotate90(prob=1/15, spatial_axes=(1, 2)),
            RandRotatev2(
                range_x=(-np.pi/2, np.pi/2),
                range_y=(-np.pi/2, np.pi/2),
                range_z=(-np.pi/2, np.pi/2),
                mean=0,
                std=np.pi/16,
                prob=0.1,
                mode='trilinear',
                padding_mode='zeros',
            ),
        ])
        contrast_transform = Compose([
            RandAdjustContrast(
                gamma=(0.7, 1.5),
                invert_image=True,
                retain_stats=True,
                prob=0.1,
            ),
            RandAdjustContrast(
                gamma=(0.7, 1.5),
                invert_image=False,
                retain_stats=True,
                prob=0.3,
            ),
        ])
        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = Compose([
            minmax_scaling_and_crop_foreground,
            RandScaleCrop(roi_scale=global_crops_scale[0], max_roi_scale=global_crops_scale[1], random_size=True),
            # PadOrResize(spatial_size=224, prob=0.5, interpolation_mode='trilinear', anti_aliasing=True),
            Resize(spatial_size=3*(224,), mode='trilinear', anti_aliasing=True),
            flips_and_rotate,
            ApplyPending(),
            RandGaussianNoise(mean=0, std=0.1, prob=0.1),
            RandGaussianSmooth(
                sigma_x=(0.5, 1),
                sigma_y=(0.5, 1),
                sigma_z=(0.5, 1),
                prob=0.75,
            ),
            RandScaleIntensityFixedMean(
                factors=0.25,
                preserve_range=True,
                prob=0.15,
            ),
            RandSimulateLowResolution(
                zoom_range=(0.5, 1),
                downsample_mode='nearest-exact',
                upsample_mode='trilinear',
                prob=0.125,
            ),
            RandAdjustContrast(
                gamma=(0.7, 1.5),
                invert_image=False,
                retain_stats=True,
                prob=0.3,
            ),
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = Compose([
            minmax_scaling_and_crop_foreground,
            RandScaleCrop(roi_scale=global_crops_scale[0], max_roi_scale=global_crops_scale[1], random_size=True),
            # PadOrResize(spatial_size=224, prob=0.5, interpolation_mode='trilinear', anti_aliasing=True),
            Resize(spatial_size=3*(224,), mode='trilinear', anti_aliasing=True),
            flips_and_rotate,
            ApplyPending(),
            RandGaussianNoise(mean=0, std=0.1, prob=0.1),
            RandGaussianSmooth(
                sigma_x=(0.5, 1),
                sigma_y=(0.5, 1),
                sigma_z=(0.5, 1),
                prob=0.1,
            ),
            RandScaleIntensityFixedMean(
                factors=0.25,
                preserve_range=True,
                prob=0.15,
            ),
            RandSimulateLowResolution(
                zoom_range=(0.5, 1),
                downsample_mode='nearest-exact',
                upsample_mode='trilinear',
                prob=0.125,
            ),
            contrast_transform,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = Compose([
            minmax_scaling_and_crop_foreground,
            RandScaleCrop(roi_scale=local_crops_scale[0], max_roi_scale=local_crops_scale[1], random_size=True),
            # PadOrResize(spatial_size=96, prob=0.5, interpolation_mode='trilinear', anti_aliasing=True),
            Resize(spatial_size=3*(96,), mode='trilinear', anti_aliasing=True),
            flips_and_rotate,
            ApplyPending(),
            RandGaussianNoise(mean=0, std=0.1, prob=0.1),
            RandGaussianSmooth(
                sigma_x=(0.5, 1),
                sigma_y=(0.5, 1),
                sigma_z=(0.5, 1),
                prob=0.5,
            ),
            RandScaleIntensityFixedMean(
                factors=0.25,
                preserve_range=True,
                prob=0.15,
            ),
            contrast_transform,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

class MinMaxScalingTransform(ImageOnlyTransform):
    def __init__(self, target_range=(0, 1)):
        super().__init__()
        self.target_range = target_range

    def get_parameters(self, **data_dict):
        img = data_dict['image']
        a, b = self.target_range
        mins = torch.min(img)
        maxs = torch.max(img)
        denom = torch.clamp(maxs - mins, min=1e-8)
        scale = (b - a) / denom
        return {
            'mins': mins,
            'scale': scale,
            'a': a
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        img = (img - params['mins']) * params['scale'] + params['a']
        return img
    
class HUWindowTransform(ImageOnlyTransform):
    """Fixed HU window -> [0, 1]. Unlike per-volume min-max, every source gets the same intensity mapping."""
    def __init__(self, window=(-1000.0, 1000.0)):
        super().__init__()
        self.window = window

    def get_parameters(self, **data_dict):
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        lo, hi = self.window
        return (img.clamp(lo, hi) - lo) / (hi - lo)


class ClipIntensityTransform(ImageOnlyTransform):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def get_parameters(self, **data_dict):
        return {}

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return torch.clamp(img, self.min_val, self.max_val)

class DataAugmentationCTiBOT2(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number,
                 intensity_norm='minmax', hu_window=(-1000.0, 1000.0), crop_inside=False):
        global_crops_size = 3*(224,)
        local_crops_size = 3*(96,)
        ignore_axes = None        
        def build_transform(patch_size, crop_scale):
            return ComposeTransforms([
                MinMaxScalingTransform(target_range=(0, 1)) if intensity_norm == 'minmax' else HUWindowTransform(hu_window),

                SpatialTransformRatioCrop(
                    patch_size,
                    patch_center_dist_from_border=0,
                    random_crop=True,
                    crop_scale=crop_scale,
                    keep_crop_inside=crop_inside,
                    p_elastic_deform=0,
                    p_rotation=0.2,
                    rotation=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    p_scaling=0.2,
                    scaling=(0.7, 1.4),
                    p_synchronize_scaling_across_axes=1,
                    bg_style_seg_sampling=False,
                ),

                RandomTransform(
                    GaussianNoiseTransform(
                        noise_variance=(0, 0.1),
                        p_per_channel=1,
                        synchronize_channels=True,
                    ),
                    apply_probability=0.1
                ),

                RandomTransform(
                    GaussianBlurTransform(
                        blur_sigma=(0.5, 1.),
                        synchronize_channels=False,
                        synchronize_axes=False,
                        p_per_channel=0.5,
                        benchmark=True,
                    ),
                    apply_probability=0.2
                ),

                RandomTransform(
                    MultiplicativeBrightnessTransform(
                        multiplier_range=BGContrast((0.75, 1.25)),
                        synchronize_channels=False,
                        p_per_channel=1,
                    ),
                    apply_probability=0.15
                ),

                RandomTransform(
                    ContrastTransform(
                        contrast_range=BGContrast((0.75, 1.25)),
                        preserve_range=True,
                        synchronize_channels=False,
                        p_per_channel=1,
                    ),
                    apply_probability=0.15
                ),

                RandomTransform(
                    SimulateLowResolutionTransform(
                        scale=(0.5, 1),
                        synchronize_channels=False,
                        synchronize_axes=True,
                        ignore_axes=ignore_axes,
                        allowed_channels=None,
                        p_per_channel=0.5,
                    ),
                    apply_probability=0.25
                ),

                RandomTransform(
                    GammaTransform(
                        gamma=BGContrast((0.7, 1.5)),
                        p_invert_image=1,
                        synchronize_channels=False,
                        p_per_channel=1,
                        p_retain_stats=1,
                    ),
                    apply_probability=0.1
                ),

                RandomTransform(
                    GammaTransform(
                        gamma=BGContrast((0.7, 1.5)),
                        p_invert_image=0,
                        synchronize_channels=False,
                        p_per_channel=1,
                        p_retain_stats=1,
                    ),
                    apply_probability=0.3
                ),

                RandomTransform(
                    MirrorTransform(allowed_axes=(0, 1, 2)),
                    apply_probability=0.1
                ),

                ClipIntensityTransform(min_val=0, max_val=1)
            ])

        # Store crop numbers
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number

        # Build transforms using the specified scale
        self.global_transfo1 = build_transform(global_crops_size, global_crops_scale)
        self.global_transfo2 = build_transform(global_crops_size, global_crops_scale)
        self.local_transfo = build_transform(local_crops_size, local_crops_scale)


    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image=image)['image'])
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image=image)['image'])
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image=image)['image'])
        return crops