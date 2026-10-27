"""GPU version of transforms.DataAugmentationCTiBOT2.

The CPU pipeline costs 0.8-3.5 s per sample on one core (see david-ct/CODE.md),
which starves the GPU. Here the DataLoader workers only load the raw volume; this module does the
multi-crop augmentation on the GPU for the whole batch:

  * crop (volume-ratio sampling as SpatialTransformRatioCrop) + optional rotation/scaling + resize
    to the crop size are folded into one affine grid_sample per crop index
  * the intensity transforms of DataAugmentationCTiBOT2 (same order, probabilities and parameter
    ranges as batchgeneratorsv2) are applied per sample on the batched crops

Two options change the recipe and default to the original behaviour:
  intensity_norm='minmax'    per-volume min-max (original) | 'hu_window' fixed HU window -> [0, 1]
  crop_inside=False          crop centres anywhere in the volume (original) | keep crops inside
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from SpatialTransformRatioCrop import create_affine_matrix_3d


def _bg_contrast(rng, lo, hi):
    # batchgeneratorsv2.BGContrast: below/above 1 with equal probability
    if rng.random() < 0.5 and lo < 1:
        return rng.uniform(lo, 1)
    return rng.uniform(max(lo, 1), hi)


def _gaussian_kernel(sigma, truncate=6.0):
    # batchgeneratorsv2.transforms.noise.gaussian_blur._build_kernel with blur_dimension's truncate=6
    n = sigma * truncate + 0.5
    r = round(n)
    ksize = r if r % 2 == 1 else (r + 1 if n - r >= 0 else r - 1)
    x = torch.linspace(-(ksize - 1) / 2, (ksize - 1) / 2, ksize)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


class GPUAugmentationCTiBOT:
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number,
                 global_size=224, local_size=96, intensity_norm="minmax", hu_window=(-1000.0, 1000.0),
                 crop_inside=False, seed=0):
        assert intensity_norm in ("minmax", "hu_window")
        self.global_crops_scale = tuple(global_crops_scale)
        self.local_crops_scale = tuple(local_crops_scale)
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.global_size = global_size
        self.local_size = local_size
        self.intensity_norm = intensity_norm
        self.hu_window = hu_window
        self.crop_inside = crop_inside
        # parameters are sampled on the CPU (cheap); its own generator keeps the global numpy RNG
        # (used e.g. by iBOTLoss to decide the CLS loss, identically on every rank) untouched
        self.rng = np.random.default_rng(seed)
        self.rotation = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.p_rotation, self.p_scaling, self.scaling = 0.2, 0.2, (0.7, 1.4)

    # ------------------------------------------------------------------ geometry
    def _crop_size(self, shape, volume_ratio):
        # SpatialTransformRatioCrop._calculate_crop_size
        x, y, z = shape
        a = int(self.rng.integers(math.ceil(x * volume_ratio), x + 1))
        min_b = math.ceil(x * y * volume_ratio / a)
        b = int(self.rng.integers(min_b, y + 1))
        c = round(x * y * z * volume_ratio / (a * b))
        return (a, b, c)

    def _sample_crop(self, shape, crop_scale):
        """Random crop size, centre and rotation/scaling, as SpatialTransformRatioCrop.get_parameters."""
        rng = self.rng
        volume_ratio = rng.uniform(*crop_scale)
        cs = np.array(self._crop_size(shape, volume_ratio), dtype=np.float64)
        do_rot = rng.uniform() < self.p_rotation
        do_scale = rng.uniform() < self.p_scaling
        angles = [rng.uniform(*self.rotation) for _ in range(3)] if do_rot else [0.0] * 3
        scales = [rng.uniform(*self.scaling)] * 3 if do_scale else [1.0] * 3
        A = create_affine_matrix_3d(angles, scales) if (do_rot or do_scale) else None
        s = np.array(shape, dtype=np.float64)
        if self.crop_inside:
            lo, hi = np.minimum(cs / 2, s / 2), np.maximum(s - cs / 2, s / 2)
            center = np.array([rng.uniform(l, h) for l, h in zip(lo, hi)])
        else:
            center = np.array([rng.uniform(0, si) for si in s])
        return cs, center, A

    @staticmethod
    def crop_theta(cs, center, A, padded_shape):
        """Affine (grid_sample convention, xyz order) mapping the output coords of one crop into the volume.

        Voxel index p of an output voxel with normalized coord u (align_corners=False):
            p = A^T diag(cs/2) u + center - 0.5        (same convention as SpatialTransformRatioCrop)
        normalized input coord on the padded volume: n = (2p + 1) / S_pad - 1"""
        if A is None:
            # crop_tensor + resize: the crop starts at floor(centre) - cs // 2
            A = np.eye(3)
            cs_ = np.asarray(cs, dtype=np.float64)
            center = np.floor(center) - np.floor(cs_ / 2) + cs_ / 2
        sp = np.array(padded_shape, dtype=np.float64)
        M = np.diag(2.0 / sp) @ np.asarray(A).T @ np.diag(np.asarray(cs, dtype=np.float64) / 2.0)
        t = 2.0 * np.asarray(center, dtype=np.float64) / sp - 1.0
        rev = [2, 1, 0]  # grid_sample expects (x, y, z) = (W, H, D)
        return np.concatenate([M[np.ix_(rev, rev)], t[rev][:, None]], axis=1)

    def _theta(self, shape, padded_shape, crop_scale):
        return self.crop_theta(*self._sample_crop(shape, crop_scale), padded_shape)

    # ------------------------------------------------------------------ intensity
    def _normalize(self, vol):
        vol = vol.float()
        if self.intensity_norm == "hu_window":
            lo, hi = self.hu_window
            return (vol.clamp(lo, hi) - lo) / (hi - lo)
        mn, mx = vol.amin(), vol.amax()
        return (vol - mn) / torch.clamp(mx - mn, min=1e-8)

    def _intensity(self, x):
        """x: (B, 1, S, S, S) float32 on GPU, modified in place where possible."""
        rng, B = self.rng, x.shape[0]
        S = x.shape[2:]
        # GaussianNoiseTransform(noise_variance=(0, 0.1)) -- the sampled value is used as the std
        for i in range(B):
            if rng.random() < 0.1:
                x[i].add_(torch.randn_like(x[i]) * rng.uniform(0, 0.1))
        # GaussianBlurTransform(sigma 0.5-1 per axis, p_per_channel 0.5), p=0.2
        for i in range(B):
            if rng.random() < 0.2 and rng.random() < 0.5:
                xi = x[i:i + 1]
                for d in range(3):
                    k = _gaussian_kernel(rng.uniform(0.5, 1.0)).to(x)
                    shape = [1, 1, 1, 1, 1]
                    shape[2 + d] = k.numel()
                    pad = [0] * 6
                    pad[2 * (2 - d)] = pad[2 * (2 - d) + 1] = k.numel() // 2
                    xi = F.conv3d(F.pad(xi, pad, mode="reflect"), k.view(shape))
                x[i:i + 1] = xi
        # MultiplicativeBrightnessTransform(BGContrast(0.75, 1.25)), p=0.15
        for i in range(B):
            if rng.random() < 0.15:
                x[i].mul_(_bg_contrast(rng, 0.75, 1.25))
        # ContrastTransform(BGContrast(0.75, 1.25), preserve_range=True), p=0.15
        for i in range(B):
            if rng.random() < 0.15:
                f = _bg_contrast(rng, 0.75, 1.25)
                mean, mn, mx = x[i].mean(), x[i].amin(), x[i].amax()
                x[i] = ((x[i] - mean) * f + mean).clamp_(mn, mx)
        # SimulateLowResolutionTransform(scale 0.5-1, synchronized axes, p_per_channel 0.5), p=0.25
        for i in range(B):
            if rng.random() < 0.25 and rng.random() < 0.5:
                sc = rng.uniform(0.5, 1.0)
                small = F.interpolate(x[i:i + 1], [round(s * sc) for s in S], mode="nearest-exact")
                x[i:i + 1] = F.interpolate(small, S, mode="trilinear")
        # GammaTransform(BGContrast(0.7, 1.5), retain stats): inverted with p=0.1, then plain with p=0.3
        for p_apply, invert in ((0.1, True), (0.3, False)):
            for i in range(B):
                if rng.random() < p_apply:
                    g = _bg_contrast(rng, 0.7, 1.5)
                    xi = -x[i] if invert else x[i]
                    mean, std = xi.mean(), xi.std()
                    mn = xi.amin()
                    rnge = xi.amax() - mn
                    xi = torch.pow((xi - mn) / torch.clamp(rnge, min=1e-7), g) * rnge + mn
                    xi = (xi - xi.mean()) * (std / torch.clamp(xi.std(), min=1e-7)) + mean
                    x[i] = -xi if invert else xi
        # MirrorTransform(axes 0, 1, 2 each with p=0.5), p=0.1
        for i in range(B):
            if rng.random() < 0.1:
                axes = [d + 1 for d in range(3) if rng.random() < 0.5]
                if axes:
                    x[i] = torch.flip(x[i], axes)
        return x.clamp_(0, 1)

    # ------------------------------------------------------------------ main entry
    @torch.no_grad()
    def __call__(self, volumes):
        """volumes: list of B tensors (1, D, H, W) (any dtype) already on the GPU.
        Returns a list of crops, each (B, 1, S, S, S) float32, global crops first (like the CPU transform)."""
        B = len(volumes)
        shapes = [tuple(v.shape[-3:]) for v in volumes]
        specs = [(self.global_size, self.global_crops_scale)] * self.global_crops_number + \
                [(self.local_size, self.local_crops_scale)] * self.local_crops_number
        # random geometry for every (crop index, sample) first, then the intensity draws
        theta = torch.tensor(np.stack([[self._theta(shapes[i], shapes[i], scale) for i in range(B)]
                                       for _, scale in specs]), dtype=torch.float32, device=volumes[0].device)
        # one volume at a time: padding the batch to a common shape would cost B x the largest volume in float32
        # (~10 GB at B=32); grid_sample's zero padding outside the volume gives the same crops
        crops = [volumes[0].new_empty((B, 1) + (size,) * 3, dtype=torch.float32) for size, _ in specs]
        groups = {size: [j for j, (s, _) in enumerate(specs) if s == size] for size, _ in specs}
        for i, v in enumerate(volumes):
            vol = self._normalize(v).reshape((1, 1) + shapes[i])
            for size, idx in groups.items():
                grid = F.affine_grid(theta[idx, i], (len(idx), 1, size, size, size), align_corners=False)
                out = F.grid_sample(vol.expand(len(idx), -1, -1, -1, -1), grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
                for j, o in zip(idx, out):
                    crops[j][i] = o
        return [self._intensity(c) for c in crops]
