# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import math
import multiprocessing as mp
import numpy as np
from torchvision.datasets import ImageFolder

import os
import PIL
import SimpleITK as sitk
import torch
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset

class CTDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = PIL.Image.open(img_path)
        image = pil_to_tensor(image).float().unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        return image

class CTFolderMask(CTDataset):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(CTFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch
        # epoch in shared memory: with persistent DataLoader workers every worker keeps its own copy of the
        # dataset, so a plain attribute set by set_epoch() would never reach them (pred_start_epoch would never start)
        self._epoch = mp.RawValue('i', -1)

    def get_pred_ratio(self):
        if 0 <= self._epoch.value < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio

        return pred_ratio

    def set_epoch(self, epoch):
        self._epoch.value = epoch

    def __getitem__(self, index):
        output = super(CTFolderMask, self).__getitem__(index)

        masks = []
        for img in output:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output, masks)

VOLUME_EXTENSIONS = ('.nii', '.nii.gz', '.npy')


def list_volumes(data_path):
    """A directory of volumes, or a .txt file with one path per line (absolute or relative to it)."""
    if data_path.endswith('.txt'):
        root = os.path.dirname(os.path.abspath(data_path))
        with open(data_path) as f:
            return [os.path.join(root, line.strip()) for line in f if line.strip()]
    return sorted(os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(VOLUME_EXTENSIONS))


def load_volume(path):
    """Volume as a (z, y, x) numpy array. .npy files are the preprocessed int16 cache (tools/build_ct_cache.py)."""
    if path.endswith('.npy'):
        return np.load(path)
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def worker_init_fn(worker_id):
    # ITK otherwise spawns one thread per core in every DataLoader worker
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


class CTDataset3D(Dataset):
    def __init__(self, image_folder, transform=None, batch_size=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = list_volumes(image_folder)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = load_volume(self.image_files[idx])
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        return image

class CTFolderMask3D(CTDataset3D):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(CTFolderMask3D, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch
        # epoch in shared memory: with persistent DataLoader workers every worker keeps its own copy of the
        # dataset, so a plain attribute set by set_epoch() would never reach them (pred_start_epoch would never start)
        self._epoch = mp.RawValue('i', -1)

    def get_pred_ratio(self):
        if 0 <= self._epoch.value < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio

        return pred_ratio

    def set_epoch(self, epoch):
        self._epoch.value = epoch

    def make_mask(self, D, H, W):
        high = self.get_pred_ratio() * D * H * W

        if self.pred_shape == 'block':
            # following BEiT (https://arxiv.org/abs/2106.08254), see at
            # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
            mask = np.zeros((D, H, W), dtype=bool)
            mask_count = 0
            while mask_count < high:
                max_mask_patches = high - mask_count

                delta = 0
                for attempt in range(10):
                    low = (min(D, H, W) // 3) ** 3
                    target_volume = random.uniform(low, max_mask_patches)
                    aspect_ratio_1 = math.exp(random.uniform(*self.log_aspect_ratio))
                    aspect_ratio_2 = math.exp(random.uniform(*self.log_aspect_ratio))
                    h = int(round((target_volume * aspect_ratio_2 / aspect_ratio_1) ** (1/3)))
                    d = int(round(h * aspect_ratio_1))
                    w = int(round(h / aspect_ratio_2))
                    if d < D and h < H and w < W and d > 0 and h > 0 and w > 0:
                        front = random.randint(0, D - d)
                        top = random.randint(0, H - h)
                        left = random.randint(0, W - w)

                        num_masked = mask[front:front + d, top:top + h, left:left + w].sum()
                        if 0 < d * h * w - num_masked <= max_mask_patches:
                            for i in range(front, front + d):
                                for j in range(top, top + h):
                                    for k in range(left, left + w):
                                        if mask[i, j, k] == 0:
                                            mask[i, j, k] = 1
                                            delta += 1

                    if delta > 0:
                        break

                if delta == 0:
                    break
                else:
                    mask_count += delta

        elif self.pred_shape == 'rand':
            total_elements = H * W * D
            mask = np.hstack([
                np.zeros(total_elements - int(high)),
                np.ones(int(high)),
            ]).astype(bool)
            np.random.shuffle(mask)
            mask = mask.reshape(D, H, W)
        else:
            # no implementation
            assert False
        return mask

    def __getitem__(self, index):
        output = super(CTFolderMask3D, self).__getitem__(index)

        masks = []
        for img in output:
            try:
                D, H, W = img.shape[-3] // self.psz, img.shape[-2] // self.psz, img.shape[-1] // self.psz
            except:
                # skip non-image
                continue
            masks.append(self.make_mask(D, H, W))

        return (output, masks)


class CTVolumeMask3D(CTFolderMask3D):
    """For --gpu_aug: returns the raw volume (augmented later on the GPU, see gpu_transforms.py) and the
    MIM masks for each crop. `crop_sizes` lists the output size of every crop (global crops first)."""
    def __init__(self, *args, crop_sizes, **kwargs):
        super(CTVolumeMask3D, self).__init__(*args, **kwargs)
        self.crop_sizes = crop_sizes

    def __getitem__(self, index):
        image = load_volume(self.image_files[index])
        if image.dtype != np.int16:
            # CT is integer HU; int16 halves host->device traffic vs float32 (and is 4x smaller than float64)
            image = np.clip(np.rint(image), -32768, 32767).astype(np.int16)
        image = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        masks = [self.make_mask(*(3 * (s // self.psz,))) for s in self.crop_sizes]
        return image, masks


def collate_volumes(batch):
    """Volumes have different shapes: keep them as a list; stack the masks per crop index."""
    volumes = [b[0] for b in batch]
    masks = [torch.from_numpy(np.stack([b[1][k] for b in batch])) for k in range(len(batch[0][1]))]
    return volumes, masks

class CTFolderMaskSliceWise(CTFolderMask3D):
    def __getitem__(self, index):
        output = super(CTFolderMask3D, self).__getitem__(index)

        masks = []
        for img in output:
            try:
                D, H, W = math.ceil(img.shape[-3] / self.psz), img.shape[-2] // self.psz, img.shape[-1] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((D, H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[:, top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if not mask[:, i, j].any():
                                            mask[:, i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)
                mask = np.tile(mask, (D, 1, 1))
            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output, masks)

class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index

class ImageFolderMask(ImageFolder):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio

        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(ImageFolderMask, self).__getitem__(index)

        masks = []
        for img in output[0]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return output + (masks,)