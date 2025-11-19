import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, scale=4, patch_size=96, augment=True):
        """
        lr_dir : directory containing LR images (bicubic x2/x3/x4)
        hr_dir : directory containing HR images
        scale  : upscaling factor
        patch_size : HR patch size to crop
        augment : whether to use random flips/rotations
        """

        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_files = sorted(os.listdir(lr_dir))
        self.hr_files = sorted(os.listdir(hr_dir))

        self.scale = scale
        self.patch_size = patch_size
        self.lr_patch = patch_size // scale
        self.augment_flag = augment

        # LR remains in [0,1]
        self.to_tensor_lr = T.ToTensor()

        # HR should be in [-1,1]
        self.to_tensor_hr = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)   # (x - 0.5)/0.5
        ])

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        # Load image paths
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])

        # Load PIL images
        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        # MUST match DIV2K scale alignment (safety check)
        assert hr.width == lr.width * self.scale
        assert hr.height == lr.height * self.scale

        # ---- CROP ----
        hr, lr = self.random_crop(hr, lr)

        # ---- AUGMENT ----
        if self.augment_flag:
            hr, lr = self.augment(hr, lr)

        # ---- Convert to tensor ----
        lr = self.to_tensor_lr(lr)   # [0,1]
        hr = self.to_tensor_hr(hr)   # [-1,1]

        return lr, hr

    # ========================
    # Paired random crop
    # ========================
    def random_crop(self, hr, lr):
        w, h = hr.size

        # Random HR crop origin
        x = random.randrange(0, w - self.patch_size)
        y = random.randrange(0, h - self.patch_size)

        # Crop HR
        hr_patch = hr.crop((x, y, x + self.patch_size, y + self.patch_size))

        # Map HR coordinates → LR coordinates
        lr_x = x // self.scale
        lr_y = y // self.scale

        lr_patch = lr.crop((
            lr_x, lr_y,
            lr_x + self.lr_patch,
            lr_y + self.lr_patch
        ))

        return hr_patch, lr_patch

    # ========================
    # Flip + rotate (paired)
    # ========================
    def augment(self, hr, lr):
        # Horizontal flip
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)

        # Vertical flip
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
            lr = lr.transpose(Image.FLIP_TOP_BOTTOM)

        # 90-degree rotation
        if random.random() < 0.5:
            hr = hr.transpose(Image.ROTATE_90)
            lr = lr.transpose(Image.ROTATE_90)

        return hr, lr
