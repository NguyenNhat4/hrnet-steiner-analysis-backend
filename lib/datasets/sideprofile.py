import os
import json
import torch
import torch.utils.data as data
import numpy as np
import cv2

from ..utils.transforms import fliplr_joints, crop, generate_target

class SideProfile(data.Dataset):
    def __init__(self, cfg, is_train=True):
        self.num_joints = 8
        self.pixel_std = 200
        self.flip = cfg.DATASET.FLIP
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA

        self.is_train = is_train

        self.root = cfg.DATASET.ROOT
        self.json_file = cfg.DATASET.TRAINSET if is_train else cfg.DATASET.TESTSET

        # Load data
        with open(os.path.join(self.root, self.json_file), 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # It assumes the images are stored in self.root/images or directly self.root
        # I'll default to searching inside self.root directly as image names are given
        # or checking where the images actually are.
        img_path = os.path.join(self.root, item['image'])
        # if not found, let's try 'images' folder
        if not os.path.exists(img_path):
            img_path = os.path.join(self.root, 'images', item['image'])
            
        data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        
        if data_numpy is None:
            raise ValueError(f"Fail to read {img_path}")

        pts = np.array(item['pts'])
        c = np.array(item['center'])
        s = item['scale']

        r = 0
        if self.is_train:
            s = s * np.clip(np.random.randn() * self.scale_factor + 1, 1 - self.scale_factor, 1 + self.scale_factor)
            r = np.clip(np.random.randn() * self.rot_factor, -self.rot_factor * 2, self.rot_factor * 2) \
                if np.random.random() <= 0.6 else 0
            
            # Flip logic: You might want to disable this for side profiles. We leave flip=False in config usually.
            if self.flip and np.random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                c[0] = data_numpy.shape[1] - c[0] - 1
                # we skip fliplr_joints symmetry pairs here since it's side-profile 
                # (unless sym is defined) - usually for profile we keep flip off.

        trans = crop(c, s, [self.image_size[0], self.image_size[1]], rot=r)
        
        img = cv2.warpAffine(data_numpy, trans, (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        for i in range(self.num_joints):
            if pts[i, 0] > 0:
                pts[i, 0:2] = np.dot(trans, np.array([pts[i, 0], pts[i, 1], 1.0]))

        target, target_weight = generate_target(pts, pts[:,0]*0+1, self.image_size, self.heatmap_size, self.sigma)
        
        # Normalize
        img = torch.Tensor(img)
        # convert to [C, H, W]
        img = img.permute(2, 0, 1)

        return img, target, target_weight

