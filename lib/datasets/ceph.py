import os
import json
import csv
import torch
import torch.utils.data as data
import numpy as np
import cv2

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

class CephDataset(data.Dataset):
    def __init__(self, cfg, is_train=True, transform=None):
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.pixel_std = 200
        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        mode = 'train' if is_train else 'test'

        self.img_dir = os.path.join(self.data_root, mode, 'Cephalograms')
        self.anno_dir = os.path.join(
            self.data_root, mode, 'Annotations',
            'Cephalometric Landmarks', 'Senior Orthodontists'
        )

        if not os.path.exists(self.img_dir):
            raise ValueError(f"Image directory {self.img_dir} does not exist.")

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(valid_extensions)])
        
        self.pixel_sizes = {}
        mapping_path = os.path.join(self.data_root, 'cephalogram_machine_mappings.csv')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.pixel_sizes[row['cephalogram_id']] = float(row['pixel_size'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
            
        data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        
        if data_numpy is None:
            raise ValueError(f"Fail to read {img_path}")

        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        h, w = data_numpy.shape[:2]

        base_name = os.path.splitext(img_name)[0]
        anno_path = os.path.join(self.anno_dir, base_name + '.json')
        with open(anno_path, 'r') as f:
            anno = json.load(f)

        pts = []
        for lm in anno['landmarks']:
            x = lm['value']['x']
            y = lm['value']['y']
            pts.append([x, y])
        
        pts = np.array(pts)
        
        min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
        
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        
        if max_x - min_x < 10 or max_y - min_y < 10:
            center_x = w / 2.0
            center_y = h / 2.0
            min_x, max_x, min_y, max_y = 0, w, 0, h

        center = torch.Tensor([center_x, center_y])
        scale = max((max_x - min_x), (max_y - min_y)) / self.pixel_std
        scale = scale * 1.5 

        r = 0
        if self.is_train:
            scale = scale * np.clip(np.random.randn() * self.scale_factor + 1, 1 - self.scale_factor, 1 + self.scale_factor)
            r = np.clip(np.random.randn() * self.rot_factor, -self.rot_factor * 2, self.rot_factor * 2) \
                if np.random.random() <= 0.6 else 0
            
            # Flip logic usually disabled for cephalometrics as it's asymmetrical profiles

        img = crop(data_numpy, center, scale, self.input_size, rot=r)

        target = np.zeros((self.num_joints, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(self.num_joints):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2], center,
                                               scale, self.output_size, rot=r)
                target[i] = generate_target(target[i], tpts[i], self.sigma,
                                            label_type=self.label_type)

        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        
        pixel_size = self.pixel_sizes.get(base_name, 0.1)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts, 'pixel_size': pixel_size}

        return img, target, meta
