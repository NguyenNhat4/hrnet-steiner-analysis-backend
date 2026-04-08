import os
import json
import csv
import torch
import torch.utils.data as data
import numpy as np
import cv2

from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

class CephDataset(data.Dataset):
    def __init__(self, cfg, is_train=True, split=None, transform=None):
        """
        Args:
            cfg: config object
            is_train: controls augmentation (True = augment, False = no augment)
            split: directory to load from ('train', 'valid', 'test').
                   If None, defaults to 'train' when is_train=True, 'valid' when is_train=False.
        """
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

        if split is not None:
            mode = split
        elif is_train:
            mode = 'train'
        else:
            mode = 'valid'

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

    def _read_image(self, img_path):
        data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if data_numpy is None:
            raise ValueError(f"Fail to read {img_path}")
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        h, w = data_numpy.shape[:2]
        return data_numpy, w, h

    def _read_annotations(self, anno_path):
        with open(anno_path, 'r') as f:
            anno = json.load(f)

        pts = []
        for lm in anno['landmarks']:
            x = lm['value']['x']
            y = lm['value']['y']
            pts.append([x, y])
        return np.array(pts)

    def _get_bounding_box(self, pts, w, h):
        min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
        
        # Center tightly on the landmarks but use a much larger scale to fit the whole head
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        
        if max_x - min_x < 10 or max_y - min_y < 10:
            center_x = w / 2.0
            center_y = h / 2.0
            min_x, max_x, min_y, max_y = 0, w, 0, h

        center = torch.Tensor([center_x, center_y])
        # Increase the multiplier significantly so we don't 'crop half of the ceph xray'
        scale = max((max_x - min_x), (max_y - min_y)) / self.pixel_std
        scale = scale * 2
        return center, scale

    def _apply_augmentation(self, scale):
        rot = 0
        flip = False
        
        if self.is_train:
            scale = scale * np.clip(np.random.randn() * self.scale_factor + 1, 
                                    1 - self.scale_factor, 1 + self.scale_factor)
            
            if np.random.random() <= 0.6:
                # Restrict rotation to a very slight amount (max 10 degrees) instead of relying on the extreme rot_factor
                rot = np.clip(np.random.randn() * self.rot_factor, -10, 10)
            
            if self.flip and np.random.random() <= 0.5:
                flip = True
                
        return scale, rot, flip

    def _generate_targets(self, pts, center, scale, rot):
        target = np.zeros((self.num_joints, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(self.num_joints):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2], center,
                                               scale, self.output_size, rot=rot)
                target[i] = generate_target(target[i], tpts[i], self.sigma,
                                            label_type=self.label_type)
        return torch.Tensor(target), torch.Tensor(tpts)

    def _apply_intensity_augmentation(self, img):
        """Apply intensity augmentations suitable for X-ray images.
        Operates on uint8 HWC image, returns uint8 HWC image."""
        img = img.astype(np.float32)

        # Brightness & contrast jitter (±20%)
        alpha = 1.0 + np.random.uniform(-0.2, 0.2)  # contrast
        beta = np.random.uniform(-20, 20)              # brightness
        img = np.clip(alpha * img + beta, 0, 255)

        # Random gamma correction (0.8 – 1.2)
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.8, 1.2)
            inv_gamma = 1.0 / gamma
            img = 255.0 * (img / 255.0) ** inv_gamma

        # Gaussian blur (occasional)
        if np.random.random() < 0.3:
            ksize = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img.astype(np.uint8), (ksize, ksize), 0).astype(np.float32)

        return np.clip(img, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        anno_path = os.path.join(self.anno_dir, base_name + '.json')

        # 1. Load image and annotations
        data_numpy, w, h = self._read_image(img_path)
        pts = self._read_annotations(anno_path)
        
        # 2. Get initial bounding box
        center, scale = self._get_bounding_box(pts, w, h)
        
        # 3. Apply data augmentation
        scale, r, flip = self._apply_augmentation(scale)
        if flip:
            data_numpy = data_numpy[:, ::-1, :]
            pts[:, 0] = w - 1 - pts[:, 0]
            center[0] = w - 1 - center[0]

        # 4. Crop image
        from ..utils.transforms import crop_v2
        center_np = center.numpy() if isinstance(center, torch.Tensor) else center
        img = crop_v2(data_numpy, center_np, scale, self.input_size, rot=r)
        
        # 5. Generate target heatmaps
        # Also ensure center is passed as numpy to transform_pixel (which is deep inside _generate_targets)
        target, tpts = self._generate_targets(pts, center_np, scale, r)

        # 6. Apply intensity augmentation (no landmark re-mapping needed)
        if self.is_train:
            img = self._apply_intensity_augmentation(img)

        # 7. Normalize image tensor
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])

        pixel_size = self.pixel_sizes.get(base_name, 0.1)
        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts, 'pixel_size': pixel_size}

        return img, target, meta
