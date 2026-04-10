import os
import sys
import contextlib
import copy
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Add current workspace to path so we can import 'lib'
sys.path.insert(0, os.path.abspath('.'))

from lib.models.hrnet import HighResolutionNet

app = FastAPI(title="HRNet Cephalometric Landmark Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
MODEL_LOADED = False
model = None

model_path = 'output/ceph_hrnet_notebook/best_model.pth'
NUM_JOINTS = 29
IMAGE_SIZE = (512, 512)
HEATMAP_SIZE = (128, 128)
USE_AMP = True

# Landmark symbol mapping (index to symbol)
LANDMARK_SYMBOLS = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "NA"
]

HRNET_W32_EXTRA = {
    "FINAL_CONV_KERNEL": 1,
    "STAGE2": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 2,
        "NUM_BLOCKS": [4, 4],
        "NUM_CHANNELS": [32, 64],
        "BLOCK": "BASIC",
        "FUSE_METHOD": "SUM",
    },
    "STAGE3": {
        "NUM_MODULES": 4,
        "NUM_BRANCHES": 3,
        "NUM_BLOCKS": [4, 4, 4],
        "NUM_CHANNELS": [32, 64, 128],
        "BLOCK": "BASIC",
        "FUSE_METHOD": "SUM",
    },
    "STAGE4": {
        "NUM_MODULES": 3,
        "NUM_BRANCHES": 4,
        "NUM_BLOCKS": [4, 4, 4, 4],
        "NUM_CHANNELS": [32, 64, 128, 256],
        "BLOCK": "BASIC",
        "FUSE_METHOD": "SUM",
    },
}


class AttrDict(dict):
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        value = self[key]
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[key] = value
        return value

    def __setattr__(self, key, value):
        self[key] = value


def build_hrnet_config(num_joints: int) -> AttrDict:
    cfg = AttrDict()
    cfg.MODEL = AttrDict()
    cfg.MODEL.NUM_JOINTS = num_joints
    cfg.MODEL.EXTRA = AttrDict(copy.deepcopy(HRNET_W32_EXTRA))
    cfg.MODEL.PRETRAINED = ''
    cfg.MODEL.INIT_WEIGHTS = False
    return cfg


def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, torch.nn.Module):
        return checkpoint_obj.state_dict()

    if isinstance(checkpoint_obj, dict):
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]
        return checkpoint_obj

    raise RuntimeError('Unsupported checkpoint format.')


def decode_heatmaps_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    b, j, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, j, -1)
    idx = flat.argmax(dim=-1)
    x = (idx % w).float()
    y = (idx // w).float()
    return torch.stack([x, y], dim=-1)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img_bgr is None:
        raise ValueError("Uploaded file is not a valid image.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

@app.on_event("startup")
def load_model():
    global model, MODEL_LOADED
    print("Loading model...")
    if not os.path.exists(model_path):
        print("Warning: Model checkpoint not found. Model will not be loaded.")
        return

    cfg = build_hrnet_config(num_joints=NUM_JOINTS)
    model = HighResolutionNet(cfg)

    loaded_obj = safe_torch_load(model_path)
    state_dict = extract_state_dict(loaded_obj)
    clean_state_dict = {
        (k[7:] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }

    missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
    if missing_keys or unexpected_keys:
        print(f"Checkpoint load report | missing: {len(missing_keys)} | unexpected: {len(unexpected_keys)}")

    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    MODEL_LOADED = True
    print("Model successfully loaded and ready for inference!")

def preprocess_image(img):
    if img is None:
        raise ValueError("Could not read image for preprocessing")

    h, w, _ = img.shape
    out_w, out_h = IMAGE_SIZE
    img_resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize with ImageNet stats
    img_resized = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_resized = (img_resized - mean) / std
    
    # Convert to PyTorch Tensor [Channels, Height, Width]
    img_tensor = torch.from_numpy(img_resized.transpose((2, 0, 1)))
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension -> [1, C, H, W]

    resize_factors = np.array([out_w / float(w), out_h / float(h)], dtype=np.float32)
    return img_tensor, w, h, resize_factors

def inference(image):
    if not MODEL_LOADED:
        raise RuntimeError("Model is not loaded.")

    img_tensor, original_w, original_h, resize_factors = preprocess_image(image)

    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    with torch.inference_mode():
        if img_tensor.is_cuda and hasattr(torch, 'autocast') and USE_AMP:
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            amp_ctx = contextlib.nullcontext()

        with amp_ctx:
            outputs = model(img_tensor)

    if isinstance(outputs, (list, tuple)):
        outputs = outputs[-1]

    preds_hm = decode_heatmaps_argmax(outputs)
    preds = preds_hm.clone()
    preds[..., 0] *= IMAGE_SIZE[0] / float(HEATMAP_SIZE[0])
    preds[..., 1] *= IMAGE_SIZE[1] / float(HEATMAP_SIZE[1])

    preds[..., 0] /= resize_factors[0]
    preds[..., 1] /= resize_factors[1]

    confidence = outputs.detach().reshape(outputs.shape[0], outputs.shape[1], -1).max(dim=-1).values

    predicted_points = preds[0]
    confidence = confidence[0]

    landmarks = []
    for i in range(len(predicted_points)):
        symbol = LANDMARK_SYMBOLS[i] if i < len(LANDMARK_SYMBOLS) else f"L{i+1}"
        landmarks.append({
            "symbol": symbol,
            "value": {
                "x": float(predicted_points[i][0]),
                "y": float(predicted_points[i][1])
            },
            "confidence": float(confidence[i])
        })

    return landmarks, int(original_w), int(original_h), None

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = _decode_image(image_bytes)

        landmarks, width, height, roi_bbox = inference(image)

        # Format response to match frontend expectations
        response = {
            "landmarks": landmarks
        }

        return JSONResponse(content=response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
