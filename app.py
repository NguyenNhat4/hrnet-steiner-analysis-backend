import os
import sys
import contextlib
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Tuple

# Add current workspace to path so we can import 'lib'
sys.path.insert(0, os.path.abspath('.'))

from lib.config import config, update_config
import lib.models as models
from lib.core.evaluation import decode_preds
from lib.utils.transforms import crop_v2, transform_preds

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

cfg_path = 'experiments/ceph/face_alignment_ceph_hrnet.yaml'
model_path = 'output/CephDataset/face_alignment_ceph_hrnet/model_best.pth'
SCALE_MULTIPLIER = 2.0
ROI_AREA_FRACTION_THRESHOLD = 0.05
SOFTARGMAX_BETA = 24.0

class Args:
    cfg = cfg_path


def _largest_component_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    # Ignore background index 0.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
    y = int(stats[largest_idx, cv2.CC_STAT_TOP])
    w = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])
    return x, y, x + w, y + h


def estimate_head_roi(img_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Estimate cephalogram head ROI using morphology + largest component."""
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Pick the version with less foreground area to avoid selecting the background.
    fg_ratio = float(np.mean(binary > 0))
    if fg_ratio > 0.5:
        binary = 255 - binary

    kernel = np.ones((9, 9), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    bbox = _largest_component_bbox(binary)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area_ratio = (bw * bh) / float(max(1, h * w))
    if area_ratio < ROI_AREA_FRACTION_THRESHOLD:
        return None

    # Expand box to keep surrounding structure and avoid over-tight ROI.
    pad_w = int(0.12 * bw)
    pad_h = int(0.12 * bh)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w - 1, x2 + pad_w)
    y2 = min(h - 1, y2 + pad_h)

    return x1, y1, x2, y2


def _softargmax_decode(output: torch.Tensor, center: np.ndarray, scale: np.ndarray, heatmap_size):
    """Decode heatmaps with soft-argmax in heatmap coordinates, then map to image space."""
    score_map = output.detach().cpu().float()
    n, c, h, w = score_map.shape

    flat = score_map.view(n, c, -1)
    probs = torch.softmax(flat * SOFTARGMAX_BETA, dim=-1).view(n, c, h, w)

    xs = torch.arange(1, w + 1, dtype=torch.float32).view(1, 1, 1, w)
    ys = torch.arange(1, h + 1, dtype=torch.float32).view(1, 1, h, 1)

    x_coords = (probs * xs).sum(dim=(2, 3))
    y_coords = (probs * ys).sum(dim=(2, 3))
    coords = torch.stack([x_coords, y_coords], dim=-1)

    preds = coords.clone()
    for i in range(n):
        preds[i] = transform_preds(coords[i], center[i], scale[i], heatmap_size)

    confidence = torch.max(probs.view(n, c, -1), dim=-1).values
    return preds, confidence


def decode_preds_fused(output: torch.Tensor, center: np.ndarray, scale: np.ndarray, heatmap_size):
    """Fuse classic argmax decode with soft-argmax decode for better sub-pixel localization."""
    preds_argmax = decode_preds(output.detach().cpu(), center, scale, heatmap_size)
    preds_soft, confidence = _softargmax_decode(output, center, scale, heatmap_size)

    alpha = confidence.clamp(0.20, 0.80).unsqueeze(-1)
    preds = preds_argmax * (1.0 - alpha) + preds_soft * alpha
    return preds, confidence


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
    if not os.path.exists(cfg_path) or not os.path.exists(model_path):
        print("Warning: Config or model checkpoint not found. Model will not be loaded.")
        return
        
    args = Args()
    update_config(config, args)
    
    # Initialize blank HRNet model
    model = models.get_face_alignment_net(config)
    
    # Load weights
    loaded_obj = torch.load(model_path, weights_only=False, map_location='cpu')
    
    if isinstance(loaded_obj, torch.nn.Module):
        state_dict = loaded_obj.state_dict()
    else:
        state_dict = loaded_obj
        
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    if isinstance(state_dict, dict):
        state_dict = {
            (k[7:] if k.startswith('module.') else k): v
            for k, v in state_dict.items()
        }
        
    model.load_state_dict(state_dict)
    model.eval()
    
    if torch.cuda.is_available():
        model.cuda()
        
    MODEL_LOADED = True
    print("Model successfully loaded and ready for inference!")

def preprocess_image(img, input_size):
    if img is None:
        raise ValueError("Could not read image for preprocessing")

    h, w, _ = img.shape

    roi_bbox = estimate_head_roi(img)
    if roi_bbox is None:
        center_x = w / 2.0
        center_y = h / 2.0
        crop_w = w
        crop_h = h
    else:
        x1, y1, x2, y2 = roi_bbox
        crop_w = max(1, x2 - x1)
        crop_h = max(1, y2 - y1)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

    center = torch.tensor([center_x, center_y], dtype=torch.float32)
    scale = max(crop_w, crop_h) / 200.0 * SCALE_MULTIPLIER

    # Use crop_v2 to match the dataset preprocessing path.
    img_cropped = crop_v2(img, center.numpy(), scale, input_size, rot=0)
    
    # Normalize with ImageNet stats
    img_cropped = img_cropped.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_cropped = (img_cropped - mean) / std
    
    # Convert to PyTorch Tensor [Channels, Height, Width]
    img_tensor = torch.from_numpy(img_cropped.transpose((2, 0, 1)))
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension -> [1, C, H, W]
    
    return img_tensor, img, center, scale, roi_bbox

def inference(image):
    if not MODEL_LOADED:
        raise RuntimeError("Model is not loaded.")

    input_size = config.MODEL.IMAGE_SIZE
    img_tensor, original_img, center, scale, roi_bbox = preprocess_image(image, input_size)

    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    with torch.inference_mode():
        if img_tensor.is_cuda and hasattr(torch, 'autocast'):
            amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            amp_ctx = contextlib.nullcontext()

        with amp_ctx:
            outputs = model(img_tensor)

    if isinstance(outputs, (list, tuple)):
        outputs = outputs[-1]

    heatmap_size = config.MODEL.HEATMAP_SIZE
    preds, confidence = decode_preds_fused(
        outputs,
        center.unsqueeze(0).numpy(),
        np.array([scale], dtype=np.float32),
        heatmap_size,
    )

    predicted_points = preds[0]
    confidence = confidence[0]

    landmarks = []
    for i in range(len(predicted_points)):
        landmarks.append({
            "id": i + 1,
            "x": float(predicted_points[i][0]),
            "y": float(predicted_points[i][1]),
            "confidence": float(confidence[i])
        })

    return landmarks, original_img.shape[1], original_img.shape[0], roi_bbox

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = _decode_image(image_bytes)

        landmarks, width, height, roi_bbox = inference(image)

        response = {
            "landmarks": landmarks,
            "image_size": {"width": width, "height": height},
        }

        if roi_bbox is not None:
            x1, y1, x2, y2 = roi_bbox
            response["roi_bbox"] = {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}

        return JSONResponse(content=response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
