import os
import sys
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile

# Add current workspace to path so we can import 'lib'
sys.path.insert(0, os.path.abspath('.'))

from lib.config import config, update_config
import lib.models as models
from lib.core.evaluation import decode_preds
from lib.utils.transforms import crop

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

class Args:
    cfg = cfg_path

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
        
    model.load_state_dict(state_dict)
    model.eval()
    
    if torch.cuda.is_available():
        model.cuda()
        
    MODEL_LOADED = True
    print("Model successfully loaded and ready for inference!")

def preprocess_image(image_path, input_size):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    # Fallback to whole-image center/scale for inference API 
    center_x = w / 2.0
    center_y = h / 2.0
    center = torch.Tensor([center_x, center_y])
    scale = max(w, h) / 200.0 * 1.5 
    
    # Crop using lib.utils.transforms
    img_cropped = crop(img, center, scale, input_size, rot=0)
    
    # Normalize with ImageNet stats
    img_cropped = img_cropped.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_cropped = (img_cropped - mean) / std
    
    # Convert to PyTorch Tensor [Channels, Height, Width]
    img_tensor = torch.from_numpy(img_cropped.transpose((2, 0, 1)))
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension -> [1, C, H, W]
    
    return img_tensor, img, center, scale

def inference(image_path):
    if not MODEL_LOADED:
        raise RuntimeError("Model is not loaded.")
        
    input_size = config.MODEL.IMAGE_SIZE
    img_tensor, original_img, center, scale = preprocess_image(image_path, input_size)
    
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
        
    with torch.no_grad():
        outputs = model(img_tensor)
        
    heatmap_size = config.MODEL.HEATMAP_SIZE
    preds = decode_preds(
        outputs.detach().cpu(), 
        center.unsqueeze(0).numpy(), 
        np.array([scale], dtype=np.float32), 
        heatmap_size
    )
    
    predicted_points = preds[0]
    
    landmarks = []
    for i in range(len(predicted_points)):
        landmarks.append({
            "id": i + 1,
            "x": float(predicted_points[i][0]),
            "y": float(predicted_points[i][1])
        })
        
    return landmarks, original_img.shape[1], original_img.shape[0]

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file to a temporary file
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())
            
        landmarks, width, height = inference(temp_path)
        
        # Cleanup
        os.remove(temp_path)
            
        return JSONResponse(content={"landmarks": landmarks, "image_size": {"width": width, "height": height}})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
