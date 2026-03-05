"""Finger-count prediction from hand crops.

Supports two backends:
  - PyTorch (.pth checkpoint)
  - ONNX Runtime (.onnx model)

Public API:
    load_finger_model(checkpoint_path, device) -> (model, device)
    predict_finger_count(model, frame_bgr, wrist_xy, device) -> (pred, conf, probs, crop)
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

_FINGERS_DIR = Path(os.environ.get(
    "FINGERS_DIR",
    str(Path(__file__).resolve().parent / "model"),
))

_ONNX_PATH = _FINGERS_DIR / "checkpoints" / "golf_finetuned_efficientnet_b0_v1.onnx"
_PTH_PATH = _FINGERS_DIR / "checkpoints" / "golf_finetuned_efficientnet_b0_v1.pth"
DEFAULT_CHECKPOINT = _ONNX_PATH if _ONNX_PATH.exists() else _PTH_PATH

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CROP_PAD_ABOVE = 200
CROP_PAD_BELOW = 80
CROP_PAD_LEFT = 150
CROP_PAD_RIGHT = 150
REFERENCE_ARM_LENGTH = 481.0
SCALE_MIN = 0.4
SCALE_MAX = 1.5


def _preprocess_numpy(crop_bgr):
    crop_rgb = crop_bgr[:, :, ::-1]
    pil_img = Image.fromarray(crop_rgb).resize((224, 224), Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[np.newaxis]


def _softmax(logits):
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _compute_arm_scale(pkl_data, frame_idx, side, conf_threshold=0.3):
    fk = f"frame_{frame_idx}"
    if fk not in pkl_data:
        return 1.0
    kps = pkl_data[fk]["keypoints"]
    scores = pkl_data[fk].get("keypoint_scores")
    sh, wr = (5, 9) if side == "LEFT" else (6, 10)
    if scores is not None and (scores[sh] < conf_threshold or scores[wr] < conf_threshold):
        return 1.0
    arm_len = float(np.sqrt((kps[wr][0] - kps[sh][0]) ** 2 + (kps[wr][1] - kps[sh][1]) ** 2))
    if arm_len < 1.0:
        return 1.0
    return float(np.clip(arm_len / REFERENCE_ARM_LENGTH, SCALE_MIN, SCALE_MAX))


def load_finger_model(checkpoint_path=None, device=None):
    """Load finger-counting model."""
    checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)

    if checkpoint_path.suffix == ".onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(str(checkpoint_path), providers=["CPUExecutionProvider"])
        return sess, "onnx"

    import torch
    import sys as _sys
    if str(_FINGERS_DIR) not in _sys.path:
        _sys.path.insert(0, str(_FINGERS_DIR))
    from finetune import create_model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_name="efficientnet_b0", num_classes=6, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, device


def _compute_hand_crop(frame_bgr, wrist_x, wrist_y, scale=1.0):
    h, w = frame_bgr.shape[:2]
    x1 = max(0, int(wrist_x - CROP_PAD_LEFT * scale))
    y1 = max(0, int(wrist_y - CROP_PAD_ABOVE * scale))
    x2 = min(w, int(wrist_x + CROP_PAD_RIGHT * scale))
    y2 = min(h, int(wrist_y + CROP_PAD_BELOW * scale))
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def predict_finger_count(model, frame_bgr, wrist_xy, device, scale=1.0):
    """Predict finger count from a single frame + wrist location."""
    crop_bgr = _compute_hand_crop(frame_bgr, wrist_xy[0], wrist_xy[1], scale=scale)
    if crop_bgr is None:
        return None, None, None, None

    inp = _preprocess_numpy(crop_bgr)
    if device == "onnx":
        logits = model.run(None, {"input": inp})[0]
        probs = _softmax(logits)[0]
    else:
        import torch
        with torch.no_grad():
            logits = model(torch.from_numpy(inp).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    predicted = int(np.argmax(probs))
    return predicted, float(probs[predicted]), probs, crop_bgr
