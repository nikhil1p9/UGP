import torch
import numpy as np
from drive_text.lane import CLRLaneEstimator

def verify_installation():
    print("--- Starting CLRNet Verification ---")
    
    # 1. Test C++ Extensions (NMS)
    try:
        from clrnet.ops import nms
        print("[PASS] CLRNet C++ extensions (NMS) loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Could not load CLRNet extensions: {e}")
        return

    # 2. Test Model Initialization
    # These paths must match your config.py settings
    config_path = "configs/clrnet/clr_dla34_culane.py"
    weight_path = "weights/culane_dla34.pth"
    
    try:
        estimator = CLRLaneEstimator(config_path, weight_path)
        print(f"[PASS] Model initialized on device: {estimator.device}")
    except FileNotFoundError as e:
        print(f"[FAIL] Missing file: {e}")
        return
    except Exception as e:
        print(f"[FAIL] Initialization error: {e}")
        return

    # 3. Test Dummy Inference
    try:
        # Create a blank dummy image (H, W, C)
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = estimator.estimate([dummy_frame])
        print(f"[PASS] Inference test successful. Source: {result.source}")
        print("--- All systems are green! ---")
    except Exception as e:
        print(f"[FAIL] Inference failed: {e}")

if __name__ == "__main__":
    verify_installation()