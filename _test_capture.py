"""Test dxcam capture for MuMu Player"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygetwindow as gw
import numpy as np
from PIL import Image

# Find MuMu window
wins = gw.getWindowsWithTitle("MuMuPlayer")
if not wins:
    print("MuMuPlayer not found"); sys.exit(1)
w = wins[0]
print(f"Window: {w.title} at ({w.left},{w.top}) {w.width}x{w.height}")

# Test dxcam (Desktop Duplication API)
try:
    import dxcam
    cam = dxcam.create()
    # Grab the region where MuMu is
    region = (w.left, w.top, w.left + w.width, w.top + w.height)
    frame = cam.grab(region=region)
    if frame is not None:
        print(f"dxcam OK: shape={frame.shape}, mean={frame.mean():.1f}")
        Image.fromarray(frame).save("_debug_dxcam.png")
        print("Saved _debug_dxcam.png")
    else:
        print("dxcam returned None")
    del cam
except Exception as e:
    print(f"dxcam error: {e}")
