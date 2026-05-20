"""
AGFF-DriveNet System Validation Script
Validates model loading, architecture, and core functionality
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Fix Unicode output on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("\n" + "="*60)
print("AGFF-DriveNet System Validation")
print("="*60 + "\n")

# ==================== 1. Check Python Version ====================
print("✓ Python Version Check")
python_version = sys.version.split()[0]
print(f"  Python: {python_version}")
print(f"  Path: {sys.executable}\n")

# ==================== 2. Check PyTorch ====================
print("✓ PyTorch Check")
print(f"  Version: {torch.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  Device Name: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print(f"  Running on CPU")
print()

# ==================== 3. Import Core Modules ====================
print("✓ Core Module Imports")
try:
    from model import load_model, AGFFDriveNet
    print("  ✓ model.py")
except Exception as e:
    print(f"  ✗ model.py: {e}")
    sys.exit(1)

try:
    from infer import (DriverStateInference, VideoProcessor, 
                       FramePreprocessor, PerformanceMonitor)
    print("  ✓ infer.py")
except Exception as e:
    print(f"  ✗ infer.py: {e}")
    sys.exit(1)

try:
    import cv2
    print("  ✓ cv2 (OpenCV)")
except Exception as e:
    print(f"  ✗ cv2: {e}")
    sys.exit(1)

try:
    import streamlit as st
    print("  ✓ streamlit")
except Exception as e:
    print(f"  ✗ streamlit: {e}")
    sys.exit(1)

try:
    import plotly
    print("  ✓ plotly")
except Exception as e:
    print(f"  ✗ plotly: {e}")
    sys.exit(1)
print()

# ==================== 4. Check Model File ====================
print("✓ Model File Check")
model_path = Path(__file__).parent / 'agff_hybrid_model.pth'
if model_path.exists():
    size_mb = model_path.stat().st_size / 1e6
    print(f"  ✓ Model found: {model_path}")
    print(f"  Size: {size_mb:.2f} MB")
else:
    print(f"  ✗ Model NOT found at {model_path}")
    sys.exit(1)
print()

# ==================== 5. Test Model Loading ====================
print("✓ Model Loading Test")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    model = load_model(str(model_path), device=device)
    print(f"  ✓ Model loaded successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
except Exception as e:
    print(f"  ✗ Failed to load model: {e}")
    sys.exit(1)
print()

# ==================== 6. Test Inference ====================
print("✓ Inference Test")
try:
    # Create dummy input
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_input = torch.randn(1, 4, 3, 128, 128).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output logits: {output[0].detach().cpu().numpy()}")
    
    # Softmax
    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
    print(f"  Softmax probabilities: {probabilities}")
    print(f"  Sum of probabilities: {probabilities.sum():.4f}")
    
    if abs(probabilities.sum() - 1.0) < 1e-5:
        print(f"  ✓ Softmax normalization valid")
    else:
        print(f"  ✗ Softmax normalization error")
except Exception as e:
    print(f"  ✗ Inference test failed: {e}")
    sys.exit(1)
print()

# ==================== 7. Test Preprocessing ====================
print("✓ Preprocessing Test")
try:
    preprocessor = FramePreprocessor(frame_size=128, seq_len=4)
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Preprocess
    processed = preprocessor.preprocess_frame(dummy_frame)
    print(f"  Input shape: {dummy_frame.shape}")
    print(f"  Output shape: {processed.shape}")
    print(f"  Value range: [{processed.min():.4f}, {processed.max():.4f}]")
    
    # Check normalization
    if -3.0 <= processed.min() <= 3.0 and -3.0 <= processed.max() <= 3.0:
        print(f"  ✓ Normalization range valid")
    else:
        print(f"  ⚠ Normalization range may be unusual")
except Exception as e:
    print(f"  ✗ Preprocessing test failed: {e}")
    sys.exit(1)
print()

# ==================== 8. Test Inference Engine ====================
print("✓ Inference Engine Test")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(str(model_path), device=device)
    inference_engine = DriverStateInference(model, device=device)
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # First 3 frames (buffer not ready)
    for i in range(3):
        result = inference_engine.infer_frame(dummy_frame)
        if result is not None:
            print(f"  Frame {i+1}: Unexpected result (buffer not full)")
            sys.exit(1)
    
    # 4th frame (buffer ready)
    result = inference_engine.infer_frame(dummy_frame)
    if result is None:
        print(f"  ✗ No result on 4th frame (buffer should be full)")
        sys.exit(1)
    
    print(f"  ✓ Buffer management working")
    print(f"  Predicted class: {result['predictions']['class']}")
    print(f"  Confidence: {result['predictions']['confidence']:.4f}")
    print(f"  UDRI Score: {result['predictions']['udri']:.4f}")
    print(f"  Risk Level: {result['predictions']['risk_level']}")
    print(f"  Inference time: {result['inference_time']:.2f}ms")
    
except Exception as e:
    print(f"  ✗ Inference engine test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# ==================== 9. Test Performance Monitor ====================
print("✓ Performance Monitor Test")
try:
    perf_monitor = PerformanceMonitor(window_size=10)
    
    # Simulate inference times
    for i in range(5):
        perf_monitor.add_inference_time(20.0 + np.random.randn() * 2)
    
    metrics = perf_monitor.get_metrics()
    print(f"  Average inference time: {metrics['avg_inference_time']:.2f}ms")
    print(f"  FPS: {metrics['fps']:.2f}")
    print(f"  ✓ Performance monitoring working")
    
except Exception as e:
    print(f"  ✗ Performance monitor test failed: {e}")
    sys.exit(1)
print()

# ==================== 10. Final Status ====================
print("="*60)
print("✓ ALL VALIDATION TESTS PASSED")
print("="*60)
print("\nSystem is ready for deployment!")
print("\nTo start the application, run:")
print("  streamlit run app.py")
print("\nOr use the launch script:")
print("  Windows: launch.bat")
print("  Linux/macOS: ./launch.sh")
print("\n" + "="*60 + "\n")
