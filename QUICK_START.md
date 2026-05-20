# AGFF-DriveNet Quick Start Guide

## 🚀 Quick Start (30 seconds)

### Option 1: Windows (Easiest)
1. Double-click `launch.bat`
2. Wait for dependencies to install (first run only)
3. Browser opens automatically at `http://localhost:8501`

### Option 2: PowerShell
```powershell
cd 'c:\Users\Mukesh Patel\Desktop\AGFF-DRIVENET'
streamlit run app.py
```

### Option 3: Command Prompt
```cmd
cd c:\Users\Mukesh Patel\Desktop\AGFF-DRIVENET
python -m streamlit run app.py
```

### Option 4: Linux/macOS
```bash
cd /path/to/AGFF-DRIVENET
chmod +x launch.sh
./launch.sh
```

## ✅ Verification Checklist

- [ ] Python 3.8+ installed: `python --version`
- [ ] All dependencies installed: `pip list | grep streamlit`
- [ ] Model file exists: `agff_hybrid_model.pth` (43.1 MB)
- [ ] All Python files present:
  - [ ] app.py
  - [ ] model.py
  - [ ] infer.py
- [ ] No syntax errors: `python -m py_compile app.py model.py infer.py`

## 🎯 First Run

1. **Allow permissions**: If prompted, allow webcam access
2. **Wait for model loading**: First startup loads model into memory (~10-30 seconds)
3. **Check GPU status**: Sidebar shows if GPU is active
4. **Select input method**:
   - **Webcam Tab**: Real-time monitoring
   - **Upload Tab**: Video file analysis

## 📊 Expected Performance

### GPU (NVIDIA RTX 3090+)
- Load time: 5-15 seconds
- Webcam FPS: 30+ FPS
- Inference time: 15-20ms

### GPU (NVIDIA GTX 1660)
- Load time: 10-20 seconds
- Webcam FPS: 20-25 FPS
- Inference time: 25-35ms

### CPU (Intel i7+)
- Load time: 20-40 seconds
- Webcam FPS: 10-15 FPS
- Inference time: 70-100ms

## 🔧 Troubleshooting

### Application won't start
```powershell
# Verify Python installation
python --version

# Reinstall dependencies
python -m pip install --upgrade -r requirements.txt

# Run with verbose output
streamlit run app.py --logger.level=debug
```

### GPU not detected
```python
python -c "import torch; print(torch.cuda.is_available())"
```

### Webcam not working
```python
import cv2
cap = cv2.VideoCapture(0)
print(f"Webcam accessible: {cap.isOpened()}")
cap.release()
```

### Low FPS on GPU
- Check other GPU applications using: `nvidia-smi`
- Restart application
- Monitor memory with: `nvidia-smi -l 1`

## 🌐 Deployment Options

### Local Deployment
```bash
streamlit run app.py
```

### LAN Access (Share within network)
```bash
streamlit run app.py --server.address=0.0.0.0
```
Then access via: `http://YOUR_IP:8501`

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Cloud Deployment (Streamlit Cloud)
1. Push to GitHub
2. Visit https://share.streamlit.io
3. Deploy from GitHub repository

## 📱 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10+ / Ubuntu 18.04+ / macOS 10.14+ | Latest LTS |
| **Python** | 3.8 | 3.10+ |
| **RAM** | 2 GB | 8 GB |
| **Storage** | 2 GB | 10 GB |
| **GPU** | None (CPU mode) | NVIDIA RTX 2060+ |
| **Webcam** | Optional | USB 2.0+ |

## 🎓 Learning Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **PyTorch Guide**: https://pytorch.org/tutorials
- **Vision Transformers**: https://huggingface.co/docs/transformers
- **OpenCV Tutorials**: https://opencv.org/university

## 📞 Support

1. **Check README.md** for detailed documentation
2. **Review error messages** in terminal
3. **Verify all files** are in correct location
4. **Test dependencies** individually:
   ```python
   python -c "import torch, cv2, streamlit, plotly"
   ```

---

**Ready to monitor drivers safely with AI!** 🚗✨
