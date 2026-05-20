"""
AGFF-DriveNet Inference Module
Preprocessing, inference, and post-processing utilities for driver monitoring

This module handles:
- Frame preprocessing (resize, normalization)
- Video frame extraction
- Temporal sequence assembly
- Model inference with optimization
- Probability computation (softmax)
- UDRI (Unified Driver Risk Index) calculation
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import time


class PreprocessConfig:
    """Configuration for preprocessing"""
    FRAME_SIZE = 128
    SEQ_LEN = 4
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    
    # Class names and weights for UDRI calculation
    CLASS_NAMES = ['SAFE', 'DISTRACTION', 'DROWSINESS', 'STRESS']
    CLASS_WEIGHTS = np.array([0.0, 0.3, 0.5, 0.8])  # Risk weights


class FramePreprocessor:
    """
    Frame preprocessing pipeline with temporal buffering
    """
    def __init__(self, frame_size=128, seq_len=4):
        self.frame_size = frame_size
        self.seq_len = seq_len
        self.frame_buffer = deque(maxlen=seq_len)
    
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame
        
        Args:
            frame: Input frame (H, W, 3) BGR format from OpenCV
        
        Returns:
            Preprocessed frame tensor (3, 128, 128) normalized
        """
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        frame = cv2.resize(frame, (self.frame_size, self.frame_size), 
                          interpolation=cv2.INTER_LINEAR)
        
        # Convert to float32 and normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        frame = (frame - PreprocessConfig.MEAN) / PreprocessConfig.STD
        
        # Convert to tensor and transpose to (C, H, W)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
        
        return frame_tensor
    
    def add_frame(self, frame):
        """
        Add frame to buffer and return whether buffer is full
        
        Args:
            frame: Input frame (H, W, 3)
        
        Returns:
            is_ready: Bool indicating if buffer has seq_len frames
        """
        processed_frame = self.preprocess_frame(frame)
        self.frame_buffer.append(processed_frame)
        return len(self.frame_buffer) == self.seq_len
    
    def get_sequence(self):
        """
        Get current frame sequence as tensor
        
        Returns:
            Sequence tensor (seq_len, 3, 128, 128)
        """
        if len(self.frame_buffer) < self.seq_len:
            # Pad with zeros if not enough frames
            frames = list(self.frame_buffer)
            while len(frames) < self.seq_len:
                frames.insert(0, torch.zeros_like(frames[0]))
            return torch.stack(frames)
        
        return torch.stack(list(self.frame_buffer))
    
    def reset(self):
        """Reset frame buffer"""
        self.frame_buffer.clear()


class DriverStateInference:
    """
    Inference engine for driver state classification
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.preprocessor = FramePreprocessor(
            frame_size=PreprocessConfig.FRAME_SIZE,
            seq_len=PreprocessConfig.SEQ_LEN
        )
    
    def infer_frame(self, frame):
        """
        Run inference on a single frame with temporal context
        
        Args:
            frame: Input frame (H, W, 3)
        
        Returns:
            result: Dict with predictions or None if not ready
        """
        # Add frame to buffer
        is_ready = self.preprocessor.add_frame(frame)
        
        if not is_ready:
            return None
        
        # Get sequence and prepare batch
        sequence = self.preprocessor.get_sequence()
        sequence = sequence.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            logits = self.model(sequence)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Compute predictions
        predictions = self._compute_predictions(probabilities)
        
        result = {
            'probabilities': probabilities,
            'predictions': predictions,
            'inference_time': inference_time,
            'class_names': PreprocessConfig.CLASS_NAMES
        }
        
        return result
    
    def _compute_predictions(self, probabilities):
        """
        Compute detailed predictions from probabilities
        
        Args:
            probabilities: Softmax probabilities (4,)
        
        Returns:
            predictions: Dict with detailed results
        """
        class_names = PreprocessConfig.CLASS_NAMES
        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]
        
        # Compute UDRI (Unified Driver Risk Index)
        udri = np.sum(probabilities * PreprocessConfig.CLASS_WEIGHTS)
        
        # Classify risk level
        if udri < 0.2:
            risk_level = "LOW RISK"
            risk_color = "#00D97E"
        elif udri < 0.4:
            risk_level = "MEDIUM RISK"
            risk_color = "#FFA500"
        elif udri < 0.6:
            risk_level = "HIGH RISK"
            risk_color = "#FF6B35"
        else:
            risk_level = "CRITICAL RISK"
            risk_color = "#D32F2F"
        
        predictions = {
            'class': class_names[predicted_class],
            'confidence': float(confidence),
            'udri': float(udri),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'probabilities': {
                class_names[i]: float(probabilities[i]) 
                for i in range(len(class_names))
            }
        }
        
        return predictions
    
    def reset(self):
        """Reset preprocessor buffer"""
        self.preprocessor.reset()


class VideoProcessor:
    """
    Video processing utilities for frame extraction and processing
    """
    @staticmethod
    def extract_frames(video_path, max_frames=None, target_fps=5):
        """
        Extract frames from video file
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract (None for all)
            target_fps: Target frames per second
        
        Returns:
            Generator yielding frames (H, W, 3)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = int(fps / target_fps) if fps > 0 else 1
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip frames to achieve target FPS
            if frame_count % frame_skip == 0:
                yield frame
                extracted_count += 1
                
                if max_frames and extracted_count >= max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
    
    @staticmethod
    def get_video_properties(video_path):
        """
        Get video properties
        
        Args:
            video_path: Path to video file
        
        Returns:
            properties: Dict with video properties
        """
        cap = cv2.VideoCapture(video_path)
        
        properties = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        
        cap.release()
        return properties


class PerformanceMonitor:
    """
    Monitor inference performance and system metrics
    """
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.inference_times = deque(maxlen=window_size)
        self.fps_values = deque(maxlen=window_size)
    
    def add_inference_time(self, inference_time):
        """Add inference time in milliseconds"""
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 0:
            avg_time = np.mean(list(self.inference_times))
            fps = 1000.0 / (avg_time + 1e-6)
            self.fps_values.append(fps)
    
    def get_metrics(self):
        """Get current performance metrics"""
        if not self.inference_times:
            return {
                'avg_inference_time': 0,
                'fps': 0,
                'max_inference_time': 0,
                'min_inference_time': 0
            }
        
        times = list(self.inference_times)
        return {
            'avg_inference_time': np.mean(times),
            'fps': np.mean(list(self.fps_values)) if self.fps_values else 0,
            'max_inference_time': np.max(times),
            'min_inference_time': np.min(times)
        }
