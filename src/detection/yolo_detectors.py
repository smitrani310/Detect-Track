"""
YOLO detector implementations for various YOLO versions.

Supports YOLOv5, YOLOv7, and YOLOv8 models with unified interface.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    import ultralytics
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not available, YOLOv8 will not work")

try:
    import yolov5
    YOLOV5_AVAILABLE = True
except ImportError:
    YOLOV5_AVAILABLE = False
    logger.warning("YOLOv5 package not available")

from ..core.base_classes import BaseDetector, Detection


class YOLOv5Detector(BaseDetector):
    """YOLOv5 detector implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize YOLOv5 detector."""
        super().__init__(config)
        self.model_name = config.get('model', 'yolov5n')
        self.weights_path = config.get('weights', f'{self.model_name}.pt')
        self.class_names = []
        
        if not YOLOV5_AVAILABLE:
            raise ImportError("YOLOv5 package not available. Install with: pip install yolov5")
    
    def load_model(self) -> None:
        """Load YOLOv5 model."""
        try:
            # Try to load from local weights first, then from torch hub
            if Path(self.weights_path).exists():
                self.model = yolov5.load(self.weights_path, device=self.device)
            else:
                # Load from torch hub
                self.model = yolov5.load(self.model_name, device=self.device)
            
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            self.class_names = self.model.names
            
            logger.info(f"YOLOv5 model loaded: {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv5 model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Perform detection on frame."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Run inference
            results = self.model(frame)
            detections = []
            
            # Parse results
            for result in results.xyxy[0]:  # xyxy format
                x1, y1, x2, y2, conf, class_id = result.tolist()
                
                if conf >= self.confidence_threshold:
                    detection = Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=int(class_id),
                        class_name=self.class_names[int(class_id)],
                        frame_id=0  # Will be set by caller
                    )
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLOv5 detection failed: {e}")
            return []
    
    def get_class_names(self) -> List[str]:
        """Get class names."""
        return list(self.class_names) if self.class_names else []


class YOLOv7Detector(BaseDetector):
    """YOLOv7 detector implementation using torch.hub."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize YOLOv7 detector."""
        super().__init__(config)
        self.model_name = 'yolov7'
        self.weights_path = config.get('weights', 'yolov7.pt')
        self.class_names = []
    
    def load_model(self) -> None:
        """Load YOLOv7 model."""
        try:
            # Load YOLOv7 from torch hub
            self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', 
                                      path_or_model=self.weights_path,
                                      force_reload=False, trust_repo=True)
            
            self.model.conf = self.confidence_threshold
            self.model.iou = self.iou_threshold
            
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
            else:
                # Default COCO class names
                self.class_names = self._get_coco_names()
            
            # Set device
            self.model.to(self.device)
            
            logger.info(f"YOLOv7 model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv7 model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Perform detection on frame."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Run inference
            results = self.model(frame)
            detections = []
            
            # Parse results - YOLOv7 returns pandas dataframe
            if hasattr(results, 'pandas'):
                df = results.pandas().xyxy[0]
                
                for _, row in df.iterrows():
                    if row['confidence'] >= self.confidence_threshold:
                        detection = Detection(
                            bbox=(row['xmin'], row['ymin'], row['xmax'], row['ymax']),
                            confidence=row['confidence'],
                            class_id=int(row['class']),
                            class_name=row['name'],
                            frame_id=0  # Will be set by caller
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLOv7 detection failed: {e}")
            return []
    
    def get_class_names(self) -> List[str]:
        """Get class names."""
        return list(self.class_names) if self.class_names else []
    
    def _get_coco_names(self) -> List[str]:
        """Get default COCO class names."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]


class YOLOv8Detector(BaseDetector):
    """YOLOv8 detector implementation using ultralytics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize YOLOv8 detector."""
        super().__init__(config)
        self.model_name = config.get('model', 'yolov8n')
        self.weights_path = config.get('weights', f'{self.model_name}.pt')
        self.class_names = []
        
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not available. Install with: pip install ultralytics")
    
    def load_model(self) -> None:
        """Load YOLOv8 model."""
        try:
            # Load model
            if Path(self.weights_path).exists():
                self.model = YOLO(self.weights_path)
            else:
                self.model = YOLO(self.model_name + '.pt')
            
            # Get class names
            if hasattr(self.model.model, 'names'):
                self.class_names = self.model.model.names
            else:
                self.class_names = self.model.names
            
            logger.info(f"YOLOv8 model loaded: {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Perform detection on frame."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Run inference
            results = self.model(frame, 
                               conf=self.confidence_threshold, 
                               iou=self.iou_threshold,
                               device=self.device,
                               verbose=False)
            
            detections = []
            
            # Parse results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box data
                        xyxy = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if conf >= self.confidence_threshold:
                            detection = Detection(
                                bbox=(float(xyxy[0]), float(xyxy[1]), 
                                     float(xyxy[2]), float(xyxy[3])),
                                confidence=float(conf),
                                class_id=class_id,
                                class_name=self.class_names[class_id],
                                frame_id=0  # Will be set by caller
                            )
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLOv8 detection failed: {e}")
            return []
    
    def get_class_names(self) -> List[str]:
        """Get class names."""
        return list(self.class_names) if self.class_names else []


def create_yolo_detector(config: Dict[str, Any]) -> BaseDetector:
    """
    Factory function to create appropriate YOLO detector.
    
    Args:
        config: Detection configuration dictionary
        
    Returns:
        Appropriate YOLO detector instance
        
    Raises:
        ValueError: If model type is not supported
    """
    model_name = config.get('model', 'yolov8n')
    
    # Determine device
    if config.get('device', 'auto') == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config['device'] = device
    
    if model_name.startswith('yolov5'):
        return YOLOv5Detector(config)
    elif model_name.startswith('yolov7'):
        return YOLOv7Detector(config)
    elif model_name.startswith('yolov8'):
        return YOLOv8Detector(config)
    else:
        raise ValueError(f"Unsupported YOLO model: {model_name}") 