"""
Video source implementations for camera and file inputs.

Provides concrete implementations of BaseVideoSource for different input types.
"""

import cv2
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
from loguru import logger

from ..core.base_classes import BaseVideoSource


class CameraSource(BaseVideoSource):
    """Video source for live camera input."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize camera source.
        
        Args:
            config: Camera configuration dictionary
        """
        super().__init__(config)
        self.device_id = config.get('device_id', 0)
        self.target_width = config.get('width', 640)
        self.target_height = config.get('height', 480)
        self.target_fps = config.get('fps', 30)
        self.cap = None
    
    def open(self) -> bool:
        """
        Open the camera.
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.device_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Get actual properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.is_opened = True
            logger.info(f"Camera opened: {self.width}x{self.height} @ {self.fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Error opening camera: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
            return ret, frame
        except Exception as e:
            logger.error(f"Error reading camera frame: {e}")
            return False, None
    
    def release(self) -> None:
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            logger.info("Camera released")
    
    @property
    def total_frames(self) -> int:
        """Camera has unlimited frames."""
        return -1  # Infinite for live camera


class FileSource(BaseVideoSource):
    """Video source for file input."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize file source.
        
        Args:
            config: File configuration dictionary
        """
        super().__init__(config)
        self.file_path = Path(config.get('path', ''))
        self.start_frame = config.get('start_frame', 0)
        self.end_frame = config.get('end_frame', -1)
        self.cap = None
        self.current_frame = 0
        self._total_frames = 0
    
    def open(self) -> bool:
        """
        Open the video file.
        
        Returns:
            True if file opened successfully, False otherwise
        """
        if not self.file_path.exists():
            logger.error(f"Video file not found: {self.file_path}")
            return False
        
        try:
            self.cap = cv2.VideoCapture(str(self.file_path))
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open video file: {self.file_path}")
                return False
            
            # Get video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Set start frame if specified
            if self.start_frame > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                self.current_frame = self.start_frame
            
            # Validate end frame
            if self.end_frame == -1:
                self.end_frame = self._total_frames
            else:
                self.end_frame = min(self.end_frame, self._total_frames)
            
            self.is_opened = True
            logger.info(f"Video file opened: {self.file_path}")
            logger.info(f"Properties: {self.width}x{self.height} @ {self.fps} FPS, "
                       f"Frames: {self.start_frame}-{self.end_frame} of {self._total_frames}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening video file: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from file.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        # Check if we've reached the end frame
        if self.current_frame >= self.end_frame:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 1
                self.frame_count += 1
            return ret, frame
        except Exception as e:
            logger.error(f"Error reading video frame: {e}")
            return False, None
    
    def release(self) -> None:
        """Release the video file."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            logger.info(f"Video file released: {self.file_path}")
    
    @property
    def total_frames(self) -> int:
        """Get total number of frames in the video."""
        return self.end_frame - self.start_frame
    
    def seek_to_frame(self, frame_number: int) -> bool:
        """
        Seek to a specific frame.
        
        Args:
            frame_number: Frame number to seek to
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_opened or self.cap is None:
            return False
        
        try:
            actual_frame = max(self.start_frame, min(frame_number, self.end_frame - 1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame)
            self.current_frame = actual_frame
            return True
        except Exception as e:
            logger.error(f"Error seeking to frame {frame_number}: {e}")
            return False


def create_video_source(config: Dict[str, Any]) -> BaseVideoSource:
    """
    Factory function to create appropriate video source.
    
    Args:
        config: Video configuration dictionary
        
    Returns:
        Appropriate video source instance
        
    Raises:
        ValueError: If source type is not supported
    """
    source_type = config.get('source', 'camera')
    
    if source_type == 'camera':
        return CameraSource(config.get('camera', {}))
    elif source_type == 'file':
        return FileSource(config.get('file', {}))
    else:
        raise ValueError(f"Unsupported video source type: {source_type}") 