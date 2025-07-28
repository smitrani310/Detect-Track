"""
Base classes and data structures for the Detect-Track system.

Defines abstract interfaces and common data structures used throughout the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
import numpy as np


@dataclass
class Detection:
    """Represents a single object detection."""
    
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    frame_id: int
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> float:
        """Get the width of the bounding box."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Get the height of the bounding box."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """Get the area of the bounding box."""
        return self.width * self.height


@dataclass
class Track:
    """Represents a tracked object across multiple frames."""
    
    track_id: int
    detections: List[Detection]
    is_confirmed: bool = False
    is_deleted: bool = False
    frames_since_update: int = 0
    
    @property
    def latest_detection(self) -> Optional[Detection]:
        """Get the most recent detection for this track."""
        return self.detections[-1] if self.detections else None
    
    @property
    def duration(self) -> int:
        """Get the number of frames this track has been active."""
        return len(self.detections)
    
    @property
    def first_frame(self) -> int:
        """Get the frame ID where this track first appeared."""
        return self.detections[0].frame_id if self.detections else -1
    
    @property
    def last_frame(self) -> int:
        """Get the frame ID where this track was last seen."""
        return self.detections[-1].frame_id if self.detections else -1


class BaseDetector(ABC):
    """Abstract base class for object detectors."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the detector.
        
        Args:
            config: Configuration dictionary for the detector
        """
        self.config = config
        self.model = None
        self.device = config.get('device', 'cpu')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.iou_threshold = config.get('iou_threshold', 0.45)
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Perform object detection on a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Get list of class names supported by the detector."""
        pass


class BaseTracker(ABC):
    """Abstract base class for object trackers."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tracker.
        
        Args:
            config: Configuration dictionary for the tracker
        """
        self.config = config
        self.max_lost_frames = config.get('max_lost_frames', 30)
        self.min_track_length = config.get('min_track_length', 3)
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.frame_count = 0
    
    @abstractmethod
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections from current frame
            frame: Current frame as numpy array
            
        Returns:
            List of updated tracks
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the tracker state."""
        pass
    
    def get_active_tracks(self) -> List[Track]:
        """Get list of currently active tracks."""
        return [track for track in self.tracks if not track.is_deleted]
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get list of confirmed tracks."""
        return [track for track in self.tracks if track.is_confirmed and not track.is_deleted]


class BaseVideoSource(ABC):
    """Abstract base class for video sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the video source.
        
        Args:
            config: Configuration dictionary for the video source
        """
        self.config = config
        self.is_opened = False
        self.frame_count = 0
        self.fps = 30.0
        self.width = 640
        self.height = 480
    
    @abstractmethod
    def open(self) -> bool:
        """
        Open the video source.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.
        
        Returns:
            Tuple of (success, frame) where success is bool and frame is numpy array or None
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release the video source."""
        pass
    
    @property
    @abstractmethod
    def total_frames(self) -> int:
        """Get total number of frames (if available)."""
        pass


@dataclass
class FrameData:
    """Container for frame data and associated metadata."""
    
    frame_id: int
    timestamp: float
    image: np.ndarray
    detections: List[Detection]
    tracks: List[Track]
    processing_time: float = 0.0
    
    @property
    def fps(self) -> float:
        """Calculate instantaneous FPS based on processing time."""
        return 1.0 / self.processing_time if self.processing_time > 0 else 0.0 