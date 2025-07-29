"""
Object tracking implementations.

Provides implementations for NvDCF, DeepSORT, and BOTSort tracking algorithms.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
from loguru import logger

try:
    from deep_sort_realtime import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logger.warning("DeepSORT not available. Install with: pip install deep-sort-realtime")

from ..core.base_classes import BaseTracker, Detection, Track


@dataclass
class TrackerDetection:
    """Detection format for trackers."""
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int


class NvDCFTracker(BaseTracker):
    """
    NvDCF (Normalized Cross-Correlation with Discriminative Correlation Filters) tracker.
    Uses OpenCV's implementation with fallbacks for different versions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NvDCF tracker."""
        super().__init__(config)
        self.trackers = {}  # track_id -> cv2.Tracker
        self.tracker_states = {}  # track_id -> state info
        self.tracker_type = self._get_available_tracker()
        
        if not self.tracker_type:
            logger.error("No OpenCV trackers available. Please install opencv-contrib-python")
            raise RuntimeError("No OpenCV trackers available")
        
        logger.info(f"Using OpenCV tracker: {self.tracker_type}")
        
    def _get_available_tracker(self) -> Optional[str]:
        """Get the best available OpenCV tracker."""
        # Try different tracker types in order of preference
        tracker_options = [
            ('TrackerCSRT_create', 'CSRT'),
            ('TrackerKCF_create', 'KCF'),
            ('TrackerMOSSE_create', 'MOSSE'),
            ('legacy.TrackerCSRT_create', 'CSRT_legacy'),
            ('legacy.TrackerKCF_create', 'KCF_legacy'),
            ('legacy.TrackerMOSSE_create', 'MOSSE_legacy')
        ]
        
        for attr_name, tracker_name in tracker_options:
            try:
                # Navigate nested attributes (e.g., cv2.legacy.TrackerCSRT_create)
                obj = cv2
                for part in attr_name.split('.'):
                    obj = getattr(obj, part)
                
                # Test if we can create a tracker
                test_tracker = obj()
                return tracker_name
            except (AttributeError, Exception):
                continue
        
        return None
    
    def _create_opencv_tracker(self):
        """Create an OpenCV tracker instance."""
        try:
            if self.tracker_type == 'CSRT':
                return cv2.TrackerCSRT_create()
            elif self.tracker_type == 'KCF':
                return cv2.TrackerKCF_create()
            elif self.tracker_type == 'MOSSE':
                return cv2.TrackerMOSSE_create()
            elif self.tracker_type == 'CSRT_legacy':
                return cv2.legacy.TrackerCSRT_create()
            elif self.tracker_type == 'KCF_legacy':
                return cv2.legacy.TrackerKCF_create()
            elif self.tracker_type == 'MOSSE_legacy':
                return cv2.legacy.TrackerMOSSE_create()
            else:
                raise RuntimeError(f"Unknown tracker type: {self.tracker_type}")
        except Exception as e:
            logger.error(f"Failed to create {self.tracker_type} tracker: {e}")
            return None
        
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """Update tracker with new detections."""
        self.frame_count += 1
        
        # Convert detections to simple format
        current_detections = []
        for det in detections:
            current_detections.append(TrackerDetection(
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id
            ))
        
        # Update existing trackers
        for track_id in list(self.trackers.keys()):
            if self.trackers[track_id] is None:
                continue
                
            success, bbox = self.trackers[track_id].update(frame)
            
            if success:
                # Update track state
                self.tracker_states[track_id]['frames_since_update'] = 0
                self.tracker_states[track_id]['bbox'] = bbox
                
                # Find the track object
                track = next((t for t in self.tracks if t.track_id == track_id), None)
                if track:
                    track.frames_since_update = 0
                    # Create detection from tracker result
                    x, y, w, h = bbox
                    det = Detection(
                        bbox=(x, y, x + w, y + h),
                        confidence=0.5,  # Tracker confidence
                        class_id=self.tracker_states[track_id]['class_id'],
                        class_name=self.tracker_states[track_id]['class_name'],
                        frame_id=self.frame_count
                    )
                    track.detections.append(det)
            else:
                # Tracker failed, increment lost frames
                self.tracker_states[track_id]['frames_since_update'] += 1
                track = next((t for t in self.tracks if t.track_id == track_id), None)
                if track:
                    track.frames_since_update += 1
        
        # Remove lost tracks
        tracks_to_remove = []
        for track_id, state in self.tracker_states.items():
            if state['frames_since_update'] > self.max_lost_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if track_id in self.trackers:
                del self.trackers[track_id]
            if track_id in self.tracker_states:
                del self.tracker_states[track_id]
            track = next((t for t in self.tracks if t.track_id == track_id), None)
            if track:
                track.is_deleted = True
        
        # Associate new detections with existing tracks or create new ones
        unmatched_detections = self._associate_detections(current_detections)
        
        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            self._create_new_track(det, frame)
        
        # Update track confirmation status
        for track in self.tracks:
            if not track.is_deleted and track.duration >= self.min_track_length:
                track.is_confirmed = True
        
        return self.get_active_tracks()
    
    def _associate_detections(self, detections: List[TrackerDetection]) -> List[TrackerDetection]:
        """Associate detections with existing tracks using IoU."""
        unmatched_detections = []
        
        for det in detections:
            best_match = None
            best_iou = 0.3  # Minimum IoU threshold
            
            for track_id, state in self.tracker_states.items():
                if state['frames_since_update'] == 0:  # Only consider recently updated tracks
                    track_bbox = state['bbox']
                    # Convert to x1, y1, x2, y2 format
                    x, y, w, h = track_bbox
                    track_bbox_xyxy = (x, y, x + w, y + h)
                    
                    iou = self._calculate_iou(det.bbox, track_bbox_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = track_id
            
            if best_match is None:
                unmatched_detections.append(det)
        
        return unmatched_detections
    
    def _create_new_track(self, detection: TrackerDetection, frame: np.ndarray) -> None:
        """Create a new track for a detection."""
        # Create OpenCV tracker
        tracker = self._create_opencv_tracker()
        
        if tracker is None:
            logger.warning("Failed to create tracker, skipping new track")
            return
        
        # Validate and prepare detection bbox
        x1, y1, x2, y2 = detection.bbox
        
        # Ensure coordinates are valid
        if x1 >= x2 or y1 >= y2:
            logger.warning(f"Invalid bounding box coordinates: {detection.bbox}")
            return
        
        # Ensure coordinates are within frame bounds
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(x1 + 1, min(x2, frame_w))
        y2 = max(y1 + 1, min(y2, frame_h))
        
        # Convert to x, y, w, h format with integer coordinates (OpenCV requirement)
        width = int(x2 - x1)
        height = int(y2 - y1)
        
        # Ensure minimum size for tracker
        if width < 10 or height < 10:
            logger.warning(f"Bounding box too small for tracking: {width}x{height}")
            return
        
        bbox = (int(x1), int(y1), width, height)
        
        # Ensure frame is in the correct format (BGR, uint8)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convert to BGR if needed (OpenCV trackers expect BGR)
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        else:
            logger.warning(f"Unexpected frame format: {frame.shape}")
            return
        
        try:
            success = tracker.init(frame, bbox)
            if success:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.trackers[track_id] = tracker
                self.tracker_states[track_id] = {
                    'bbox': bbox,
                    'frames_since_update': 0,
                    'class_id': detection.class_id,
                    'class_name': f'class_{detection.class_id}'
                }
                
                # Create track object
                det = Detection(
                    bbox=(x1, y1, x2, y2),  # Use validated coordinates
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=f'class_{detection.class_id}',
                    frame_id=self.frame_count
                )
                
                track = Track(
                    track_id=track_id,
                    detections=[det],
                    is_confirmed=False,
                    is_deleted=False,
                    frames_since_update=0
                )
                
                self.tracks.append(track)
                logger.debug(f"Successfully created track {track_id} with bbox {bbox}")
            else:
                logger.warning(f"Tracker initialization failed for bbox {bbox}")
        
        except Exception as e:
            logger.error(f"Exception during tracker initialization: {e}")
            logger.debug(f"Frame shape: {frame.shape}, bbox: {bbox}")
            return
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.trackers.clear()
        self.tracker_states.clear()
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0


class DeepSORTTracker(BaseTracker):
    """DeepSORT tracker implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DeepSORT tracker."""
        super().__init__(config)
        
        if not DEEPSORT_AVAILABLE:
            raise ImportError("DeepSORT not available. Install with: pip install deep-sort-realtime")
        
        # DeepSORT configuration
        self.max_age = config.get('max_age', 30)
        self.n_init = config.get('n_init', 3)
        self.max_iou_distance = config.get('max_iou_distance', 0.7)
        self.max_cosine_distance = config.get('max_cosine_distance', 0.2)
        
        # Initialize DeepSORT
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            max_iou_distance=self.max_iou_distance,
            max_cosine_distance=self.max_cosine_distance,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        logger.info("DeepSORT tracker initialized")
    
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """Update tracker with new detections."""
        self.frame_count += 1
        
        # Convert detections to DeepSORT format
        det_list = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # DeepSORT expects [x1, y1, x2, y2, confidence]
            det_list.append([x1, y1, x2, y2, det.confidence])
        
        # Update tracker
        tracks = self.tracker.update_tracks(det_list, frame=frame)
        
        # Convert DeepSORT tracks to our format
        self.tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            # Get track information
            track_id = track.track_id
            bbox = track.to_ltwh()  # [left, top, width, height]
            x, y, w, h = bbox
            xyxy_bbox = (x, y, x + w, y + h)
            
            # Create detection for this track
            det = Detection(
                bbox=xyxy_bbox,
                confidence=0.8,  # Tracker confidence
                class_id=0,  # DeepSORT doesn't preserve class info by default
                class_name='object',
                frame_id=self.frame_count
            )
            
            # Create track object
            track_obj = Track(
                track_id=track_id,
                detections=[det],
                is_confirmed=track.is_confirmed(),
                is_deleted=track.is_deleted(),
                frames_since_update=track.time_since_update
            )
            
            self.tracks.append(track_obj)
        
        return self.get_active_tracks()
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            max_iou_distance=self.max_iou_distance,
            max_cosine_distance=self.max_cosine_distance,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True
        )
        self.tracks.clear()
        self.frame_count = 0


class BOTSortTracker(BaseTracker):
    """
    BOTSort tracker implementation.
    Simplified version focusing on basic tracking with motion prediction.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize BOTSort tracker."""
        super().__init__(config)
        
        self.track_high_thresh = config.get('track_high_thresh', 0.6)
        self.track_low_thresh = config.get('track_low_thresh', 0.1)
        self.new_track_thresh = config.get('new_track_thresh', 0.7)
        self.match_thresh = config.get('match_thresh', 0.8)
        
        # Simple motion model parameters
        self.motion_lambda = 0.98
        
        logger.info("BOTSort tracker initialized")
    
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """Update tracker with new detections."""
        self.frame_count += 1
        
        # Predict motion for existing tracks
        self._predict_motion()
        
        # Associate detections with tracks
        matched_tracks, unmatched_dets, unmatched_tracks = self._associate(detections)
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            track = self.tracks[track_idx]
            detection = detections[det_idx]
            detection.frame_id = self.frame_count
            
            track.detections.append(detection)
            track.frames_since_update = 0
            track.is_confirmed = True
            
            # Update motion model
            self._update_motion(track, detection)
        
        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].frames_since_update += 1
            if self.tracks[track_idx].frames_since_update > self.max_lost_frames:
                self.tracks[track_idx].is_deleted = True
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            if detection.confidence >= self.new_track_thresh:
                self._create_new_track(detection)
        
        return self.get_active_tracks()
    
    def _predict_motion(self) -> None:
        """Predict motion for all active tracks."""
        for track in self.tracks:
            if track.is_deleted or len(track.detections) < 2:
                continue
            
            # Simple velocity-based prediction
            last_det = track.detections[-1]
            prev_det = track.detections[-2]
            
            # Calculate velocity
            dx = last_det.center[0] - prev_det.center[0]
            dy = last_det.center[1] - prev_det.center[1]
            
            # Predict next position
            pred_x = last_det.center[0] + dx * self.motion_lambda
            pred_y = last_det.center[1] + dy * self.motion_lambda
            
            # Store prediction for association
            if not hasattr(track, 'predicted_center'):
                track.predicted_center = (pred_x, pred_y)
            else:
                track.predicted_center = (pred_x, pred_y)
    
    def _associate(self, detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections with tracks using IoU and motion prediction."""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        # Calculate cost matrix
        active_tracks = [i for i, t in enumerate(self.tracks) if not t.is_deleted]
        cost_matrix = np.zeros((len(active_tracks), len(detections)))
        
        for i, track_idx in enumerate(active_tracks):
            track = self.tracks[track_idx]
            if len(track.detections) == 0:
                continue
                
            last_detection = track.detections[-1]
            
            for j, detection in enumerate(detections):
                # Calculate IoU
                iou = self._calculate_iou(last_detection.bbox, detection.bbox)
                
                # Calculate distance penalty if motion prediction available
                distance_cost = 0.0
                if hasattr(track, 'predicted_center'):
                    pred_center = track.predicted_center
                    det_center = detection.center
                    distance = np.sqrt((pred_center[0] - det_center[0])**2 + 
                                     (pred_center[1] - det_center[1])**2)
                    distance_cost = min(distance / 100.0, 1.0)  # Normalize distance
                
                # Combined cost (lower is better)
                cost_matrix[i, j] = 1.0 - iou + distance_cost
        
        # Simple greedy assignment
        matched_tracks = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(active_tracks)))
        
        while len(unmatched_tracks) > 0 and len(unmatched_dets) > 0:
            # Find minimum cost
            min_cost = float('inf')
            min_track_idx = -1
            min_det_idx = -1
            
            for i in unmatched_tracks:
                for j in unmatched_dets:
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_track_idx = i
                        min_det_idx = j
            
            # Check if cost is acceptable
            if min_cost < (1.0 - self.match_thresh):
                matched_tracks.append((active_tracks[min_track_idx], min_det_idx))
                unmatched_tracks.remove(min_track_idx)
                unmatched_dets.remove(min_det_idx)
            else:
                break
        
        # Convert unmatched track indices back to original indices
        unmatched_tracks = [active_tracks[i] for i in unmatched_tracks]
        
        return matched_tracks, unmatched_dets, unmatched_tracks
    
    def _create_new_track(self, detection: Detection) -> None:
        """Create a new track."""
        detection.frame_id = self.frame_count
        
        track = Track(
            track_id=self.next_track_id,
            detections=[detection],
            is_confirmed=False,
            is_deleted=False,
            frames_since_update=0
        )
        
        self.tracks.append(track)
        self.next_track_id += 1
    
    def _update_motion(self, track: Track, detection: Detection) -> None:
        """Update motion model for a track."""
        # Simple implementation - motion prediction is done in _predict_motion
        pass
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                      bbox2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0


def create_tracker(config: Dict[str, Any]) -> BaseTracker:
    """
    Factory function to create appropriate tracker.
    
    Args:
        config: Tracking configuration dictionary
        
    Returns:
        Appropriate tracker instance
        
    Raises:
        ValueError: If tracker type is not supported
    """
    tracker_type = config.get('algorithm', 'botsort')  # Changed default to botsort as fallback
    
    if tracker_type == 'nvdcf':
        return NvDCFTracker(config)
    elif tracker_type == 'deepsort':
        return DeepSORTTracker(config)
    elif tracker_type == 'botsort':
        return BOTSortTracker(config)
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}") 