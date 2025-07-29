"""
Main pipeline for the Detect-Track system.

Orchestrates the complete pipeline: Frame Extraction → Detection → Conversion → Tracking → Logging
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional
from loguru import logger

from ..config.config_manager import ConfigManager
from ..core.base_classes import BaseVideoSource, BaseDetector, BaseTracker, FrameData, Detection
from ..video.video_sources import create_video_source
from ..detection.yolo_detectors import create_yolo_detector
from ..tracking.trackers import create_tracker
from ..utils.logging_system import DetectionTrackLogger


class DetectionTrackingPipeline:
    """
    Main pipeline for object detection and tracking.
    
    Implements the complete workflow:
    1. Frame Extraction: Read video frames
    2. Detection: Run YOLO on each frame
    3. Conversion: Transform detections to tracker input
    4. Tracking: Process frames through tracker
    5. Logging: Record per-frame outputs
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Initialize components
        self.video_source: Optional[BaseVideoSource] = None
        self.detector: Optional[BaseDetector] = None
        self.tracker: Optional[BaseTracker] = None
        self.logger: Optional[DetectionTrackLogger] = None
        
        # Pipeline state
        self.is_running = False
        self.frame_count = 0
        self.start_time = 0.0
        
        # Performance settings
        self.skip_frames = self.config_manager.get('performance.skip_frames', 0)
        self.resize_input = self.config_manager.get('performance.resize_input', True)
        self.target_size = tuple(self.config_manager.get('performance.target_size', [640, 640]))
        
        # Display settings
        self.display_config = self.config_manager.display_config
        
        logger.info("Pipeline initialized")
    
    def setup(self) -> bool:
        """
        Set up all pipeline components.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # 1. Setup video source
            logger.info("Setting up video source...")
            self.video_source = create_video_source(self.config_manager.video_config)
            if not self.video_source.open():
                logger.error("Failed to open video source")
                return False
            
            # 2. Setup detector
            logger.info("Setting up detector...")
            detection_config = self.config_manager.detection_config.copy()
            detection_config.update(self.config_manager.performance_config)
            self.detector = create_yolo_detector(detection_config)
            self.detector.load_model()
            
            # 3. Setup tracker
            logger.info("Setting up tracker...")
            tracking_config = self.config_manager.tracking_config.copy()
            self.tracker = create_tracker(tracking_config)
            
            # 4. Setup logger
            logger.info("Setting up logger...")
            self.logger = DetectionTrackLogger(self.config_manager.logging_config)
            
            # Setup video writer if needed
            if self.logger.save_video:
                self.logger.setup_video_writer(
                    self.video_source.fps,
                    (self.video_source.width, self.video_source.height)
                )
            
            logger.info("Pipeline setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline setup failed: {e}")
            return False
    
    def run(self) -> None:
        """Run the complete detection and tracking pipeline."""
        if not self.setup():
            logger.error("Pipeline setup failed, aborting")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        logger.info("Starting detection and tracking pipeline...")
        
        try:
            with self.logger:
                while self.is_running:
                    # Step 1: Frame Extraction
                    success, frame = self._extract_frame()
                    if not success:
                        break
                    
                    # Step 2: Detection
                    detections = self._run_detection(frame)
                    
                    # Step 3: Conversion (detections are already in correct format)
                    converted_detections = self._convert_detections(detections)
                    
                    # Step 4: Tracking
                    tracks = self._run_tracking(converted_detections, frame)
                    
                    # Step 5: Logging
                    self._log_results(frame, converted_detections, tracks)
                    
                    # Display frame if enabled
                    if self.display_config.get('show_video', True):
                        if not self._display_frame(frame, converted_detections, tracks):
                            break  # User pressed 'q' to quit
            
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user")
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
        finally:
            self._cleanup()
    
    def _extract_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Step 1: Frame Extraction
        Read the next frame from video source.
        
        Returns:
            Tuple of (success, frame)
        """
        # Check if we should skip frames
        if self.skip_frames > 0 and self.frame_count % (self.skip_frames + 1) != 0:
            success, frame = self.video_source.read_frame()
            if success:
                self.frame_count += 1
            return success, frame
        
        success, frame = self.video_source.read_frame()
        if not success:
            logger.info("End of video stream reached")
            return False, None
        
        self.frame_count += 1
        
        # Resize frame if configured AND source allows resizing (preserve original resolution for video files)
        if (self.resize_input and frame is not None and 
            hasattr(self.video_source, 'allow_resize') and 
            self.video_source.allow_resize):
            frame = cv2.resize(frame, self.target_size)
            logger.debug(f"Resized frame to {self.target_size}")
        elif frame is not None and hasattr(self.video_source, 'allow_resize') and not self.video_source.allow_resize:
            logger.debug(f"Preserving original video resolution: {frame.shape[1]}x{frame.shape[0]}")
        
        return True, frame
    
    def _run_detection(self, frame: np.ndarray) -> list[Detection]:
        """
        Step 2: Detection
        Run YOLO detection on the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections
        """
        detections = self.detector.detect(frame)
        
        # Set frame ID for all detections
        for detection in detections:
            detection.frame_id = self.frame_count
        
        return detections
    
    def _convert_detections(self, detections: list[Detection]) -> list[Detection]:
        """
        Step 3: Conversion
        Transform detections to tracker input format.
        (In our case, detections are already in the correct format)
        
        Args:
            detections: Raw detections from detector
            
        Returns:
            Converted detections ready for tracker
        """
        # Our Detection objects are already in the correct format
        # This step is here for completeness and future extensibility
        return detections
    
    def _run_tracking(self, detections: list[Detection], frame: np.ndarray) -> list:
        """
        Step 4: Tracking
        Process detections through the tracker.
        
        Args:
            detections: Converted detections
            frame: Current frame
            
        Returns:
            List of tracks
        """
        tracks = self.tracker.update(detections, frame)
        return tracks
    
    def _log_results(self, frame: np.ndarray, detections: list[Detection], 
                    tracks: list) -> None:
        """
        Step 5: Logging
        Record per-frame outputs.
        
        Args:
            frame: Current frame
            detections: Frame detections
            tracks: Frame tracks
        """
        # Calculate processing time
        current_time = time.time()
        frame_duration = current_time - getattr(self, '_last_frame_time', current_time)
        processing_time = min(frame_duration, 0.1)  # Cap at 100ms to avoid unrealistic values
        self._last_frame_time = current_time
        
        # Create annotated frame for logging
        annotated_frame = self.logger.draw_annotations(
            frame, detections, tracks,
            show_detections=self.display_config.get('show_detections', True),
            show_tracks=self.display_config.get('show_tracks', True),
            show_fps=self.display_config.get('show_fps', True),
            fps=1.0 / processing_time if processing_time > 0 else 0.0
        )
        
        # Create frame data object
        frame_data = FrameData(
            frame_id=self.frame_count,
            timestamp=current_time,
            image=annotated_frame,
            detections=detections,
            tracks=tracks,
            processing_time=processing_time
        )
        
        # Log the frame data
        self.logger.log_frame_data(frame_data)
    
    def _display_frame(self, frame: np.ndarray, detections: list[Detection], 
                      tracks: list) -> bool:
        """
        Display the current frame with annotations.
        
        Args:
            frame: Current frame
            detections: Frame detections
            tracks: Frame tracks
            
        Returns:
            True to continue, False to stop
        """
        # Calculate REAL-TIME FPS using recent frame times
        current_time = time.time()
        
        # Initialize frame time tracking
        if not hasattr(self, '_recent_frame_times'):
            self._recent_frame_times = []
        
        self._recent_frame_times.append(current_time)
        
        # Keep only last 30 frame times for rolling average
        if len(self._recent_frame_times) > 30:
            self._recent_frame_times.pop(0)
        
        # Calculate FPS from recent frames
        if len(self._recent_frame_times) >= 2:
            time_span = self._recent_frame_times[-1] - self._recent_frame_times[0]
            fps = (len(self._recent_frame_times) - 1) / time_span if time_span > 0 else 0.0
        else:
            fps = 0.0
        
        # Create display frame
        display_frame = self.logger.draw_annotations(
            frame, detections, tracks,
            show_detections=self.display_config.get('show_detections', True),
            show_tracks=self.display_config.get('show_tracks', True),
            show_fps=self.display_config.get('show_fps', True),
            fps=fps
        )
        
        # Display frame
        cv2.imshow('Detect-Track', display_frame)
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Quit requested by user")
            return False
        
        return True
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        self.is_running = False
        
        if self.video_source:
            self.video_source.release()
        
        cv2.destroyAllWindows()
        
        logger.info("Pipeline cleanup completed")
    
    def stop(self) -> None:
        """Stop the pipeline."""
        self.is_running = False
        logger.info("Pipeline stop requested")
    
    def switch_detector(self, model_name: str) -> bool:
        """
        Switch to a different YOLO model.
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Switching detector to {model_name}")
            
            # Update configuration
            self.config_manager.set('detection.model', model_name)
            
            # Create new detector
            detection_config = self.config_manager.detection_config.copy()
            detection_config.update(self.config_manager.performance_config)
            new_detector = create_yolo_detector(detection_config)
            new_detector.load_model()
            
            # Replace detector
            self.detector = new_detector
            
            logger.info(f"Successfully switched to {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch detector: {e}")
            return False
    
    def switch_tracker(self, tracker_name: str) -> bool:
        """
        Switch to a different tracking algorithm.
        
        Args:
            tracker_name: Name of the tracker to switch to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Switching tracker to {tracker_name}")
            
            # Update configuration
            self.config_manager.set('tracking.algorithm', tracker_name)
            
            # Create new tracker
            tracking_config = self.config_manager.tracking_config.copy()
            new_tracker = create_tracker(tracking_config)
            
            # Replace tracker (this will reset tracking state)
            self.tracker = new_tracker
            
            logger.info(f"Successfully switched to {tracker_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch tracker: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.
        
        Returns:
            Dictionary containing status information
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time > 0 else 0
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0.0
        
        status = {
            'is_running': self.is_running,
            'frame_count': self.frame_count,
            'elapsed_time': elapsed_time,
            'current_fps': fps,
            'detector_model': self.config_manager.get('detection.model'),
            'tracker_algorithm': self.config_manager.get('tracking.algorithm'),
            'video_source': self.config_manager.get('video.source')
        }
        
        return status 