"""
Logging system for the Detect-Track system.

Handles logging of detections, tracks, and performance metrics.
"""

import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from loguru import logger

from ..core.base_classes import Detection, Track, FrameData


class DetectionTrackLogger:
    """Handles logging of detection and tracking results."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize logger.
        
        Args:
            config: Logging configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Output settings
        self.save_video = config.get('save_video', True)
        self.save_detections = config.get('save_detections', True)
        self.save_tracks = config.get('save_tracks', True)
        
        # Output files
        self.video_output = self.output_dir / config.get('video_output', 'tracked_video.mp4')
        self.detections_output = self.output_dir / config.get('detections_output', 'detections.json')
        self.tracks_output = self.output_dir / config.get('tracks_output', 'tracks.json')
        self.metrics_output = self.output_dir / 'metrics.csv'
        
        # Data storage
        self.all_detections = []
        self.all_tracks = []
        self.frame_metrics = []
        
        # Video writer
        self.video_writer = None
        self.video_fps = 30.0
        self.video_size = (640, 480)
        
        logger.info(f"Logger initialized. Output directory: {self.output_dir}")
    
    def setup_video_writer(self, fps: float, frame_size: tuple) -> None:
        """
        Setup video writer for saving output video.
        
        Args:
            fps: Frame rate of output video
            frame_size: (width, height) of output video
        """
        if not self.save_video:
            return
        
        self.video_fps = fps
        self.video_size = frame_size
        
        # Try different codecs in order of preference
        codecs_to_try = [
            ('mp4v', '.mp4'),  # MPEG-4 Part 2
            ('XVID', '.avi'),  # Xvid
            ('MJPG', '.avi'),  # Motion JPEG
            ('X264', '.mp4'),  # H.264
        ]
        
        for codec_name, extension in codecs_to_try:
            try:
                # Update output filename with correct extension
                output_path = self.video_output
                if not str(output_path).endswith(extension):
                    output_path = output_path.with_suffix(extension)
                
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                self.video_writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    fps,
                    frame_size
                )
                
                # Test if writer is properly initialized
                if self.video_writer.isOpened():
                    logger.info(f"Video writer initialized: {output_path} using {codec_name} codec")
                    self.video_output = output_path  # Update path with correct extension
                    return
                else:
                    self.video_writer.release()
                    self.video_writer = None
                    
            except Exception as e:
                logger.debug(f"Failed to initialize video writer with {codec_name}: {e}")
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                continue
        
        # If all codecs fail, disable video saving
        logger.warning("Failed to initialize video writer with any codec. Video saving disabled.")
        self.save_video = False
        self.video_writer = None
    
    def log_frame_data(self, frame_data: FrameData) -> None:
        """
        Log data for a single frame.
        
        Args:
            frame_data: FrameData object containing frame information
        """
        # Store detections
        if self.save_detections:
            frame_detections = []
            for detection in frame_data.detections:
                det_dict = {
                    'frame_id': frame_data.frame_id,
                    'timestamp': frame_data.timestamp,
                    'bbox': detection.bbox,
                    'confidence': detection.confidence,
                    'class_id': detection.class_id,
                    'class_name': detection.class_name
                }
                frame_detections.append(det_dict)
            self.all_detections.extend(frame_detections)
        
        # Store tracks
        if self.save_tracks:
            frame_tracks = []
            for track in frame_data.tracks:
                if track.latest_detection:
                    track_dict = {
                        'frame_id': frame_data.frame_id,
                        'timestamp': frame_data.timestamp,
                        'track_id': track.track_id,
                        'bbox': track.latest_detection.bbox,
                        'confidence': track.latest_detection.confidence,
                        'class_id': track.latest_detection.class_id,
                        'class_name': track.latest_detection.class_name,
                        'is_confirmed': track.is_confirmed,
                        'duration': track.duration,
                        'frames_since_update': track.frames_since_update
                    }
                    frame_tracks.append(track_dict)
            self.all_tracks.extend(frame_tracks)
        
        # Store performance metrics
        metrics = {
            'frame_id': frame_data.frame_id,
            'timestamp': frame_data.timestamp,
            'processing_time': frame_data.processing_time,
            'fps': frame_data.fps,
            'num_detections': len(frame_data.detections),
            'num_tracks': len([t for t in frame_data.tracks if not t.is_deleted])
        }
        self.frame_metrics.append(metrics)
        
        # Write frame to video if enabled
        if self.video_writer and self.video_writer.isOpened() and self.save_video:
            try:
                # Validate frame format and size
                frame = frame_data.image
                
                # Ensure frame is the correct size
                if frame.shape[:2] != (self.video_size[1], self.video_size[0]):  # (height, width)
                    frame = cv2.resize(frame, self.video_size)
                
                # Ensure frame is BGR format with uint8 dtype
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                else:
                    logger.warning(f"Unexpected frame format for video writing: {frame.shape}")
                    return
                
                # Write frame
                success = self.video_writer.write(frame)
                if not success:
                    logger.warning("Failed to write frame to video file")
                    
            except Exception as e:
                logger.error(f"Error writing frame to video: {e}")
                # Disable video writing to prevent further errors
                self.save_video = False
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
    
    def draw_annotations(self, frame: np.ndarray, detections: List[Detection], 
                        tracks: List[Track], show_detections: bool = True, 
                        show_tracks: bool = True, show_fps: bool = True,
                        fps: float = 0.0) -> np.ndarray:
        """
        Draw annotations on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            tracks: List of tracks
            show_detections: Whether to show detection boxes
            show_tracks: Whether to show track information
            show_fps: Whether to show FPS
            fps: Current FPS value
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw detections
        if show_detections:
            for detection in detections:
                x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw tracks
        if show_tracks:
            for track in tracks:
                if track.is_deleted or not track.latest_detection:
                    continue
                
                x1, y1, x2, y2 = [int(coord) for coord in track.latest_detection.bbox]
                
                # Choose color based on track ID
                colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
                         (0, 255, 255), (128, 0, 128), (255, 165, 0)]
                color = colors[track.track_id % len(colors)]
                
                # Draw track box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw track ID
                track_label = f"ID: {track.track_id}"
                if track.is_confirmed:
                    track_label += " ✓"
                
                cv2.putText(annotated_frame, track_label, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw track center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(annotated_frame, (center_x, center_y), 3, color, -1)
        
        # Draw FPS
        if show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(annotated_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return annotated_frame
    
    def save_results(self) -> None:
        """Save all logged results to files."""
        # Save detections
        if self.save_detections and self.all_detections:
            with open(self.detections_output, 'w') as f:
                json.dump(self.all_detections, f, indent=2)
            logger.info(f"Detections saved to {self.detections_output}")
        
        # Save tracks
        if self.save_tracks and self.all_tracks:
            with open(self.tracks_output, 'w') as f:
                json.dump(self.all_tracks, f, indent=2)
            logger.info(f"Tracks saved to {self.tracks_output}")
        
        # Save metrics
        if self.frame_metrics:
            with open(self.metrics_output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.frame_metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.frame_metrics)
            logger.info(f"Metrics saved to {self.metrics_output}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the session.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.frame_metrics:
            return {}
        
        total_frames = len(self.frame_metrics)
        total_time = sum(m['processing_time'] for m in self.frame_metrics)
        avg_fps = sum(m['fps'] for m in self.frame_metrics) / total_frames
        total_detections = sum(m['num_detections'] for m in self.frame_metrics)
        
        # Count unique tracks
        unique_tracks = set()
        for track_data in self.all_tracks:
            unique_tracks.add(track_data['track_id'])
        
        stats = {
            'total_frames': total_frames,
            'total_processing_time': total_time,
            'average_fps': avg_fps,
            'total_detections': total_detections,
            'unique_tracks': len(unique_tracks),
            'avg_detections_per_frame': total_detections / total_frames if total_frames > 0 else 0
        }
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.video_writer:
            self.video_writer.release()
            logger.info("Video writer released")
        
        # Print summary
        stats = self.get_summary_stats()
        if stats:
            logger.info("Session Summary:")
            logger.info(f"  Total frames processed: {stats['total_frames']}")
            logger.info(f"  Average FPS: {stats['average_fps']:.2f}")
            logger.info(f"  Total detections: {stats['total_detections']}")
            logger.info(f"  Unique tracks: {stats['unique_tracks']}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.save_results()
        self.cleanup() 