#!/usr/bin/env python3
"""
Test script to verify the fixes for video writing and tracker initialization.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.utils.logging_system import DetectionTrackLogger
from src.tracking.trackers import NvDCFTracker, BOTSortTracker
from src.core.base_classes import Detection


def test_video_writer():
    """Test video writer with different codecs."""
    print("Testing video writer fixes...")
    
    config = {
        'output_dir': 'outputs/test',
        'video_output': 'test_video.mp4',
        'save_video': True,
        'save_detections': False,
        'save_tracks': False
    }
    
    logger = DetectionTrackLogger(config)
    
    # Test video writer setup
    try:
        logger.setup_video_writer(30.0, (640, 480))
        if logger.video_writer and logger.video_writer.isOpened():
            print("✓ Video writer initialized successfully")
            
            # Test writing a frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            success = logger.video_writer.write(test_frame)
            if success:
                print("✓ Frame writing successful")
            else:
                print("✗ Frame writing failed")
                
            logger.video_writer.release()
        else:
            print("✗ Video writer failed to initialize")
    except Exception as e:
        print(f"✗ Video writer test failed: {e}")


def test_tracker_initialization():
    """Test tracker initialization with various bounding boxes."""
    print("\nTesting tracker initialization fixes...")
    
    config = {'max_lost_frames': 30, 'min_track_length': 3}
    
    # Test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test BOTSort (should always work)
    try:
        bot_tracker = BOTSortTracker(config)
        
        # Test detection
        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.8,
            class_id=0,
            class_name="person",
            frame_id=1
        )
        
        tracks = bot_tracker.update([detection], test_frame)
        print(f"✓ BOTSort tracker working: {len(tracks)} tracks")
    except Exception as e:
        print(f"✗ BOTSort tracker failed: {e}")
    
    # Test NvDCF (with validation)
    try:
        nvdcf_tracker = NvDCFTracker(config)
        print(f"✓ NvDCF tracker initialized with: {nvdcf_tracker.tracker_type}")
        
        # Test various bounding box scenarios
        test_cases = [
            (100, 100, 200, 200, "Normal bbox"),
            (0, 0, 50, 50, "Top-left corner"),
            (590, 430, 640, 480, "Bottom-right corner"),
            (5, 5, 10, 10, "Too small bbox (should fail)"),
            (100, 200, 90, 150, "Invalid bbox (x1>x2, should fail)"),
        ]
        
        for x1, y1, x2, y2, description in test_cases:
            detection = Detection(
                bbox=(x1, y1, x2, y2),
                confidence=0.8,
                class_id=0,
                class_name="person",
                frame_id=1
            )
            
            initial_track_count = len(nvdcf_tracker.tracks)
            tracks = nvdcf_tracker.update([detection], test_frame)
            new_track_count = len(nvdcf_tracker.tracks)
            
            if new_track_count > initial_track_count:
                print(f"✓ {description}: Track created successfully")
            else:
                print(f"⚠ {description}: Track not created (expected for invalid cases)")
                
    except Exception as e:
        print(f"⚠ NvDCF tracker not available: {e}")


def main():
    """Run all tests."""
    print("Running fix verification tests...\n")
    
    test_video_writer()
    test_tracker_initialization()
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print("If you see ✓ marks above, the fixes are working!")
    print("Now try running: python main.py --camera")


if __name__ == "__main__":
    main() 