#!/usr/bin/env python3
"""
Detect-Track Main Demo Script

This script demonstrates the complete object detection and tracking system.
It supports:
- Multiple YOLO models (v5/v7/v8 nano/small)
- Multiple tracking algorithms (NvDCF, DeepSORT, BOTSort)
- Live camera or video file input
- Real-time model/tracker switching
- Comprehensive logging and output

Usage:
    python main.py                                  # Use default config
    python main.py --config custom_config.yaml     # Use custom config
    python main.py --camera                         # Force camera input
    python main.py --video path/to/video.mp4       # Use video file
    python main.py --model yolov8s                 # Use specific model
    python main.py --tracker botsort               # Use specific tracker
"""

import argparse
import sys
from pathlib import Path
import time
import threading
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.pipeline.pipeline import DetectionTrackingPipeline
from src.config.config_manager import ConfigManager


class InteractiveDemo:
    """Interactive demo with runtime switching capabilities."""
    
    def __init__(self, pipeline: DetectionTrackingPipeline):
        """Initialize interactive demo."""
        self.pipeline = pipeline
        self.is_running = True
        
    def print_controls(self):
        """Print available controls."""
        print("\n" + "="*60)
        print("DETECT-TRACK INTERACTIVE CONTROLS")
        print("="*60)
        print("MODELS:")
        print("  1 - YOLOv5 Nano    2 - YOLOv5 Small")
        print("  3 - YOLOv7         4 - YOLOv8 Nano    5 - YOLOv8 Small")
        print("\nTRACKERS:")
        print("  n - NvDCF          d - DeepSORT       b - BOTSort")
        print("\nCONTROLS:")
        print("  s - Status         h - Help           q - Quit")
        print("="*60)
        print("Video Controls: Press 'q' in video window to quit")
        print("Current Status:")
        status = self.pipeline.get_status()
        print(f"  Model: {status.get('detector_model', 'Unknown')}")
        print(f"  Tracker: {status.get('tracker_algorithm', 'Unknown')}")
        print(f"  Source: {status.get('video_source', 'Unknown')}")
        print("="*60 + "\n")
    
    def handle_input(self):
        """Handle user input for interactive controls."""
        model_mapping = {
            '1': 'yolov5n',
            '2': 'yolov5s', 
            '3': 'yolov7',
            '4': 'yolov8n',
            '5': 'yolov8s'
        }
        
        tracker_mapping = {
            'n': 'nvdcf',
            'd': 'deepsort',
            'b': 'botsort'
        }
        
        while self.is_running and self.pipeline.is_running:
            try:
                user_input = input("Enter command (h for help): ").strip().lower()
                
                if user_input == 'q':
                    print("Stopping pipeline...")
                    self.pipeline.stop()
                    self.is_running = False
                    break
                    
                elif user_input == 'h':
                    self.print_controls()
                    
                elif user_input == 's':
                    status = self.pipeline.get_status()
                    print(f"\nCurrent Status:")
                    print(f"  Running: {status['is_running']}")
                    print(f"  Frames: {status['frame_count']}")
                    print(f"  Elapsed: {status['elapsed_time']:.1f}s")
                    print(f"  FPS: {status['current_fps']:.1f}")
                    print(f"  Model: {status['detector_model']}")
                    print(f"  Tracker: {status['tracker_algorithm']}\n")
                    
                elif user_input in model_mapping:
                    model_name = model_mapping[user_input]
                    print(f"Switching to {model_name}...")
                    if self.pipeline.switch_detector(model_name):
                        print(f"✓ Successfully switched to {model_name}")
                    else:
                        print(f"✗ Failed to switch to {model_name}")
                        
                elif user_input in tracker_mapping:
                    tracker_name = tracker_mapping[user_input]
                    print(f"Switching to {tracker_name}...")
                    if self.pipeline.switch_tracker(tracker_name):
                        print(f"✓ Successfully switched to {tracker_name}")
                    else:
                        print(f"✗ Failed to switch to {tracker_name}")
                        
                else:
                    print("Unknown command. Press 'h' for help.")
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nStopping pipeline...")
                self.pipeline.stop()
                self.is_running = False
                break
    
    def run(self):
        """Run the interactive demo."""
        self.print_controls()
        
        # Start input handler in separate thread
        input_thread = threading.Thread(target=self.handle_input, daemon=True)
        input_thread.start()
        
        # Run the pipeline
        self.pipeline.run()
        
        # Cleanup
        self.is_running = False
        print("\nDemo completed!")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect-Track: Object Detection and Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Default config
  python main.py --camera                          # Force camera input
  python main.py --video sample.mp4                # Use video file
  python main.py --model yolov8s                   # Use YOLOv8 Small
  python main.py --tracker deepsort                # Use DeepSORT tracker
  python main.py --config custom.yaml              # Custom config
  python main.py --no-display                      # Headless mode
  python main.py --interactive                     # Interactive mode

Supported Models: yolov5n, yolov5s, yolov7, yolov8n, yolov8s
Supported Trackers: nvdcf, deepsort, botsort
        """
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--camera', action='store_true',
                       help='Use camera input (overrides config)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (overrides config)')
    parser.add_argument('--model', type=str, default=None,
                       choices=['yolov5n', 'yolov5s', 'yolov7', 'yolov8n', 'yolov8s'],
                       help='YOLO model to use (overrides config)')
    parser.add_argument('--tracker', type=str, default=None,
                       choices=['nvdcf', 'deepsort', 'botsort'],
                       help='Tracking algorithm to use (overrides config)')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without video display (headless mode)')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive mode for runtime switching')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (overrides config)')
    parser.add_argument('--confidence', type=float, default=None,
                       help='Detection confidence threshold (overrides config)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    log_level = "DEBUG" if verbose else "INFO"
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file logger
    logger.add(
        "logs/detect_track_{time}.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="100 MB",
        retention="10 days"
    )


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        logger.info("Starting Detect-Track System")
        logger.info(f"Arguments: {vars(args)}")
        
        # Create pipeline with custom config if provided
        pipeline = DetectionTrackingPipeline(args.config)
        
        # Override config with command line arguments
        if args.camera:
            pipeline.config_manager.set('video.source', 'camera')
            
        if args.video:
            pipeline.config_manager.set('video.source', 'file')
            pipeline.config_manager.set('video.file.path', args.video)
            
        if args.model:
            pipeline.config_manager.set('detection.model', args.model)
            
        if args.tracker:
            pipeline.config_manager.set('tracking.algorithm', args.tracker)
            
        if args.no_display:
            pipeline.config_manager.set('display.show_video', False)
            
        if args.output_dir:
            pipeline.config_manager.set('logging.output_dir', args.output_dir)
            
        if args.confidence:
            pipeline.config_manager.set('detection.confidence_threshold', args.confidence)
        
        # Print configuration summary
        print("\n" + "="*60)
        print("DETECT-TRACK SYSTEM CONFIGURATION")
        print("="*60)
        print(f"Video Source: {pipeline.config_manager.get('video.source')}")
        if pipeline.config_manager.get('video.source') == 'file':
            print(f"Video File: {pipeline.config_manager.get('video.file.path')}")
        print(f"Detection Model: {pipeline.config_manager.get('detection.model')}")
        print(f"Tracking Algorithm: {pipeline.config_manager.get('tracking.algorithm')}")
        print(f"Confidence Threshold: {pipeline.config_manager.get('detection.confidence_threshold')}")
        print(f"Output Directory: {pipeline.config_manager.get('logging.output_dir')}")
        print(f"Display Video: {pipeline.config_manager.get('display.show_video')}")
        print("="*60)
        
        # Run pipeline
        if args.interactive:
            print("\nStarting Interactive Mode...")
            print("The pipeline will start, and you can switch models/trackers in real-time.")
            demo = InteractiveDemo(pipeline)
            demo.run()
        else:
            print("\nStarting Pipeline...")
            print("Press 'q' in the video window to quit, or Ctrl+C in terminal.")
            pipeline.run()
        
        logger.info("Detect-Track System completed successfully")
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 