#!/usr/bin/env python3
"""
DeepStream High-Performance Pipeline
Enterprise-grade object detection and tracking with maximum GPU acceleration

Expected Performance Gains:
- 3-5x faster inference with TensorRT FP16
- GPU-optimized memory management
- Multi-threaded pipeline processing
- Zero-copy operations where possible

Usage:
    python main_deepstream.py --camera                    # Use camera with default settings
    python main_deepstream.py --video path/to/video.mp4   # Process video file
    python main_deepstream.py --camera --model yolov8s    # Use larger model
    python main_deepstream.py --camera --tracker nvdcf    # Use NvDCF tracker
"""

import argparse
import sys
import os
from pathlib import Path
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.deepstream import DeepStreamPipeline, create_deepstream_config


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    log_level = "DEBUG" if verbose else "INFO"
    
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Add file logging
    logger.add(
        "logs/deepstream.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days"
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="DeepStream High-Performance Pipeline - Enterprise Object Detection & Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --camera                           # Camera with BOTSort tracker
  %(prog)s --video demo.mp4                   # Process video file
  %(prog)s --camera --model yolov8s           # Use YOLOv8s model
  %(prog)s --camera --tracker nvdcf           # Use NvDCF tracker
  %(prog)s --camera --no-display              # Headless processing
  %(prog)s --camera --confidence 0.3          # Lower confidence threshold
  %(prog)s --camera --output-dir results      # Custom output directory
        """
    )
    
    # Video source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--camera', 
        action='store_true', 
        help='Use camera as video source'
    )
    source_group.add_argument(
        '--video', 
        type=str, 
        help='Path to video file'
    )
    
    # Camera settings
    parser.add_argument(
        '--camera-id', 
        type=int, 
        default=0, 
        help='Camera ID (default: 0)'
    )
    
    # Model settings
    parser.add_argument(
        '--model', 
        type=str, 
        default='yolov8n',
        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x', 'yolov5n', 'yolov5s'],
        help='YOLO model to use (default: yolov8n)'
    )
    
    # Tracker settings
    parser.add_argument(
        '--tracker', 
        type=str, 
        default='botsort',
        choices=['botsort', 'nvdcf'],
        help='Tracking algorithm (default: botsort - fastest)'
    )
    
    # Detection settings
    parser.add_argument(
        '--confidence', 
        type=float, 
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )
    
    # Output settings
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Output directory (default: outputs)'
    )
    
    # Display settings
    parser.add_argument(
        '--no-display', 
        action='store_true', 
        help='Disable video display (headless mode)'
    )
    
    # Performance settings
    parser.add_argument(
        '--precision', 
        type=str, 
        default='fp16',
        choices=['fp32', 'fp16', 'int8'],
        help='TensorRT precision mode (default: fp16 - 2x speed boost)'
    )
    
    # Debugging
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate and process arguments"""
    
    # Check video file exists
    if args.video and not os.path.exists(args.video):
        logger.error(f"❌ Video file not found: {args.video}")
        return False
    
    # Check confidence range
    if not 0.0 <= args.confidence <= 1.0:
        logger.error(f"❌ Confidence threshold must be between 0.0 and 1.0, got: {args.confidence}")
        return False
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    return True


def print_system_info():
    """Print system and configuration information"""
    logger.info("🚀 DeepStream High-Performance Pipeline")
    logger.info("=" * 60)
    
    try:
        import torch
        import tensorrt as trt
        import cv2
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        logger.info(f"🔥 PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
        logger.info(f"⚡ TensorRT: {trt.__version__}")
        logger.info(f"📹 OpenCV: {cv2.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning("⚠️ CUDA not available")
            
    except Exception as e:
        logger.warning(f"⚠️ Could not get system info: {e}")
    
    logger.info("=" * 60)


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Print system info
    print_system_info()
    
    # Validate arguments
    if not validate_arguments(args):
        return 1
    
    try:
        # Create configuration
        config = create_deepstream_config(
            source_type='camera' if args.camera else 'file',
            source_path=args.video or '',
            camera_id=args.camera_id,
            model=args.model,
            tracker=args.tracker,
            confidence=args.confidence,
            output_dir=args.output_dir,
            display=not args.no_display
        )
        
        # Add TensorRT precision
        config['detection']['tensorrt_precision'] = args.precision
        
        # Print configuration
        logger.info("⚙️ Pipeline Configuration:")
        logger.info(f"  📹 Source: {'Camera ' + str(args.camera_id) if args.camera else args.video}")
        logger.info(f"  🤖 Model: {args.model}")
        logger.info(f"  🎯 Tracker: {args.tracker}")
        logger.info(f"  🔥 TensorRT: {args.precision.upper()} precision")
        logger.info(f"  🎚️ Confidence: {args.confidence}")
        logger.info(f"  📁 Output: {args.output_dir}")
        logger.info(f"  🖥️ Display: {'Enabled' if not args.no_display else 'Disabled'}")
        
        # Initialize and run pipeline
        logger.info("🚀 Initializing DeepStream pipeline...")
        pipeline = DeepStreamPipeline(config)
        
        logger.info("▶️ Starting high-performance processing...")
        logger.info("Press 'q' in video window to quit, or Ctrl+C in terminal")
        
        success = pipeline.run()
        
        if success:
            logger.info("✅ DeepStream pipeline completed successfully")
            return 0
        else:
            logger.error("❌ Pipeline execution failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 DeepStream pipeline interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 