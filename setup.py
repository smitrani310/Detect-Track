#!/usr/bin/env python3
"""
Setup script for Detect-Track system.
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    dirs = ['logs', 'outputs', 'models', 'data']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def download_sample_models():
    """Download sample models for testing."""
    print("Downloading sample models...")
    try:
        # This will download models on first use
        import torch
        print("✓ PyTorch available")
        
        # Test YOLOv8 import
        try:
            from ultralytics import YOLO
            print("✓ Ultralytics available")
        except ImportError:
            print("⚠ Ultralytics not available - YOLOv8 will not work")
        
        # Test DeepSORT import
        try:
            from deep_sort_realtime import DeepSort
            print("✓ DeepSORT available")
        except ImportError:
            print("⚠ DeepSORT not available - install with: pip install deep-sort-realtime")
            
        return True
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        return False

def test_system():
    """Test the system components."""
    print("Testing system components...")
    try:
        # Test imports
        sys.path.append(str(Path(__file__).parent / 'src'))
        
        from src.config.config_manager import ConfigManager
        from src.core.base_classes import Detection, Track
        
        # Test config
        config = ConfigManager()
        print("✓ Configuration manager working")
        
        print("✓ System test passed")
        return True
    except Exception as e:
        print(f"✗ System test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("="*60)
    print("DETECT-TRACK SYSTEM SETUP")
    print("="*60)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("Setup failed due to dependency installation issues")
        return False
    
    # Test models
    download_sample_models()
    
    # Test system
    if not test_system():
        print("Setup failed due to system test issues")
        return False
    
    print("\n" + "="*60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("You can now run the system with:")
    print("  python main.py --camera                    # Camera input")
    print("  python main.py --video sample.mp4          # Video file")
    print("  python main.py --interactive --camera      # Interactive mode")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 