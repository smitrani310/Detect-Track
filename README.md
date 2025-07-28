# Detect-Track

A comprehensive, modular object detection and tracking system supporting multiple YOLO models and tracking algorithms.

## Features

### 🎯 Detection Models
- **YOLOv5** (nano, small) - Fast and efficient
- **YOLOv7** - High accuracy
- **YOLOv8** (nano, small) - Latest YOLO architecture
- Easy model switching at runtime

### 🎯 Tracking Algorithms
- **NvDCF** - OpenCV's Normalized Cross-Correlation tracker
- **DeepSORT** - Deep learning-based tracking with re-identification
- **BOTSort** - Motion prediction-based tracking
- All trackers support:
  - Continuing tracks for up to N frames without detections
  - Recovering object identity after partial occlusions

### 🎯 Input Sources
- **Live Camera** - Real-time webcam input
- **Video Files** - Support for various video formats
- Configurable resolution and frame rate

### 🎯 Advanced Features
- **Real-time switching** between models and trackers
- **Comprehensive logging** of detections, tracks, and metrics
- **Video output** with annotations
- **JSON export** of results
- **Performance metrics** tracking
- **Interactive controls**

## Pipeline Architecture

The system follows a clean 5-stage pipeline:

```
1. Frame Extraction → 2. Detection → 3. Conversion → 4. Tracking → 5. Logging
```

1. **Frame Extraction**: Read video frames from camera or file
2. **Detection**: Run YOLO model on each frame
3. **Conversion**: Transform detections to tracker input format
4. **Tracking**: Process frames through selected tracker
5. **Logging**: Record per-frame outputs and generate reports

## Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Additional Setup for YOLOv7
```bash
# YOLOv7 requires additional setup via torch.hub
python -c "import torch; torch.hub.load('WongKinYiu/yolov7', 'yolov7', trust_repo=True)"
```

## Quick Start

### Basic Usage
```bash
# Use camera with default settings
python main.py --camera

# Use video file
python main.py --video path/to/your/video.mp4

# Specify model and tracker
python main.py --camera --model yolov8s --tracker deepsort
```

### Interactive Mode
```bash
# Enable runtime switching of models and trackers
python main.py --camera --interactive
```

In interactive mode, you can:
- Press `1-5` to switch between YOLO models
- Press `n`, `d`, `b` to switch between trackers
- Press `s` for status, `h` for help, `q` to quit

## Configuration

The system uses YAML configuration files. See `src/config/config.yaml` for all options:

```yaml
# Video Input
video:
  source: 'camera'  # or 'file'
  camera:
    device_id: 0
    width: 640
    height: 480
  file:
    path: 'path/to/video.mp4'

# Detection
detection:
  model: 'yolov8n'  # yolov5n, yolov5s, yolov7, yolov8n, yolov8s
  confidence_threshold: 0.5

# Tracking
tracking:
  algorithm: 'deepsort'  # nvdcf, deepsort, botsort
  max_lost_frames: 30
```

### Custom Configuration
```bash
python main.py --config custom_config.yaml
```

## Advanced Usage

### Command Line Options
```bash
python main.py [OPTIONS]

Options:
  --config PATH          Custom configuration file
  --camera              Force camera input
  --video PATH          Video file path
  --model MODEL         YOLO model (yolov5n|yolov5s|yolov7|yolov8n|yolov8s)
  --tracker TRACKER     Tracker (nvdcf|deepsort|botsort)
  --no-display         Headless mode (no video display)
  --interactive        Enable runtime model/tracker switching
  --output-dir PATH    Output directory for results
  --confidence FLOAT   Detection confidence threshold
  --verbose            Enable debug logging
```

### Examples

```bash
# High-accuracy setup
python main.py --video sample.mp4 --model yolov8s --tracker deepsort

# Fast setup for real-time
python main.py --camera --model yolov5n --tracker nvdcf

# Headless processing
python main.py --video input.mp4 --no-display --output-dir results/

# Custom configuration
python main.py --config configs/high_precision.yaml --interactive
```

## Tracker Comparison

| Tracker | Speed | Accuracy | Features |
|---------|-------|----------|----------|
| **NvDCF** | ⭐⭐⭐ | ⭐⭐ | Fast, lightweight, good for real-time |
| **DeepSORT** | ⭐⭐ | ⭐⭐⭐ | Re-identification, handles occlusions well |
| **BOTSort** | ⭐⭐ | ⭐⭐⭐ | Motion prediction, robust tracking |

## Model Comparison

| Model | Speed | Accuracy | Size | Best Use Case |
|-------|-------|----------|------|---------------|
| **YOLOv5n** | ⭐⭐⭐ | ⭐⭐ | 1.9MB | Real-time applications |
| **YOLOv5s** | ⭐⭐ | ⭐⭐⭐ | 14MB | Balanced speed/accuracy |
| **YOLOv7** | ⭐⭐ | ⭐⭐⭐ | 75MB | High accuracy requirements |
| **YOLOv8n** | ⭐⭐⭐ | ⭐⭐ | 3.2MB | Latest nano model |
| **YOLOv8s** | ⭐⭐ | ⭐⭐⭐ | 11MB | Latest small model |

## Output Files

The system generates several output files in the specified output directory:

```
outputs/
├── tracked_video.mp4      # Annotated video with detections and tracks
├── detections.json        # Per-frame detection results
├── tracks.json           # Tracking results with track IDs
└── metrics.csv          # Performance metrics (FPS, processing time)
```

### Sample JSON Output

**Detections:**
```json
[
  {
    "frame_id": 1,
    "timestamp": 1699123456.789,
    "bbox": [100, 150, 200, 300],
    "confidence": 0.85,
    "class_id": 0,
    "class_name": "person"
  }
]
```

**Tracks:**
```json
[
  {
    "frame_id": 1,
    "timestamp": 1699123456.789,
    "track_id": 1,
    "bbox": [100, 150, 200, 300],
    "confidence": 0.85,
    "class_id": 0,
    "class_name": "person",
    "is_confirmed": true,
    "duration": 15,
    "frames_since_update": 0
  }
]
```

## Architecture

```
src/
├── config/           # Configuration management
├── core/            # Base classes and data structures
├── detection/       # YOLO model implementations
├── tracking/        # Tracking algorithm implementations
├── video/          # Video input sources
├── utils/          # Logging and utilities
└── pipeline/       # Main pipeline orchestrator
```

## Key Features Implementation

### ✅ Multi-Model Support
Switch between YOLOv5, v7, and v8 models at runtime without restarting.

### ✅ Robust Tracking
All trackers implement:
- Track continuation for up to N frames without detections
- Identity recovery after occlusions
- Confirmed track management

### ✅ Flexible Input
Support for both live camera and video file inputs with configurable parameters.

### ✅ Comprehensive Logging
- Frame-by-frame detection and tracking results
- Performance metrics and timing
- Annotated video output
- JSON export for analysis

### ✅ Real-time Performance
Optimized pipeline with configurable frame skipping and resizing for real-time performance.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   python main.py --model yolov5n  # Use smaller model
   ```

2. **Camera not found**
   ```bash
   python main.py --camera --verbose  # Check debug logs
   ```

3. **Low FPS**
   - Use smaller model (yolov5n, yolov8n)
   - Enable frame skipping in config
   - Reduce input resolution

4. **DeepSORT import error**
   ```bash
   pip install deep-sort-realtime
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv5: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- YOLOv7: [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- YOLOv8: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- DeepSORT: [levan92/deep_sort_realtime](https://github.com/levan92/deep_sort_realtime)

---

**Built with ❤️ for computer vision enthusiasts**