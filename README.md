# Detect-Track

A comprehensive, modular object detection and tracking system with **enterprise-grade DeepStream-style pipeline** featuring TensorRT GPU acceleration, automatic PyTorch fallback, and multi-threaded architecture for maximum performance.

## Features

### 🚀 **Dual Pipeline Architecture**
- **🔥 DeepStream-Style Pipeline** - Enterprise-grade with TensorRT GPU acceleration (22+ FPS)
- **🐍 Standard Pipeline** - Reliable PyTorch-based processing (17+ FPS)
- **⚡ Automatic Failover** - Seamless switching between TensorRT and PyTorch
- **🧵 Multi-threaded** - Parallel capture, inference, and display threads

### 🎯 Detection Models
- **YOLOv5** (nano, small) - Fast and efficient
- **YOLOv7** - High accuracy  
- **YOLOv8** (nano, small) - Latest YOLO architecture
- **🔥 TensorRT Optimization** - GPU-accelerated inference engines
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
- **🔥 TensorRT GPU Acceleration** - Maximum performance inference engines
- **⚡ Automatic Failover** - PyTorch fallback for robust operation
- **🧵 Multi-threaded Architecture** - Parallel processing pipelines
- **📊 Enterprise Monitoring** - Detailed performance statistics and profiling
- **Real-time switching** between models and trackers
- **Comprehensive logging** of detections, tracks, and metrics
- **Video output** with annotations
- **JSON export** of results
- **Interactive controls**

## Pipeline Architecture

The system offers **two high-performance pipelines** with automatic failover:

### 🔥 **DeepStream-Style Pipeline** (Enterprise)
```
Multi-threaded GPU Pipeline:
📹 Capture Thread → 🔥 TensorRT Inference → 🎯 Tracking → 📊 Display Thread
                      ↓ (fallback)
                   🐍 PyTorch Inference
```

**Features:**
- **TensorRT GPU acceleration** - Maximum inference speed  
- **Multi-threaded architecture** - Parallel capture, inference, display
- **Automatic PyTorch fallback** - Robust error handling
- **Enterprise monitoring** - Detailed performance profiling
- **GPU memory management** - Optimized buffer pools

### 🐍 **Standard Pipeline** (Reliable)
```
Single-threaded Reliable Pipeline:
1. Frame Extraction → 2. Detection → 3. Conversion → 4. Tracking → 5. Logging
```

**Features:**
- **PyTorch-based inference** - Proven stability
- **Real-time switching** - Dynamic model/tracker changes  
- **Interactive controls** - Runtime configuration
- **Comprehensive logging** - Detailed output files

## Installation

### Prerequisites
- Python 3.8+
- **CUDA 11.0+** (required for DeepStream pipeline)
- **NVIDIA GPU** with CUDA support (for TensorRT acceleration)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔥 **DeepStream Pipeline Setup** (For Maximum Performance)
For enterprise-grade TensorRT acceleration:

```bash
# Install TensorRT (for GPU acceleration)
pip install nvidia-tensorrt

# Install PyCUDA (for GPU memory management)  
pip install pycuda

# Verify TensorRT installation
python -c "import tensorrt as trt; print(f'TensorRT {trt.__version__} ready')"
```

**Note:** DeepStream pipeline automatically falls back to PyTorch if TensorRT is unavailable.

### Additional Setup for YOLOv7
```bash
# YOLOv7 requires additional setup via torch.hub
python -c "import torch; torch.hub.load('WongKinYiu/yolov7', 'yolov7', trust_repo=True)"
```

## Quick Start

### 🔥 **DeepStream Pipeline** (Maximum Performance)
```bash
# Enterprise-grade GPU acceleration
python main_deepstream.py --camera --tracker botsort

# Video file with TensorRT acceleration  
python main_deepstream.py --video path/to/your/video.mp4 --model yolov8n

# Custom precision and verbose logging
python main_deepstream.py --camera --precision fp16 --verbose
```

### 🐍 **Standard Pipeline** (Reliable & Interactive)
```bash
# Use camera with default settings
python main.py --camera

# Use video file
python main.py --video path/to/your/video.mp4

# Specify model and tracker
python main.py --camera --model yolov8s --tracker deepsort
```

### 📊 **Performance Comparison**
| Pipeline | FPS | Features | Best For |
|----------|-----|----------|----------|
| **DeepStream** | 22+ FPS | TensorRT, Multi-threading | Production, High throughput |
| **Standard** | 17+ FPS | Interactive, Switching | Development, Flexibility |

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

#### 🔥 **DeepStream Pipeline** (`main_deepstream.py`)
```bash
python main_deepstream.py [OPTIONS]

Options:
  --camera              Use camera input
  --video PATH          Video file path
  --camera-id INT       Camera device ID (default: 0)
  --model MODEL         YOLO model (yolov5n|yolov8n|yolov8s)
  --tracker TRACKER     Tracker (botsort|nvdcf)
  --precision PRECISION TensorRT precision (fp32|fp16|int8)
  --confidence FLOAT    Detection confidence threshold
  --output-dir PATH     Output directory for results
  --no-display         Headless mode (no video display)
  --verbose            Enable detailed logging and profiling
```

#### 🐍 **Standard Pipeline** (`main.py`)
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

#### 🔥 **DeepStream Pipeline Examples**
```bash
# Maximum performance real-time processing
python main_deepstream.py --camera --tracker botsort --precision fp16

# High-throughput video processing
python main_deepstream.py --video sample.mp4 --model yolov8s --verbose

# Headless production processing  
python main_deepstream.py --video input.mp4 --no-display --output-dir results/

# Enterprise monitoring with detailed stats
python main_deepstream.py --camera --verbose --tracker botsort
```

#### 🐍 **Standard Pipeline Examples**  
```bash
# High-accuracy setup with interactive controls
python main.py --video sample.mp4 --model yolov8s --tracker deepsort --interactive

# Fast real-time with runtime switching
python main.py --camera --model yolov5n --tracker nvdcf --interactive

# Development and testing
python main.py --video input.mp4 --no-display --output-dir results/

# Custom configuration
python main.py --config configs/high_precision.yaml --interactive
```

## Performance Benchmarks

### 🔥 **DeepStream Pipeline Performance**
| Tracker | FPS | Inference Time | Best Use Case |
|---------|-----|----------------|---------------|
| **BOTSort** | 22+ FPS | 31ms | Production, robust tracking |
| **NvDCF** | 25+ FPS | 28ms | High-speed, lightweight |

### 🐍 **Standard Pipeline Performance**  
| Tracker | FPS | Features | Best Use Case |
|---------|-----|----------|---------------|
| **BOTSort** | 17+ FPS | Motion prediction, robust | Development, testing |
| **NvDCF** | 20+ FPS | Fast, lightweight | Real-time applications |
| **DeepSORT** | 15+ FPS | Re-identification, occlusion handling | High accuracy needs |

### 📊 **Tracker Feature Comparison**
| Tracker | Speed | Accuracy | Features |
|---------|-------|----------|----------|
| **BOTSort** | ⭐⭐⭐ | ⭐⭐⭐ | Motion prediction, robust tracking, confirmed tracks |
| **NvDCF** | ⭐⭐⭐ | ⭐⭐ | Fast, lightweight, good for real-time |
| **DeepSORT** | ⭐⭐ | ⭐⭐⭐ | Re-identification, handles occlusions well |

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
├── pipeline/       # Standard pipeline orchestrator
└── deepstream/     # 🔥 Enterprise DeepStream pipeline
    ├── deepstream_pipeline.py  # Multi-threaded GPU pipeline
    ├── tensorrt_yolo.py        # TensorRT inference engine
    └── __init__.py             # Module interface

📁 Entry Points:
├── main.py              # 🐍 Standard pipeline entry point
├── main_deepstream.py   # 🔥 DeepStream pipeline entry point  
└── requirements.txt     # Python dependencies
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

## 📺 Video Resolution Handling

The system handles video resolution differently based on the input source:

### 🎥 **Camera Sources**
- **Resizing allowed** for performance optimization
- Configurable via `performance.resize_input` and `performance.target_size`
- Default: Resize to 640x640 for consistent processing
- Example: Camera input at 1920x1080 → Resized to 640x640

### 🎬 **Video Files**  
- **Original resolution preserved** automatically
- Ignores `performance.resize_input` setting to maintain video quality
- No interpolation artifacts from resizing
- Example: Video file at 1920x1080 → Processed at 1920x1080

### ⚙️ **Configuration**
```yaml
performance:
  resize_input: true  # Only affects camera sources
  target_size: [640, 640]  # Only applied to camera input
```

This ensures optimal performance for live camera feeds while preserving the original quality and aspect ratio of video files.

## 🔥 DeepStream Configuration

The DeepStream pipeline supports additional configuration options:

```yaml
# DeepStream-specific settings
detection:
  tensorrt_precision: 'fp16'  # fp32, fp16, int8
  workspace_size: 1073741824  # 1GB TensorRT workspace

# Multi-threading settings  
performance:
  gpu_memory_pool_size: 10    # Number of GPU memory buffers
  max_queue_size: 30          # Frame queue buffer size
  
# Enterprise monitoring
logging:
  enable_profiling: true      # Detailed performance stats
  log_inference_times: true   # Per-frame timing
```

## Troubleshooting

### Common Issues

#### 🔥 **DeepStream Pipeline Issues**

1. **TensorRT not found**
   ```bash
   pip install nvidia-tensorrt pycuda
   # Pipeline automatically falls back to PyTorch
   ```

2. **TensorRT inference fails**
   ```bash
   python main_deepstream.py --verbose  # Check detailed logs
   # System automatically uses PyTorch fallback
   ```

3. **CUDA out of memory**
   ```bash
   python main_deepstream.py --model yolov5n --precision fp16
   ```

#### 🐍 **Standard Pipeline Issues**

1. **Camera not found**
   ```bash
   python main.py --camera --verbose  # Check debug logs
   ```

2. **Low FPS**
   - Use smaller model (yolov5n, yolov8n)
   - Enable frame skipping in config
   - Reduce input resolution

3. **DeepSORT import error**
   ```bash
   pip install deep-sort-realtime
   # Use BOTSort instead: --tracker botsort
   ```

4. **OpenCV tracker issues**
   ```bash
   pip install opencv-contrib-python  # For NvDCF tracker
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

### Core Technologies
- **YOLOv5**: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- **YOLOv7**: [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- **YOLOv8**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **DeepSORT**: [levan92/deep_sort_realtime](https://github.com/levan92/deep_sort_realtime)

### Enterprise GPU Acceleration
- **NVIDIA TensorRT**: High-performance deep learning inference
- **NVIDIA CUDA**: Parallel computing platform and programming model
- **PyCUDA**: Python bindings for CUDA
- **NVIDIA DeepStream**: Inspiration for enterprise-grade pipeline architecture

---

**🚀 Built with ❤️ for computer vision enthusiasts - Now with enterprise-grade GPU acceleration!**