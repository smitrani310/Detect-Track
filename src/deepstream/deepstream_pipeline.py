"""
DeepStream-Style High-Performance Pipeline
Enterprise-grade video analytics with maximum GPU acceleration
"""

import time
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from dataclasses import dataclass

from .tensorrt_yolo import TensorRTYOLO
from ..core.base_classes import Detection, Track, BaseVideoSource, FrameData
from ..video.video_sources import CameraSource, FileSource
from ..tracking.trackers import BOTSortTracker, NvDCFTracker
from ..utils.logging_system import DetectionTrackLogger


@dataclass
class FrameMetadata:
    """Frame metadata for GPU pipeline processing"""
    frame_id: int
    timestamp: float
    original_shape: Tuple[int, int]
    preprocessed_data: np.ndarray
    gpu_memory_ptr: Optional[Any] = None


class GPUMemoryPool:
    """GPU memory pool for zero-copy operations"""
    
    def __init__(self, pool_size: int = 10, frame_size: Tuple[int, int] = (640, 640)):
        self.pool_size = pool_size
        self.frame_size = frame_size
        self.available_buffers = queue.Queue()
        self.in_use_buffers = set()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize GPU memory buffers"""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            buffer_size = self.frame_size[0] * self.frame_size[1] * 3 * 4  # Float32
            
            for i in range(self.pool_size):
                gpu_buffer = cuda.mem_alloc(buffer_size)
                self.available_buffers.put(gpu_buffer)
                
            logger.info(f"🔥 GPU memory pool initialized: {self.pool_size} buffers")
        except Exception as e:
            logger.warning(f"⚠️ GPU memory pool initialization failed: {e}")
    
    def get_buffer(self):
        """Get available GPU buffer"""
        try:
            buffer = self.available_buffers.get_nowait()
            self.in_use_buffers.add(buffer)
            return buffer
        except queue.Empty:
            logger.warning("⚠️ GPU memory pool exhausted, creating temporary buffer")
            import pycuda.driver as cuda
            buffer_size = self.frame_size[0] * self.frame_size[1] * 3 * 4
            return cuda.mem_alloc(buffer_size)
    
    def release_buffer(self, buffer):
        """Release GPU buffer back to pool"""
        if buffer in self.in_use_buffers:
            self.in_use_buffers.remove(buffer)
            self.available_buffers.put(buffer)


class DeepStreamPipeline:
    """
    High-Performance DeepStream-Style Pipeline
    Enterprise-grade video analytics with maximum GPU acceleration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DeepStream pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        self.frame_count = 0
        
        # Performance tracking
        self.performance_stats = {
            'total_frames': 0,
            'total_time': 0.0,
            'inference_times': [],
            'tracking_times': [],
            'preprocessing_times': [],
            'postprocessing_times': []
        }
        
        # Pipeline components
        self.video_source: Optional[BaseVideoSource] = None
        self.tensorrt_yolo: Optional[TensorRTYOLO] = None
        self.tracker = None
        self.logger_system: Optional[DetectionTrackLogger] = None
        self.gpu_memory_pool: Optional[GPUMemoryPool] = None
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=30)  # Prevent memory overflow
        self.result_queue = queue.Queue(maxsize=30)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Display settings
        self.display_enabled = config.get('display', {}).get('enabled', True)
        self.display_fps = True
        self.display_tracks = True
        self.display_detections = True
        
        logger.info("🚀 DeepStream Pipeline initialized")
    
    def setup(self) -> bool:
        """Setup all pipeline components"""
        try:
            logger.info("⚙️ Setting up DeepStream pipeline components...")
            
            # Setup video source
            if not self._setup_video_source():
                return False
            
            # Setup TensorRT YOLO engine
            if not self._setup_tensorrt_yolo():
                return False
            
            # Setup tracker
            if not self._setup_tracker():
                return False
            
            # Setup logger
            if not self._setup_logger():
                return False
            
            # Setup GPU memory pool
            self._setup_gpu_memory_pool()
            
            logger.info("✅ DeepStream pipeline setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline setup failed: {e}")
            return False
    
    def _setup_video_source(self) -> bool:
        """Setup video input source"""
        try:
            if self.config.get('video_source', {}).get('type') == 'camera':
                camera_id = self.config.get('video_source', {}).get('camera_id', 0)
                self.video_source = CameraSource({'camera_id': camera_id})
            else:
                video_path = self.config.get('video_source', {}).get('path', '')
                self.video_source = FileSource({'video_path': video_path})
            
            success = self.video_source.open()
            if success:
                logger.info(f"📹 Video source opened successfully")
                return True
            else:
                logger.error("❌ Failed to open video source")
                return False
                
        except Exception as e:
            logger.error(f"❌ Video source setup failed: {e}")
            return False
    
    def _setup_tensorrt_yolo(self) -> bool:
        """Setup TensorRT YOLO engine"""
        try:
            model_name = self.config.get('detection', {}).get('model', 'yolov8n')
            model_path = f"models/{model_name}.pt"
            engine_path = f"src/models/tensorrt_engines/{model_name}_fp16.engine"
            
            # Check if model file exists
            import os
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ Model file {model_path} not found, downloading...")
                from ultralytics import YOLO
                yolo_temp = YOLO(model_name)  # This will download the model
                model_path = f"{model_name}.pt"
            
            self.tensorrt_yolo = TensorRTYOLO(
                model_path=model_path,
                engine_path=engine_path,
                precision="fp16",  # 2x speed boost
                max_batch_size=1
            )
            
            logger.info("🔥 TensorRT YOLO engine ready for maximum performance")
            return True
            
        except Exception as e:
            logger.error(f"❌ TensorRT YOLO setup failed: {e}")
            return False
    
    def _setup_tracker(self) -> bool:
        """Setup high-performance tracker"""
        try:
            tracker_type = self.config.get('tracking', {}).get('algorithm', 'botsort')
            
            if tracker_type == 'botsort':
                self.tracker = BOTSortTracker(self.config)
                logger.info("🎯 BOTSort tracker initialized for maximum performance")
            elif tracker_type == 'nvdcf':
                self.tracker = NvDCFTracker(self.config)
                logger.info("🎯 NvDCF tracker initialized")
            else:
                logger.error(f"❌ Unsupported tracker type: {tracker_type}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Tracker setup failed: {e}")
            return False
    
    def _setup_logger(self) -> bool:
        """Setup logging system"""
        try:
            # DetectionTrackLogger expects only the config (output_dir is inside config)
            logger_config = self.config.get('logging', {})
            self.logger_system = DetectionTrackLogger(logger_config)
            logger.info("📝 Logging system initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Logger setup failed: {e}")
            return False
    
    def _setup_gpu_memory_pool(self):
        """Setup GPU memory pool for optimization"""
        try:
            self.gpu_memory_pool = GPUMemoryPool(pool_size=10)
        except Exception as e:
            logger.warning(f"⚠️ GPU memory pool setup failed: {e}")
    
    def run(self) -> bool:
        """Run the high-performance pipeline"""
        if not self.setup():
            logger.error("❌ Pipeline setup failed, aborting")
            return False
        
        self.running = True
        pipeline_start = time.time()
        
        try:
            logger.info("🚀 Starting DeepStream high-performance pipeline...")
            
            # Start pipeline threads
            capture_future = self.executor.submit(self._capture_thread)
            process_future = self.executor.submit(self._processing_thread)
            display_future = self.executor.submit(self._display_thread)
            
            # Wait for threads to complete
            capture_future.result()
            process_future.result()
            if self.display_enabled:
                display_future.result()
            
        except KeyboardInterrupt:
            logger.info("🛑 Pipeline interrupted by user")
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            return False
        finally:
            self.running = False
            pipeline_time = time.time() - pipeline_start
            self.performance_stats['total_time'] = pipeline_time
            self._cleanup()
            self._print_performance_stats()
        
        return True
    
    def _capture_thread(self):
        """High-speed frame capture thread"""
        frame_id = 0
        
        while self.running:
            try:
                success, frame = self.video_source.read_frame()
                if not success:
                    break
                
                if frame is None:
                    continue
                
                # Create frame metadata
                metadata = FrameMetadata(
                    frame_id=frame_id,
                    timestamp=time.time(),
                    original_shape=frame.shape[:2],
                    preprocessed_data=frame
                )
                
                # Add to processing queue (non-blocking)
                try:
                    self.frame_queue.put(metadata, timeout=0.001)
                    frame_id += 1
                except queue.Full:
                    # Skip frame if queue is full (maintain real-time processing)
                    pass
                    
            except Exception as e:
                logger.error(f"❌ Capture thread error: {e}")
                break
    
    def _processing_thread(self):
        """High-performance processing thread"""
        while self.running:
            try:
                # Get frame from queue
                try:
                    metadata = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process frame
                result = self._process_frame(metadata)
                
                if result:
                    # Add to display queue
                    try:
                        self.result_queue.put(result, timeout=0.001)
                    except queue.Full:
                        # Skip if display queue is full
                        pass
                
                self.frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Processing thread error: {e}")
    
    def _process_frame(self, metadata: FrameMetadata) -> Optional[Dict[str, Any]]:
        """Process single frame with maximum performance"""
        process_start = time.time()
        
        try:
            frame = metadata.preprocessed_data
            original_shape = metadata.original_shape
            
            # Preprocessing
            preprocess_start = time.time()
            preprocessed_frame = self.tensorrt_yolo.preprocess_frame(frame)
            preprocess_time = (time.time() - preprocess_start) * 1000
            self.performance_stats['preprocessing_times'].append(preprocess_time)
            
            # TensorRT Inference with PyTorch fallback support
            inference_start = time.time()
            raw_outputs = self.tensorrt_yolo.inference(
                preprocessed_frame=preprocessed_frame,
                original_frame=frame
            )
            inference_time = (time.time() - inference_start) * 1000
            self.performance_stats['inference_times'].append(inference_time)
            
            # Post-processing
            postprocess_start = time.time()
            detections_raw = self.tensorrt_yolo.postprocess_outputs(
                raw_outputs, 
                original_shape,
                conf_threshold=self.config.get('detection', {}).get('confidence_threshold', 0.5)
            )
            
            # Convert to Detection objects
            detections = []
            for det in detections_raw:
                detection = Detection(
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    class_id=det['class_id'],
                    class_name=det['class_name'],
                    frame_id=metadata.frame_id
                )
                detections.append(detection)
            
            postprocess_time = (time.time() - postprocess_start) * 1000
            self.performance_stats['postprocessing_times'].append(postprocess_time)
            
            # Tracking
            tracking_start = time.time()
            tracks = self.tracker.update(detections, frame)
            tracking_time = (time.time() - tracking_start) * 1000
            self.performance_stats['tracking_times'].append(tracking_time)
            
            # Log results
            if self.logger_system:
                processing_time = (time.time() - process_start) if process_start else 0
                frame_data = FrameData(
                    frame_id=metadata.frame_id,
                    timestamp=metadata.timestamp,
                    image=frame,
                    detections=detections,
                    tracks=tracks,
                    processing_time=processing_time
                )
                self.logger_system.log_frame_data(frame_data)
            
            total_process_time = (time.time() - process_start) * 1000
            self.performance_stats['total_frames'] += 1
            
            return {
                'frame': frame,
                'detections': detections,
                'tracks': tracks,
                'frame_id': metadata.frame_id,
                'timestamp': metadata.timestamp,
                'processing_time': total_process_time,
                'inference_time': inference_time,
                'tracking_time': tracking_time
            }
            
        except Exception as e:
            logger.error(f"❌ Frame processing error: {e}")
            return None
    
    def _display_thread(self):
        """High-performance display thread"""
        if not self.display_enabled:
            return
        
        fps_counter = []
        last_fps_update = time.time()
        
        while self.running:
            try:
                # Get processed result
                try:
                    result = self.result_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                frame = result['frame']
                detections = result['detections']
                tracks = result['tracks']
                processing_time = result['processing_time']
                
                # Calculate display FPS
                current_time = time.time()
                fps_counter.append(current_time)
                
                # Keep only last 30 frames for FPS calculation
                if len(fps_counter) > 30:
                    fps_counter.pop(0)
                
                # Calculate FPS every 0.5 seconds
                if current_time - last_fps_update > 0.5:
                    if len(fps_counter) >= 2:
                        time_span = fps_counter[-1] - fps_counter[0]
                        display_fps = (len(fps_counter) - 1) / time_span if time_span > 0 else 0.0
                    else:
                        display_fps = 0.0
                    last_fps_update = current_time
                else:
                    display_fps = 0.0
                
                # Draw annotations
                annotated_frame = self.logger_system.draw_annotations(
                    frame, detections, tracks,
                    show_detections=self.display_detections,
                    show_tracks=self.display_tracks,
                    show_fps=self.display_fps,
                    fps=display_fps
                )
                
                # Add performance overlay
                perf_text = f"Inference: {processing_time:.1f}ms | GPU: TensorRT FP16"
                cv2.putText(annotated_frame, perf_text, (10, annotated_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Display frame
                cv2.imshow('DeepStream Pipeline - High Performance', annotated_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("🛑 Quit requested by user")
                    self.running = False
                    break
                
                self.result_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Display thread error: {e}")
    
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up DeepStream pipeline...")
        
        try:
            if self.video_source:
                self.video_source.release()
            
            if self.logger_system:
                self.logger_system.cleanup()
            
            if self.executor:
                self.executor.shutdown(wait=True)
            
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def _print_performance_stats(self):
        """Print comprehensive performance statistics"""
        stats = self.performance_stats
        
        if stats['total_frames'] == 0:
            return
        
        avg_fps = stats['total_frames'] / stats['total_time'] if stats['total_time'] > 0 else 0
        
        # Calculate averages
        avg_inference = np.mean(stats['inference_times']) if stats['inference_times'] else 0
        avg_tracking = np.mean(stats['tracking_times']) if stats['tracking_times'] else 0
        avg_preprocess = np.mean(stats['preprocessing_times']) if stats['preprocessing_times'] else 0
        avg_postprocess = np.mean(stats['postprocessing_times']) if stats['postprocessing_times'] else 0
        
        # Get TensorRT stats
        tensorrt_stats = self.tensorrt_yolo.get_performance_stats() if self.tensorrt_yolo else {}
        
        logger.info("📊 DeepStream Pipeline Performance Summary:")
        logger.info(f"  🎯 Average FPS: {avg_fps:.2f}")
        logger.info(f"  📦 Total Frames: {stats['total_frames']}")
        logger.info(f"  ⏱️  Total Runtime: {stats['total_time']:.1f}s")
        logger.info(f"  🔥 TensorRT Inference: {avg_inference:.2f}ms")
        logger.info(f"  🎯 Tracking: {avg_tracking:.2f}ms")
        logger.info(f"  📋 Preprocessing: {avg_preprocess:.2f}ms")
        logger.info(f"  📋 Postprocessing: {avg_postprocess:.2f}ms")
        
        if tensorrt_stats:
            logger.info(f"  ⚡ TensorRT Theoretical FPS: {tensorrt_stats.get('fps_theoretical', 0):.1f}")
        
        logger.info("🚀 DeepStream pipeline completed successfully")


def create_deepstream_config(
    source_type: str = "camera",
    source_path: str = "",
    camera_id: int = 0,
    model: str = "yolov8n",
    tracker: str = "botsort",
    confidence: float = 0.5,
    output_dir: str = "outputs",
    display: bool = True
) -> Dict[str, Any]:
    """Create optimized DeepStream configuration"""
    
    return {
        'video_source': {
            'type': source_type,
            'path': source_path,
            'camera_id': camera_id
        },
        'detection': {
            'model': model,
            'confidence_threshold': confidence,
            'tensorrt_precision': 'fp16',  # 2x speed boost
            'max_batch_size': 1
        },
        'tracking': {
            'algorithm': tracker,
            'max_lost_frames': 60,
            'min_track_length': 5,
            # BOTSort optimized settings
            'botsort': {
                'track_high_thresh': 0.5,
                'track_low_thresh': 0.2,
                'new_track_thresh': 0.8,
                'match_thresh': 0.3,
                'motion_weight': 0.3,
                'iou_weight': 0.7
            },
            # NvDCF optimized settings
            'nvdcf': {
                'max_age': 60,
                'max_tracks': 4,
                'update_frequency': 3
            }
        },
        'display': {
            'enabled': display,
            'show_fps': True,
            'show_tracks': True,
            'show_detections': True
        },
        'logging': {
            'output_dir': output_dir,
            'save_video': True,
            'save_detections': True,
            'save_tracks': True,
            'save_metrics': True
        },
        'performance': {
            'gpu_memory_pool_size': 10,
            'max_queue_size': 30,
            'thread_workers': 4
        }
    } 