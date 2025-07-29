"""
TensorRT YOLO Engine Converter and Inference
High-performance GPU-optimized YOLO inference for DeepStream-style pipeline
"""

import os
import time
import numpy as np
import torch
import tensorrt as trt
from typing import List, Tuple, Optional, Dict, Any
from ultralytics import YOLO
import cv2
from loguru import logger


class TensorRTYOLO:
    """
    High-performance TensorRT YOLO engine for maximum GPU acceleration
    Replaces standard YOLO inference with optimized TensorRT engine
    """
    
    def __init__(self, model_path: str, engine_path: Optional[str] = None, 
                 precision: str = "fp16", max_batch_size: int = 1,
                 workspace_size: int = 1 << 30):  # 1GB
        """
        Initialize TensorRT YOLO engine
        
        Args:
            model_path: Path to YOLO model (.pt file)
            engine_path: Path to save/load TensorRT engine
            precision: Precision mode ('fp32', 'fp16', 'int8')
            max_batch_size: Maximum batch size for inference
            workspace_size: TensorRT workspace size in bytes
        """
        self.model_path = model_path
        self.engine_path = engine_path or model_path.replace('.pt', '.engine')
        self.precision = precision
        self.max_batch_size = max_batch_size
        self.workspace_size = workspace_size
        
        # TensorRT components with MAXIMUM logging for debugging
        self.logger = trt.Logger(trt.Logger.VERBOSE)
        self.engine = None
        self.context = None
        self.stream = None
        
        # PyTorch fallback for robustness
        self.pytorch_model = None
        self.use_pytorch_fallback = False
        
        # Model metadata
        self.input_shape = None
        self.output_shapes = None
        self.class_names = None
        self.device_inputs = []
        self.device_outputs = []
        self.host_outputs = []
        
        # Performance tracking
        self.inference_times = []
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize or build TensorRT engine with PyTorch fallback"""
        try:
            # Initialize PyTorch fallback first
            self._init_pytorch_fallback()
            
            if os.path.exists(self.engine_path):
                logger.info(f"🔄 Loading existing TensorRT engine: {self.engine_path}")
                self._load_engine()
            else:
                logger.info(f"🔨 Building new TensorRT engine from: {self.model_path}")
                self._build_engine()
                
            self._setup_inference()
            logger.info("✅ TensorRT engine ready for maximum performance!")
            
        except Exception as e:
            logger.error(f"❌ TensorRT initialization failed: {e}")
            logger.warning("🔄 Falling back to PyTorch inference for compatibility")
            self.use_pytorch_fallback = True
    
    def _build_engine(self):
        """Build TensorRT engine from YOLO model"""
        logger.info("🚀 Building TensorRT engine - this may take a few minutes...")
        
        # Load YOLO model to get metadata
        yolo_model = YOLO(self.model_path)
        self.class_names = yolo_model.names
        
        # Export to ONNX first (required for TensorRT)
        onnx_path = self.model_path.replace('.pt', '.onnx')
        logger.info(f"📦 Exporting YOLO to ONNX: {onnx_path}")
        
        try:
            yolo_model.export(
                format='onnx',
                imgsz=640,  # Standard YOLO input size
                dynamic=False,  # Fixed input shape for better optimization
                simplify=True,  # Simplify ONNX model
                opset=17  # ONNX opset version
            )
        except Exception as e:
            logger.error(f"❌ ONNX export failed: {e}")
            raise
        
        # Build TensorRT engine from ONNX
        builder = trt.Builder(self.logger)
        config = builder.create_builder_config()
        
        # Set precision mode
        if self.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("🎯 Using FP16 precision for 2x speed boost")
        elif self.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("⚡ Using INT8 precision for maximum speed")
        
        # Set workspace size (TensorRT 10.x API)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_size)
        
        # Create network with explicit batch and strong typing
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX model
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                logger.error("❌ Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")
        
        # Build engine
        logger.info("⚙️ Building TensorRT engine (this may take 5-10 minutes)...")
        build_start = time.time()
        
        # TensorRT 10.x API - use build_serialized_network
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("❌ Failed to build TensorRT engine")
        
        build_time = time.time() - build_start
        logger.info(f"✅ TensorRT engine built in {build_time:.1f} seconds")
        
        # Save serialized engine
        with open(self.engine_path, 'wb') as f:
            f.write(serialized_engine)
        logger.info(f"💾 Engine saved to: {self.engine_path}")
        
        # Load the engine from the serialized data
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        
        # Clean up ONNX file
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
    
    def _load_engine(self):
        """Load existing TensorRT engine"""
        runtime = trt.Runtime(self.logger)
        
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"❌ Failed to load TensorRT engine from {self.engine_path}")
        
        logger.info("✅ TensorRT engine loaded successfully")
    
    def _setup_inference(self):
        """Setup inference context and memory allocation"""
        self.context = self.engine.create_execution_context()
        
        # Create proper tensor mappings
        self.tensor_names = []
        self.tensor_device_memory = {}
        self.tensor_host_memory = {}
        
        # Get input/output shapes and create proper mappings
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            
            self.tensor_names.append(name)
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_shape = shape
                self.input_tensor_name = name
                logger.info(f"📥 Input tensor: {name} {shape} {dtype}")
            else:
                if self.output_shapes is None:
                    self.output_shapes = []
                self.output_shapes.append(shape)
                logger.info(f"📤 Output tensor: {name} {shape} {dtype}")
        
        # Allocate GPU memory with explicit CUDA context management
        import pycuda.driver as cuda
        import pycuda.autoinit  # Initialize CUDA context
        
        # Ensure CUDA context is current
        cuda.Context.get_current().push()
        
        try:
            self.stream = cuda.Stream()
        except Exception as e:
            logger.error(f"❌ CUDA stream creation failed: {e}")
            cuda.Context.get_current().pop()
            raise
        
        # Allocate device memory for inputs and outputs with proper mapping
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            
            # Allocate device memory
            device_mem = cuda.mem_alloc(size)
            self.tensor_device_memory[name] = device_mem
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                # Keep legacy list for compatibility
                self.device_inputs.append(device_mem)
            else:
                # Allocate host memory for outputs
                host_mem = np.empty(shape, dtype=dtype)
                self.tensor_host_memory[name] = host_mem
                # Keep legacy lists for compatibility
                self.device_outputs.append(device_mem)
                self.host_outputs.append(host_mem)
        
        logger.info("🔥 TensorRT inference engine ready for maximum performance!")
    
    def _init_pytorch_fallback(self):
        """Initialize PyTorch model as fallback"""
        try:
            self.pytorch_model = YOLO(self.model_path)
            self.pytorch_model.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Set class names from PyTorch model (critical for postprocessing)
            if self.class_names is None:
                self.class_names = self.pytorch_model.names
                logger.info(f"🏷️ Loaded {len(self.class_names)} class names from PyTorch model")
            
            logger.info("🐍 PyTorch fallback model initialized")
        except Exception as e:
            logger.warning(f"⚠️ PyTorch fallback initialization failed: {e}")
    
    def _pytorch_inference(self, frame: np.ndarray) -> List[np.ndarray]:
        """PyTorch fallback inference - returns TensorRT-compatible format"""
        try:
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run YOLO inference
            results = self.pytorch_model(frame_rgb, verbose=False)
            
            # Convert results to TensorRT-compatible format
            # Expected: [batch, num_detections, 85] where 85 = x_center,y_center,w,h + conf + 80_classes
            outputs = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    # Get detections
                    boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # [N, 4] x1,y1,x2,y2
                    confidences = result.boxes.conf.cpu().numpy()  # [N]
                    classes = result.boxes.cls.cpu().numpy()  # [N]
                    
                    num_detections = len(boxes_xyxy)
                    
                    # Convert xyxy to center format (x_center, y_center, width, height)
                    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Create output array with TensorRT format: [x_center, y_center, w, h, conf, class0, class1, ..., class79]
                    output = np.zeros((num_detections, 85), dtype=np.float32)
                    
                    # Fill in coordinates and confidence
                    output[:, 0] = x_center
                    output[:, 1] = y_center
                    output[:, 2] = width
                    output[:, 3] = height
                    output[:, 4] = confidences
                    
                    # Fill in class probabilities (one-hot encoded)
                    for i, cls_id in enumerate(classes):
                        output[i, 5 + int(cls_id)] = 1.0  # Set class probability to 1.0
                    
                    # Add batch dimension: [1, num_detections, 85]
                    output_batch = np.expand_dims(output, axis=0)
                    outputs.append(output_batch)
                else:
                    # No detections - return empty with correct shape
                    empty_output = np.zeros((1, 0, 85), dtype=np.float32)
                    outputs.append(empty_output)
            
            return outputs
            
        except Exception as e:
            logger.error(f"❌ PyTorch fallback inference failed: {e}")
            # Return empty output with correct TensorRT format
            return [np.zeros((1, 0, 85), dtype=np.float32)]
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for YOLO inference
        Optimized for speed with minimal memory copies
        """
        # Resize and normalize (optimized)
        input_h, input_w = self.input_shape[2], self.input_shape[3]
        
        # Resize frame
        frame_resized = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB and normalize
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_transposed, axis=0)
        
        return np.ascontiguousarray(frame_batch)
    
    def inference(self, preprocessed_frame: np.ndarray = None, original_frame: np.ndarray = None) -> List[np.ndarray]:
        """
        High-speed TensorRT inference with automatic PyTorch fallback
        Returns raw model outputs for post-processing
        """
        # Use PyTorch fallback if TensorRT failed or if explicitly requested
        if self.use_pytorch_fallback:
            if original_frame is not None:
                return self._pytorch_inference(original_frame)
            else:
                logger.error("❌ PyTorch fallback requires original frame")
                return [np.empty((0, 6))]
        
        # Try TensorRT inference
        try:
            return self._tensorrt_inference(preprocessed_frame)
        except Exception as e:
            logger.error(f"❌ TensorRT inference failed: {e}")
            logger.warning("🔄 Switching to PyTorch fallback")
            self.use_pytorch_fallback = True
            
            if original_frame is not None:
                return self._pytorch_inference(original_frame)
            else:
                logger.error("❌ PyTorch fallback requires original frame")
                return [np.empty((0, 6))]
    
    def _tensorrt_inference(self, preprocessed_frame: np.ndarray) -> List[np.ndarray]:
        """
        TensorRT inference implementation
        """
        import pycuda.driver as cuda
        
        inference_start = time.time()
        
        # Validate input data format and shape
        expected_shape = tuple(self.input_shape)
        logger.debug(f"🔍 Input tensor debug:")
        logger.debug(f"   📐 Expected shape: {expected_shape}")
        logger.debug(f"   📐 Actual shape: {preprocessed_frame.shape}")
        logger.debug(f"   🔢 Data type: {preprocessed_frame.dtype}")
        logger.debug(f"   💾 Contiguous: {preprocessed_frame.flags['C_CONTIGUOUS']}")
        logger.debug(f"   🎯 Input tensor name: {self.input_tensor_name}")
        
        if preprocessed_frame.shape != expected_shape:
            raise RuntimeError(f"❌ Input shape mismatch: got {preprocessed_frame.shape}, expected {expected_shape}")
        
        # Ensure input is contiguous and correct dtype
        if not preprocessed_frame.flags['C_CONTIGUOUS']:
            preprocessed_frame = np.ascontiguousarray(preprocessed_frame)
            logger.debug("🔧 Made input tensor contiguous")
        
        if preprocessed_frame.dtype != np.float32:
            preprocessed_frame = preprocessed_frame.astype(np.float32)
            logger.debug("🔧 Converted input to float32")
        
        # Copy input to GPU using proper tensor mapping
        input_device_mem = self.tensor_device_memory[self.input_tensor_name]
        cuda.memcpy_htod_async(input_device_mem, preprocessed_frame, self.stream)
        
        # Set tensor addresses using proper name mapping
        for name in self.tensor_names:
            device_mem = self.tensor_device_memory[name]
            self.context.set_tensor_address(name, int(device_mem))
        
        # Synchronize before execution to ensure all data is ready
        self.stream.synchronize()
        
        # Execute inference with error checking
        success = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not success:
            raise RuntimeError("❌ TensorRT inference execution failed")
        
        # Copy outputs from GPU using proper tensor mapping
        outputs = []
        for name in self.tensor_names:
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                device_mem = self.tensor_device_memory[name]
                host_mem = self.tensor_host_memory[name]
                cuda.memcpy_dtoh_async(host_mem, device_mem, self.stream)
                outputs.append(host_mem.copy())
        
        # Synchronize stream
        self.stream.synchronize()
        
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms
        self.inference_times.append(inference_time)
        
        return outputs
    
    def postprocess_outputs(self, outputs: List[np.ndarray], 
                          original_shape: Tuple[int, int], 
                          conf_threshold: float = 0.5,
                          iou_threshold: float = 0.45) -> List[Dict[str, Any]]:
        """
        Post-process TensorRT outputs to get detections
        Optimized for speed
        """
        detections = []
        
        # Assuming YOLO output format: [batch, num_detections, 85] 
        # where 85 = x,y,w,h + confidence + 80 classes
        output = outputs[0][0]  # Remove batch dimension
        
        # Filter by confidence
        valid_detections = output[output[:, 4] > conf_threshold]
        
        if len(valid_detections) == 0:
            return detections
        
        # Extract bounding boxes and scores
        boxes = valid_detections[:, :4]
        scores = valid_detections[:, 4]
        class_scores = valid_detections[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        class_confidences = np.max(class_scores, axis=1)
        
        # Final confidence = objectness * class confidence
        final_scores = scores * class_confidences
        
        # Convert from center format to corner format
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Scale to original image size
        orig_h, orig_w = original_shape
        input_h, input_w = self.input_shape[2], self.input_shape[3]
        
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        
        x1 = (x1 * scale_x).astype(int)
        y1 = (y1 * scale_y).astype(int)
        x2 = (x2 * scale_x).astype(int)
        y2 = (y2 * scale_y).astype(int)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), final_scores.tolist(), 
            conf_threshold, iou_threshold
        )
        
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'bbox': (int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                    'confidence': float(final_scores[i]),
                    'class_id': int(class_ids[i]),
                    'class_name': (self.class_names or {}).get(int(class_ids[i]), f'class_{int(class_ids[i])}')
                })
        
        return detections
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time_ms': np.mean(self.inference_times),
            'min_inference_time_ms': np.min(self.inference_times),
            'max_inference_time_ms': np.max(self.inference_times),
            'fps_theoretical': 1000.0 / np.mean(self.inference_times),
            'total_inferences': len(self.inference_times)
        }
    
    def __del__(self):
        """Cleanup GPU memory"""
        if hasattr(self, 'device_inputs'):
            for device_input in self.device_inputs:
                try:
                    device_input.free()
                except:
                    pass
        
        if hasattr(self, 'device_outputs'):
            for device_output in self.device_outputs:
                try:
                    device_output.free()
                except:
                    pass 