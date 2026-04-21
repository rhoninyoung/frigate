# Frigate NVR — Architecture Analysis

> Generated analysis covering two key dimensions:
> 1. **Multi-platform compatibility** — how the underlying layer achieves hardware abstraction
> 2. **Algorithm flow** — how the detection pipeline achieves high performance

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Multi-Platform Compatibility & Interface Abstraction](#2-multi-platform-compatibility--interface-abstraction)
   - 2.1 [Detector Plugin System](#21-detector-plugin-system)
   - 2.2 [Abstract Detection Interface (`DetectionApi`)](#22-abstract-detection-interface-detectionapi)
   - 2.3 [Model Runner Abstraction (`BaseModelRunner`)](#23-model-runner-abstraction-basemodelrunner)
   - 2.4 [Supported Hardware Backends](#24-supported-hardware-backends)
   - 2.5 [Model Format & Tensor Normalization](#25-model-format--tensor-normalization)
   - 2.6 [Dynamic Plugin Loading](#26-dynamic-plugin-loading)
3. [Algorithm Flow & High-Performance Design](#3-algorithm-flow--high-performance-design)
   - 3.1 [Process Topology](#31-process-topology)
   - 3.2 [Frame Capture (FFmpeg → Shared Memory)](#32-frame-capture-ffmpeg--shared-memory)
   - 3.3 [Motion Detection (Pre-filter)](#33-motion-detection-pre-filter)
   - 3.4 [Region-Based Detection Strategy](#34-region-based-detection-strategy)
   - 3.5 [Detector Process & Shared-Memory IPC](#35-detector-process--shared-memory-ipc)
   - 3.6 [Object Tracking (Norfair + Kalman Filter)](#36-object-tracking-norfair--kalman-filter)
   - 3.7 [Post-Processing & Downstream Pipeline](#37-post-processing--downstream-pipeline)
   - 3.8 [CUDA Graph Acceleration](#38-cuda-graph-acceleration)
   - 3.9 [Asynchronous Detector Path](#39-asynchronous-detector-path)
4. [Summary: Key Performance Decisions](#4-summary-key-performance-decisions)

---

## 1. System Overview

Frigate is a fully local Network Video Recorder designed for minimal resource usage and maximum inference throughput. It is built as a **Python multiprocessing application** where each major concern (capture, detection, tracking, recording, embedding, etc.) lives in a dedicated OS process or thread. Communication between processes is handled by a mix of **multiprocessing `Queue`s**, **POSIX shared memory**, and **ZeroMQ (ZMQ)** publish/subscribe sockets.

```
┌────────────────────────────────────────────────────────────┐
│                       FrigateApp (main)                    │
│  ┌───────────┐  ┌─────────────┐  ┌────────────────────┐   │
│  │  FastAPI  │  │  Dispatcher │  │  StatsEmitter      │   │
│  │  (uvicorn)│  │ MQTT/WS/Push│  │  (MQTT/WebSocket)  │   │
│  └───────────┘  └─────────────┘  └────────────────────┘   │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              CameraMaintainer (thread)               │  │
│  │  ┌──────────────┐   ┌──────────────────────────────┐│  │
│  │  │ FFmpegCapture │   │     CameraTracker (process)  ││  │
│  │  │   (thread)    │   │  motion → region → detect    ││  │
│  │  └──────────────┘   └──────────────────────────────┘│  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │       ObjectDetectProcess (one per detector)         │  │
│  │   DetectorRunner / AsyncDetectorRunner               │  │
│  │   └─► LocalObjectDetector ──► DetectionApi plugin   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ RecordProc  │  │ EmbedProc    │  │ ReviewProcess    │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

---

## 2. Multi-Platform Compatibility & Interface Abstraction

### 2.1 Detector Plugin System

All hardware-specific inference code lives under `frigate/detectors/plugins/`. Each file implements a single hardware backend and is auto-discovered at startup using **`pkgutil.iter_modules`**:

```python
# frigate/detectors/detector_types.py
_included_modules = pkgutil.iter_modules(plugins.__path__, plugins.__name__ + ".")

for _, name, _ in _included_modules:
    plugin_modules.append(importlib.import_module(name))

api_types = {det.type_key: det for det in DetectionApi.__subclasses__()}
```

The resulting `api_types` dictionary maps string keys (e.g. `"cpu"`, `"edgetpu"`, `"onnx"`) to their concrete `DetectionApi` subclass. User configuration selects the backend by name; the framework instantiates the correct class automatically via a Pydantic discriminated union (`DetectorConfig`).

### 2.2 Abstract Detection Interface (`DetectionApi`)

Every backend inherits from the **abstract base class** `DetectionApi` (`frigate/detectors/detection_api.py`):

```python
class DetectionApi(ABC):
    type_key: str                        # unique string key ("cpu", "edgetpu", …)
    supported_models: List[ModelTypeEnum]

    @abstractmethod
    def __init__(self, detector_config: BaseDetectorConfig): ...

    @abstractmethod
    def detect_raw(self, tensor_input) -> np.ndarray: ...
```

The contract is minimal: receive a pre-processed NumPy tensor, return a `(20, 6)` float32 array where each row is `[class_id, confidence, y_min, x_min, y_max, x_max]`. Everything above this interface (motion gating, region cropping, threshold filtering, tracking) is hardware-agnostic.

### 2.3 Model Runner Abstraction (`BaseModelRunner`)

For backends that share the ONNX Runtime or OpenVINO runtime, a second level of abstraction exists in `frigate/detectors/detection_runners.py`:

```
BaseModelRunner (ABC)
├── ONNXModelRunner        — plain ORT InferenceSession
├── CudaGraphRunner        — ORT IOBinding + CUDA Graph capture/replay
├── MIGraphXModelRunner    — AMD ROCm / MIGraphX via ORT
└── OpenVINOModelRunner    — Intel OpenVINO Core
```

Each runner exposes `get_input_names()`, `get_input_width()`, and `run(input_dict)`. The `ONNXDetector` plugin selects the right runner at construction time through `get_optimized_runner()`, which probes available hardware:

```python
def get_optimized_runner(model_path, device, model_type=None):
    if cuda_available:
        return CudaGraphRunner(...)   # NVIDIA GPU
    elif migraphx_available:
        return MIGraphXModelRunner(...)  # AMD GPU
    elif openvino_available and device in ("AUTO", "GPU", "NPU"):
        return OpenVINOModelRunner(...)  # Intel GPU/NPU
    else:
        return ONNXModelRunner(...)   # CPU fallback
```

### 2.4 Supported Hardware Backends

| Plugin file | `type_key` | Hardware target |
|---|---|---|
| `cpu_tfl.py` | `cpu` | Generic x86/ARM CPU via TFLite |
| `edgetpu_tfl.py` | `edgetpu` | Google Coral USB/PCIe EdgeTPU |
| `tensorrt.py` | `tensorrt` | NVIDIA Jetson / desktop GPU (TensorRT engine) |
| `onnx.py` | `onnx` | Generic ONNX (auto-selects CUDA/ROCm/OpenVINO/CPU) |
| `openvino.py` | `openvino` | Intel CPU/GPU/VPU/NPU via OpenVINO |
| `hailo8l.py` | `hailo8l` | Hailo-8 / Hailo-8L AI accelerator |
| `rknn.py` | `rknn` | Rockchip NPU (RK3588, etc.) |
| `axengine.py` | `axengine` | Axera AX650N NPU |
| `synaptics.py` | `synaptics` | Synaptics Astra SL1680 NPU |
| `memryx.py` | `memryx` | MemryX MX3 AI accelerator |
| `deepstack.py` | `deepstack` | Remote DeepStack HTTP API |
| `degirum.py` | `degirum` | DeGirum ORCA accelerator |
| `zmq_ipc.py` | `zmq_ipc` | Any external process via ZMQ IPC |

### 2.5 Model Format & Tensor Normalization

Hardware diversity also means different model input conventions. `ModelConfig` (`frigate/detectors/detector_config.py`) centralizes all such metadata:

| Field | Purpose |
|---|---|
| `input_tensor` | Layout: `nhwc` / `nchw` / `hwnc` / `hwcn` |
| `input_pixel_format` | Color space: `rgb` / `bgr` / `yuv` |
| `input_dtype` | Data type: `int` (uint8), `float` (normalized), `float_denorm` |
| `model_type` | Architecture: `ssd`, `yolox`, `yolonas`, `yolo-generic`, `rfdetr`, `dfine` |

`BaseLocalDetector._transform_input()` applies the required transpose and dtype conversion before handing the tensor to `detect_api.detect_raw()`. Each post-processing helper (`post_process_yolo`, `post_process_yolox`, `post_process_rfdetr`, `post_process_dfine`) normalises model-specific output formats into the common `(20, 6)` detection array.

### 2.6 Dynamic Plugin Loading

Because some accelerator SDKs (OpenVINO on ARM, Hailo SDK) may not be installed, plugins are imported inside a `try/except ImportError` block. A failed import is logged as an error but does not crash the application; the plugin simply does not appear in `api_types`.

---

## 3. Algorithm Flow & High-Performance Design

### 3.1 Process Topology

```
FFmpegCapture thread
       │  (YUV420 bytes → Shared Memory)
       ▼
 frame_queue (mp.Queue)
       │
       ▼
CameraTracker process  (one per camera)
  ├─ ImprovedMotionDetector
  ├─ Region selector
  ├─ RemoteObjectDetector  ─── detection_queue (mp.Queue) ──►  DetectorRunner process
  │       ◄────────────────── ZMQ notification + SHM output ──────────────────────┘
  ├─ NorfairTracker
  └─ detected_objects_queue ──► TrackedObjectProcessor thread
                                        │
                              ┌─────────┴──────────┐
                              │                    │
                         EventProcessor      OutputProcess
                         TimelineProcessor   RecordProcess
                         EmbeddingProcess    ReviewProcess
```

### 3.2 Frame Capture (FFmpeg → Shared Memory)

`FFmpegCapture` (inside `frigate/video/ffmpeg.py`) spawns an **ffmpeg subprocess** that decodes the camera stream and pipes raw **YUV420** frames to stdout. A dedicated reader thread calls `capture_frames()` which:

1. Reads exactly `frame_size` bytes from ffmpeg's stdout pipe.
2. Writes the bytes directly into a **POSIX shared memory segment** (`SharedMemoryFrameManager`) whose name is the camera name. This eliminates copying frames across processes.
3. Pushes `(frame_name, frame_time)` onto `frame_queue`.

Frame drops are handled with adaptive back-pressure: if `frame_queue` is full, the oldest frame is discarded (`skipped_fps` counter increments). Only the **most recent frame** is processed, keeping latency low at the cost of some temporal redundancy.

### 3.3 Motion Detection (Pre-filter)

`ImprovedMotionDetector` (`frigate/motion/improved_motion.py`) is a lightweight CPU-only stage that gates expensive inference:

1. **Resize** the YUV luma plane (Y channel) down to a configurable `frame_height` (default ≈ 50–100 px tall) to dramatically reduce pixel count.
2. **Running average** (`cv2.accumulateWeighted`) maintains a background model.
3. **Absolute difference** + Gaussian blur + threshold to isolate pixels that changed.
4. **OpenCV contour detection** groups changed pixels into bounding boxes.
5. The result is a list of *motion boxes* — regions of the frame where movement was detected.

During a **calibration** phase (first N frames) the motion detector returns a full-frame box so that initial object positions are established before the background model stabilises. The detector also integrates with the PTZ auto-tracker: if the pan/tilt motor is moving, it immediately returns a large box to trigger a full re-scan rather than trying to detect motion against a moving background.

### 3.4 Region-Based Detection Strategy

Running inference on the full (e.g. 1920×1080) frame every tick would be prohibitively expensive. Frigate instead crops small **detection regions** and resizes each one to the model input size (default 320×320). The region selection logic in `process_frames()` (`frigate/video/detect.py`) balances completeness against efficiency:

1. **Startup scan**: on first frame, a set of overlapping grid regions covers the whole frame to bootstrap the tracker.
2. **Tracked-object regions**: a cluster region is grown around each non-stationary tracked object's bounding box. Objects that have been motionless longer than a threshold are *stationary* and are re-checked only every `stationary.interval` frames.
3. **Motion regions**: standalone motion boxes (those not already inside a tracked-object region) are clustered and expanded to the nearest grid cell via `get_cluster_region_from_grid()`. A pre-built **region grid** (computed daily) ensures regions align with sensible crop boundaries.
4. Regions that are too similar are merged (`reduce_detections`) to avoid duplicate inference calls.

The net result: on a quiet scene with one moving person, Frigate may only run the detector on **1–2 small crops** per frame rather than the full image.

### 3.5 Detector Process & Shared-Memory IPC

The separation of the camera tracking process from the detector process is deliberate: one detector process can serve **multiple cameras simultaneously** (a multiprocessing `Queue` serialises requests).

The IPC protocol is zero-copy where possible:

```
CameraTracker (RemoteObjectDetector.detect)
  1. Write tensor into pre-allocated SHM (name = camera name)
  2. Put camera name onto detection_queue (mp.Queue)
  3. Block on ZMQ SUB socket waiting for notification

DetectorRunner
  1. Pop camera name from detection_queue
  2. Map the SHM read-only via np.ndarray (zero copy)
  3. Run detect_raw(input_frame)
  4. Write (20,6) float32 result array into output SHM (out-<name>)
  5. Publish camera name on ZMQ PUB socket

CameraTracker
  6. ZMQ notification received → read output SHM → parse detections
```

The output SHM is a fixed `20 × 6 × 4 = 480 bytes` — small enough to avoid cache pressure while holding up to 20 detections.

### 3.6 Object Tracking (Norfair + Kalman Filter)

`NorfairTracker` (`frigate/track/norfair_tracker.py`) wraps the [Norfair](https://github.com/tryolabs/norfair) multi-object tracking library which uses a **Kalman filter** to predict object positions between detections.

Key design choices:

- **Per-class trackers**: separate `Tracker` instances per object class prevent cross-class ID confusion (e.g. a "car" and a "person" overlapping).
- **Custom distance function**: instead of Euclidean centre distance, Frigate uses a normalised distance that accounts for bounding-box width, height, and position relative to the object's own dimensions. This prevents large objects from "stealing" detections meant for nearby small objects.
- **Stationary classification**: `StationaryMotionClassifier` tracks a rolling history of box positions; once variance drops below threshold the object is promoted to *stationary* and detection frequency is reduced.
- **IoU-based track merging**: duplicate tracks are merged using intersection-over-union.

### 3.7 Post-Processing & Downstream Pipeline

After tracking, `CameraTracker` puts a snapshot of all tracked objects onto `detected_objects_queue`. `TrackedObjectProcessor` (a thread in the main process) performs:

- **Event lifecycle management**: creates/updates/ends `Event` database rows (via Peewee ORM → SQLite).
- **Thumbnail selection**: `is_better_thumbnail()` selects the frame where the object is largest, highest-scoring, and not cut off at frame edges.
- **Snapshot & clip recording**: communicates with `RecordProcess` via ZMQ.
- **MQTT / WebSocket dispatch**: pushed to `Dispatcher` → `MqttClient`, `WebSocketClient`, `WebPushClient`.
- **Embeddings**: triggers `EmbeddingProcess` for semantic search, face recognition, LPR, and GenAI description generation.
- **Timeline**: event milestones appended to the `Timeline` table for the review interface.

### 3.8 CUDA Graph Acceleration

When an NVIDIA GPU is detected, the `ONNXDetector` uses `CudaGraphRunner` instead of the plain `ONNXModelRunner`. On the **first inference**:

1. ORT `IOBinding` pins input/output tensors to CUDA device memory.
2. The CUDA Graph is *captured* — all GPU kernel launches are recorded.

On every subsequent call the graph is simply **replayed** with updated input data. This eliminates CPU–GPU kernel-launch overhead and CUDA API call latency, giving a significant throughput improvement for small, repeated inference workloads (which is exactly Frigate's workload pattern of many small 320×320 crops).

A warm-up inference is run at detector startup to trigger any JIT compilation (PTX on newer NVIDIA architectures, MIGraphX graph compilation on AMD) before the watchdog timer starts.

### 3.9 Asynchronous Detector Path

For accelerators like MemryX that support pipelined inference, Frigate uses `AsyncDetectorRunner` instead of `DetectorRunner`. The async path splits inference into two threads:

- **`_detect_worker`**: pulls frames from the queue and calls `async_send_input()` — returns immediately without blocking.
- **`_result_worker`**: blocks on `async_receive_output()` from the accelerator and writes results to output SHM.

This allows the accelerator's internal pipeline to stay full while the host CPU is already preparing the next frame, maximising hardware utilisation.

---

## 4. Summary: Key Performance Decisions

| Technique | Where | Benefit |
|---|---|---|
| **Shared memory frame passing** | Capture → Tracker → Detector | Eliminates inter-process frame copies |
| **Motion pre-filter** | `ImprovedMotionDetector` | Skips inference on frames with no activity |
| **Region-based cropping** | `process_frames` | Runs inference on small crops (≤320×320) not full frames |
| **Stationary object suppression** | `NorfairTracker` | Reduces inference frequency for non-moving objects |
| **Dedicated detector process** | `ObjectDetectProcess` | Isolates GPU/accelerator memory from camera processes |
| **ZMQ pub/sub notification** | `ObjectDetectorPublisher/Subscriber` | Low-latency, non-polling result delivery |
| **CUDA Graph replay** | `CudaGraphRunner` | Near-zero GPU kernel launch overhead |
| **Async detector pipeline** | `AsyncDetectorRunner` | Saturates accelerator pipeline (e.g. MemryX) |
| **Per-class Kalman tracking** | `NorfairTracker` | Avoids full detection every frame; predicts positions |
| **Plugin auto-discovery** | `detector_types.py` | Zero configuration change needed when a new backend is added |
