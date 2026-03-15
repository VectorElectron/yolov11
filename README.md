# YOLOv11 Vectron

A lightweight, operator-based object detection engine designed for cross-platform and cross-language deployment, based on the YOLOv11 model.

## Key Features

### 🚀 Full Operator-based Pipeline
YOLOv11 Vectron implements the entire object detection process through a series of specialized operators, each optimized for their specific task:

- **Resize Operator**: Efficiently resizes images for optimal processing
- **Detection Operator**: Performs object detection using YOLOv11 model
- **NMS (Non-Maximum Suppression) Operator**: Filters redundant detections
- **Compose Operator**: Integrates all operations into a unified pipeline

This modular design allows for easy customization and optimization of each component.

### 🔄 Cross-Language & Edge Support
The ONNX-based architecture enables versatile deployment:

- **Multiple Languages**: Python (native support), C++, C#, Java, JavaScript, and more via ONNX runtime bindings
- **Edge Acceleration**: Supports TensorRT for NVIDIA devices and other edge acceleration frameworks
- **Cross-Platform**: Runs seamlessly on Windows, Linux, macOS, mobile devices, and edge devices

### 📦 Lightweight Design
- Minimal dependencies (only ONNX runtime and NumPy)
- Compact model files
- Efficient inference with ONNX runtime optimization

## Installation

### From PyPI
```bash
pip install yolov11-vectron
```

### From Source
```bash
pip install -e .
```

### Package Location
After pip installation, the package is located at `site-packages/yolov11_vectron/`.

### Model Files
The required model files are included in the package. Key files:
- `yolo11n_one.onnx` (combined model with preprocessing and postprocessing)
- `yolov11-dic80.txt` (class dictionary for COCO dataset)

## Python Usage

After installing the package, you can use YOLOv11 Vectron as follows:

```python
import numpy as np
from imageio.v2 import imread
from yolov11_vectron import detect

# Load an image
image = imread('path/to/your/image.png')

# Run object detection with default parameters
results = detect(image)

# Print results
print("Detection Results:")
for box, label, confidence in results:
    print(f"Label: {label}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Bounding Box: {box}")
    print("-")

# Advanced usage with custom parameters
custom_results = detect(
    image,
    dial=640*1.414,      # Maximum dimension for resizing
    conf=0.25,            # Confidence threshold
    iou=0.25,             # IoU threshold for NMS
    topk=1024             # Maximum number of detections
)
```

### Testing
You can run the built-in test to verify the installation:

```python
from yolov11_vectron import test
test()
```

This will load a test image, run detection, and display the results with bounding boxes and labels.

![Python Example](https://github.com/user-attachments/assets/7b63d4ed-13d4-49d9-b6f6-1b9c3fa2e5ba)

## JavaScript Usage

YOLOv11 Vectron supports pure frontend deployment using ONNX Runtime Web. For a complete implementation, please refer to `yolov11_vectron/model/index.html`.

### Live Demo
Try the complete mobile-compatible pure frontend object detection experience at:
[https://vectorelectron.github.io/release/yolov11/index.html](https://vectorelectron.github.io/release/yolov11/index.html)

### Quick Start

```html
<!DOCTYPE html>
<html>
<head>
    <title>YOLOv11 Vectron JS Example</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.0/dist/ort.min.js"></script>
</head>
<body>
    <input type="file" id="imageInput" accept="image/*">
    <button onclick="runDetection()">Run Detection</button>
    <div id="results"></div>

    <script>
        let session;
        let classNames = [];

        // Initialize model (simplified)
        async function init() {
            const basePath = './model/'; // Path to model files
            
            // Load class names
            const dictResp = await fetch(basePath + 'yolov11-dic80.txt');
            const text = await dictResp.text();
            classNames = text.split('\n');

            // Load model
            const opt = { executionProviders: ['wasm'] };
            session = await ort.InferenceSession.create(basePath + 'yolo11n_one.onnx', opt);

            console.log('Model loaded successfully');
        }

        // Run detection (simplified)
        async function runDetection() {
            // Full implementation available in model/index.html
            console.log('Running detection...');
            // Implementation details omitted - refer to index.html
        }

        // Initialize on page load
        window.onload = init;
    </script>
</body>
</html>
```

For the complete implementation with image processing and UI components, please refer to the full code in `yolov11_vectron/model/index.html`.

![Web Example](https://github.com/user-attachments/assets/8b50e9af-103a-4e79-9de4-95e5ebcbc759)

## License
BSD 3-Clause License