# MNIST CNN Inference API

[![Deployed](https://img.shields.io/badge/deploy-Railway-blueviolet)](https://railway.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688)](https://fastapi.tiangolo.com/)
[![TorchScript](https://img.shields.io/badge/TorchScript-2.0+-EE4C2C)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

> **Production-ready inference API serving TorchScript CNN model for handwritten digit recognition**

- [Frontend Demo](https://shafwanadhi.github.io/mnist-handwriting-visualizer/)
- [API docs](https://mnist-cnn-torchscript-inference-api-production.up.railway.app/docs)
- [API deployment](https://mnist-cnn-torchscript-inference-api-production.up.railway.app/)

---

## Overview

This repository contains the **backend inference service** for the MNIST CNN Visualizer project. It serves a TorchScript-optimized convolutional neural network through a FastAPI REST interface, designed for:

- **Feature map extraction** for educational visualization
- **Horizontal scalability** via stateless architecture
- **Production deployment** on Railway with health monitoring

**Related Repository:** [mnist-visualizer](https://github.com/ShafwanAdhi/mnist-handwriting-visualizer.git) (Frontend application)

---

## System Architecture

```
┌──────────────────────────────────────┐
│         FastAPI Application          │
│  ┌────────────────────────────────┐  │
│  │   CORS Middleware              │  │
│  │   (Allow cross-origin calls)   │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │   TorchScript Model Loader     │  │
│  │   • Loaded once at startup     │  │
│  │   • CPU-optimized inference    │  │
│  │   • No GPU required            │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │   /predict Endpoint            │  │
│  │   • Input: 28×28 pixel array   │  │
│  │   • Output: prediction + maps  │  │
│  │   • Latency: ~5-15ms           │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│      Railway Deployment Platform     │
│  • Auto-scaling workers              │
│  • Health check monitoring           │
│  • Zero-downtime deployments         │
└──────────────────────────────────────┘
```

---

## Design Decisions

### Why TorchScript?
- **Framework-agnostic deployment**: No Python GIL bottlenecks
- **Optimized inference**: Graph optimizations and operator fusion
- **Smaller footprint**: Serialized model without training dependencies
- **Production-ready**: Used by Meta, Tesla, and other ML teams at scale

### Why FastAPI?
- **Async by default**: Non-blocking I/O for concurrent requests
- **Automatic API docs**: OpenAPI/Swagger out of the box
- **Type safety**: Pydantic models prevent runtime errors

### Why Stateless Architecture?
- **Horizontal scalability**: Add instances without shared state
- **No session management**: Each request is independent
- **Cloud-native**: Perfect fit for container orchestration
- **Cost-effective**: Pay only for compute time used

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | FastAPI 0.104+ | Async REST API |
| **ML Runtime** | TorchScript 2.0+ | Model inference |
| **Validation** | Pydantic 2.0+ | Request/response schemas |
| **Server** | Uvicorn | ASGI web server |
| **Deployment** | Railway | Cloud platform |
| **Language** | Python 3.9+ | Core implementation |

---

## Model Architecture

```
Input: 28×28 grayscale image (normalized)
  │
  ├─► Conv1: 1→10 filters (3×3, padding=1) + ReLU ──► Feature Maps 1-10
  │                                                    (28×28×10)
  │
  ├─► Conv2: 10→10 filters (3×3, padding=1) + ReLU ──► Feature Maps 11-20
  │                                                     (28×28×10)
  │
  ├─► MaxPool2d (2×2) ──────────────────────────────► Pooled Maps 21-30
  │                                                     (14×14×10)
  │
  ├─► Global Average Pooling (AdaptiveAvgPool2d) ───► (1×1×10)
  │   └─► Spatial dimensions reduced to 1×1 per channel
  │
  ├─► Flatten ──────────────────────────────────────► 10 features
  │
  └─► FC: 10→10 (Linear classifier) ────────────────► Logits
      └─► Softmax → Probabilities (0-9)
```

### Architecture Highlights

**Design Philosophy:**
- **Lightweight backbone**: Only 2 conv layers (10 filters each)
- **Global Average Pooling (GAP)**: Replaces traditional flatten+dense layers
- **Minimal parameters**: Reduces overfitting on small MNIST dataset
- **Preserved spatial info**: No pooling until after both conv layers

**Key Components:**
```python
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling (reduces spatial to 1×1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier (10 features → 10 classes)
        self.fc = nn.Linear(10, 10)
```

**Why This Architecture?**

1. **Global Average Pooling vs Flatten+FC:**
   - Traditional: 14×14×10 = 1,960 params needed in FC1
   - GAP approach: Reduces to just 10 params
   - **Result:** 99% fewer parameters, same accuracy

2. **Padding=1 Strategy:**
   - Maintains 28×28 spatial dimensions through conv layers
   - Allows maxpool to cleanly reduce to 14×14
   - Better feature preservation

3. **Feature Map Extraction:**
   - `convrel1`: After Conv1+ReLU (28×28×10)
   - `convrel2`: After Conv2+ReLU (28×28×10)
   - `pool1`: After MaxPool (14×14×10)
   - Enables visualization of learned features at each stage

**Total Parameters:** ~930  
**Model Size:** 1.2 MB (TorchScript)  

---

## API Contract

### Base URL
```
Production: https://mnist-cnn-torchscript-inference-api-production.up.railway.app/
Local: http://localhost:8000
```

### Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "service": "mnist-cnn-inference",
  "status": "running"
}
```

---

#### `POST /predict`
Performs digit recognition and returns feature maps.

**Request Body:**
```json
{
  "pixels": [
    [0.0, 0.0, ..., 0.0],  // Row 1 (28 values)
    [0.0, 0.5, ..., 0.0],  // Row 2
    ...
    [0.0, 0.0, ..., 0.0]   // Row 28
  ]
}
```

**Schema:**
- `pixels`: 2D array (28×28) of floats in range [0.0, 1.0]
- Must be pre-normalized using MNIST statistics:
  ```python
  normalized = (pixel_value - 0.1307) / 0.3081
  ```

**Response:**
```json
{
  "status": "success",
  "prediction": 7,
  "latency_ms": 12.34,
  "probabilities": [0.001, 0.002, ..., 0.982, 0.003],
  
  "featuremap11": [[...], ...],   // Conv1+ReLU, Filter 1 (28×28)
  "featuremap12": [[...], ...],   // Conv1+ReLU, Filter 2 (28×28)
  ...
  "featuremap110": [[...], ...],  // Conv1+ReLU, Filter 10 (28×28)
  
  "featuremap21": [[...], ...],   // Conv2+ReLU, Filter 1 (28×28)
  "featuremap22": [[...], ...],   // Conv2+ReLU, Filter 2 (28×28)
  ...
  "featuremap210": [[...], ...],  // Conv2+ReLU, Filter 10 (28×28)
  
  "featuremap31": [[...], ...],   // MaxPool, Filter 1 (14×14)
  "featuremap32": [[...], ...],   // MaxPool, Filter 2 (14×14)
  ...
  "featuremap310": [[...], ...]   // MaxPool, Filter 10 (14×14)
}
```

**Feature Map Details:**
- **Layer 1 (convrel1):** 10 maps @ 28×28 — Edge and basic pattern detection
- **Layer 2 (convrel2):** 10 maps @ 28×28 — Complex feature combinations
- **Layer 3 (pool1):** 10 maps @ 14×14 — Spatially downsampled features

**Fields:**
- `prediction` (int): Predicted digit class (0-9)
- `latency_ms` (float): Inference time in milliseconds
- `probabilities` (array): Softmax scores for all 10 classes
- `featuremap1X` (28×28 array): Activation map from Conv1, filter X
- `featuremap2X` (28×28 array): Activation map from Conv2, filter X
- `featuremap3X` (14×14 array): Activation map from MaxPool, filter X

**Error Response:**
```json
{
  "status": "error",
  "message": "Input must be a 28x28 matrix"
}
```

---

### Interactive Documentation
FastAPI provides automatic interactive docs:
- **Swagger UI:** `https://mnist-cnn-torchscript-inference-api-production.up.railway.app/docs`
- **ReDoc:** `https://mnist-cnn-torchscript-inference-api-production.up.railway.app/redoc`

---

## Quick Start

### Prerequisites
- Python 3.9+
- pip or conda

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/ShafwanAdhi/mnist-cnn-torchscript-inference-api.git
cd mnist-cnn-torchscript-inference-api
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Ensure model file exists**
```bash
# Model should be present: new_final_mnist_cnn.ts
ls -lh new_final_mnist_cnn.ts
```

5. **Run the server**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

6. **Test the API**
```bash
# Health check
curl http://localhost:8000/

# Sample prediction (requires valid 28×28 array)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pixels": [[0.0, ...], ...]}'
```

---

## Project Structure

```
mnist-api/
├── main.py                       # FastAPI application
├── new_final_mnist_cnn.ts        # TorchScript model
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version (Railway)                      
└── README.md
```

---

## Deployment

### Railway (Current Platform)

1. **Connect repository to Railway**
```bash
railway login
railway init
railway link
```

2. **Configure environment**
```bash
railway variables set PYTHON_VERSION=3.9.18
```

3. **Deploy**
```bash
git push  # Auto-deploys on push to main
```

**Railway Configuration:**
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Health Check:** `GET /` returns 200

### Cold Start Considerations
- **Model loading:** TorchScript model loaded once at startup (~500ms)
- **First request latency:** May be higher due to Python interpreter warm-up
- **Mitigation:** Railway keeps instances warm with health checks

---

## Configuration

### Environment Variables
```bash
# Optional: Override default host/port
HOST=0.0.0.0
PORT=8000

# Optional: CORS origins (default: allow all)
ALLOWED_ORIGINS=https://your-frontend.com
```

### CORS Settings
Located in `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Testing

### Unit Tests (Coming Soon)
```bash
pytest tests/ -v
```

### Manual Testing
```python
# test_api.py
import requests

url = "http://localhost:8000/predict"
payload = {
    "pixels": [[0.0] * 28 for _ in range(28)]  # Blank canvas
}

response = requests.post(url, json=payload)
print(response.json())
```

---

## Related Repositories

| Repository | Description | Link |
|-----------|-------------|------|
| **mnist-visualizer** | Interactive web interface | [View Repo](https://github.com/ShafwanAdhi/mnist-handwriting-visualizer.git) |
| **mnist-api** (this repo) | Inference API service | You are here |

---

## Future Enhancements

- Add model versioning support
- Implement request caching (Redis)
- Add batch prediction endpoint
- GPU inference option (CUDA)
- Prometheus metrics export
- Docker containerization
- Load testing benchmarks
- A/B testing for multiple models

---

## Dependencies

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
torch==2.1.0
numpy==1.24.3
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Author

**Shafwan Adih Dwi Nugraha**
- GitHub: [@ShafwanAdhi](https://github.com/ShafwanAdhi)
- LinkedIn: [Shafwan Adhi Dwi](https://www.linkedin.com/in/shafwan-adhi-dwi-b90943321/)
- Email: adhishafwan@gmail.com

---

## Acknowledgments

- **PyTorch Team** — TorchScript optimization
- **FastAPI** — Modern Python web framework
- **Railway** — Seamless deployment platform
