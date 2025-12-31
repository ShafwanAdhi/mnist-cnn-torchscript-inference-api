from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import time

app = FastAPI(
    title="MNIST CNN TorchScript API",
    description="Inference service for CNN MNIST TorchScript model",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Atau ganti dengan domain spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = torch.jit.load("final_mnist_cnn.ts", map_location="cpu")
model.eval()

class MNISTRequest(BaseModel):
    pixels: list  #[28][28]

@app.get("/")
def root():
    return {
        "service": "mnist-cnn-inference",
        "status": "running"
    }

@app.post("/predict")
def predict(request: MNISTRequest):
    start_time = time.perf_counter()
    if len(request.pixels) != 28 or any(len(row) != 28 for row in request.pixels):
        return {
            "status": "error",
            "message": "Input must be a 28x28 matrix"
        }

    image = np.array(request.pixels, dtype=np.float32)
    image = image.reshape(1, 1, 28, 28)  # (N, C, H, W)
    input_tensor = torch.from_numpy(image).float()
    input_tensor = (input_tensor - 0.1307) / 0.3081

    with torch.no_grad():
        logits, a,b,c = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        #format adalah: (layer, baris)
        featuremap11 = a[0][0]
        featuremap12 = a[0][1]
        featuremap13 = a[0][2]
        featuremap14 = a[0][3]
        featuremap15 = a[0][4]

        featuremap21 = b[0][0]
        featuremap22 = b[0][1]
        featuremap23 = b[0][2]
        featuremap24 = b[0][3]
        featuremap25 = b[0][4]

        featuremap31 = c[0][0]
        featuremap32 = c[0][1]
        featuremap33 = c[0][2]
        featuremap34 = c[0][3]
        featuremap35 = c[0][4]

    latency_ms = (time.perf_counter() - start_time) * 1000

    return {
        "status": "success",
        "prediction": pred,
        "latency_ms": round(latency_ms, 2),
        "probabilities": probs.squeeze(0).tolist(),
        "featuremap11": featuremap11.cpu().tolist(),
        "featuremap12": featuremap12.cpu().tolist(),
        "featuremap13": featuremap13.cpu().tolist(),
        "featuremap14": featuremap14.cpu().tolist(),
        "featuremap15": featuremap15.cpu().tolist(),
        "featuremap21": featuremap21.cpu().tolist(),
        "featuremap22": featuremap22.cpu().tolist(),
        "featuremap23": featuremap23.cpu().tolist(),
        "featuremap24": featuremap24.cpu().tolist(),
        "featuremap25": featuremap25.cpu().tolist(),
        "featuremap31": featuremap31.cpu().tolist(),
        "featuremap32": featuremap32.cpu().tolist(),
        "featuremap33": featuremap33.cpu().tolist(),
        "featuremap34": featuremap34.cpu().tolist(),
        "featuremap35": featuremap35.cpu().tolist(),
    }
