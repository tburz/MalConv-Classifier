import json
import os
from typing import Any, Dict, Union

import joblib
import numpy as np
import torch
import torch.nn as nn


# Classification cutoff used to convert probability into Malware vs Benign.
THRESHOLD = 0.5

# EMBER v2 feature vector length expected by the model.
FEATURE_DIM = 2381


class MalConv(nn.Module):
    """
    Simplified MalConv-style model used for inference on EMBER feature vectors.
    """
    def __init__(self, input_dim: int = 2381, output_dim: int = 1):
        super().__init__()
        # First 1D convolution processes the input feature sequence in large chunks.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=128, stride=128)

        # Second convolution refines the learned feature map.
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, stride=1)

        # Global max pooling reduces the sequence dimension to a fixed-size vector.
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers produce the final binary prediction score.
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input arrives as (batch, 2381); add a channel dimension for Conv1d.
        x = x.unsqueeze(1)          # (batch, 1, 2381)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Pool across the sequence dimension, leaving one value per channel.
        x = self.pool(x).squeeze(-1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # Sigmoid converts the output logit into a probability in [0, 1].
        x = self.sigmoid(x)
        return x


def _array_stats(name: str, arr: np.ndarray) -> None:
    """
    Print compact debug statistics for an array.
    Useful for validating input, scaled features, and scaler parameters.
    """
    arr = np.asarray(arr)
    print(
        (
            f"DEBUG: {name} "
            f"shape={arr.shape} "
            f"dtype={arr.dtype} "
            f"min={float(np.min(arr)):.6f} "
            f"max={float(np.max(arr)):.6f} "
            f"mean={float(np.mean(arr)):.6f} "
            f"std={float(np.std(arr)):.6f} "
            f"nan_count={int(np.isnan(arr).sum())} "
            f"inf_count={int(np.isinf(arr).sum())} "
            f"nonzero_count={int(np.count_nonzero(arr))}"
        ),
        flush=True,
    )
    preview = arr.reshape(-1)[:16].tolist()
    print(f"DEBUG: {name} first_16={preview}", flush=True)


def _confidence_from_threshold(prob: float, threshold: float) -> str:
    """
    Convert distance from the threshold into a simple confidence label.
    """
    distance = abs(prob - threshold)
    if distance > 0.4:
        return "High"
    if distance > 0.2:
        return "Medium"
    return "Low"


def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    SageMaker model loader.
    Loads the saved PyTorch model and scaler from the model artifact directory.
    """
    print("DEBUG: entering model_fn", flush=True)

    model_path = os.path.join(model_dir, "malconv_final.pt")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    print(f"DEBUG: model path = {model_path}", flush=True)
    print(f"DEBUG: scaler path = {scaler_path}", flush=True)
    print(f"DEBUG: threshold = {THRESHOLD}", flush=True)

    # Inference runs on CPU in this deployment.
    device = torch.device("cpu")

    model = MalConv()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    scaler = joblib.load(scaler_path)

    print(f"DEBUG: scaler class = {type(scaler)}", flush=True)

    # Log scaler statistics to help verify the correct preprocessing artifact loaded.
    if hasattr(scaler, "mean_"):
        _array_stats("scaler.mean_", np.asarray(scaler.mean_, dtype=np.float32))
    if hasattr(scaler, "scale_"):
        _array_stats("scaler.scale_", np.asarray(scaler.scale_, dtype=np.float32))

    print("DEBUG: model_fn complete", flush=True)
    return {"model": model, "scaler": scaler, "device": device}


def _parse_input(request_body: Union[str, bytes, Dict[str, Any]], content_type: str):
    """
    Parse incoming request data into a 2D NumPy array shaped (1, FEATURE_DIM).
    Supports JSON and simple CSV/plain-text numeric input.
    """
    print(f"DEBUG: _parse_input content_type={content_type}", flush=True)

    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")

    if content_type == "application/json":
        payload = json.loads(request_body)

        # Accept either {"features": [...]} or a raw JSON list.
        if isinstance(payload, dict) and "features" in payload:
            features = payload["features"]
        elif isinstance(payload, list):
            features = payload
        else:
            raise ValueError("JSON payload must be either a list or {'features': [...]}")

        arr = np.asarray(features, dtype=np.float32).reshape(1, -1)
        print(f"DEBUG: parsed json array shape = {arr.shape}", flush=True)
        _array_stats("parsed_json_features", arr)
        return arr

    if content_type in ("text/csv", "application/csv", "text/plain"):
        values = [v.strip() for v in request_body.replace("\n", ",").split(",") if v.strip()]
        arr = np.asarray([float(v) for v in values], dtype=np.float32).reshape(1, -1)
        print(f"DEBUG: parsed csv/plain array shape = {arr.shape}", flush=True)
        _array_stats("parsed_csv_features", arr)
        return arr

    raise ValueError(f"Unsupported content type: {content_type}")


def input_fn(request_body, content_type: str = "application/json"):
    """
    SageMaker input handler.
    Converts the raw request body into model-ready numeric input.
    """
    print("DEBUG: entering input_fn", flush=True)
    data = _parse_input(request_body, content_type)
    print(f"DEBUG: leaving input_fn with shape {data.shape}", flush=True)
    return data


def predict_fn(input_data, model_bundle):
    """
    SageMaker prediction handler.
    Validates input, applies the saved scaler, runs the model,
    and formats the prediction result.
    """
    print("DEBUG: entering predict_fn", flush=True)

    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    device = model_bundle["device"]

    print(f"DEBUG: raw input shape = {input_data.shape}", flush=True)

    if input_data.ndim != 2:
        raise ValueError(f"Expected 2D input array, got shape {input_data.shape}")

    if input_data.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} features, got {input_data.shape[1]}")

    if np.isnan(input_data).any():
        raise ValueError("Input data contains NaN values")

    if np.isinf(input_data).any():
        raise ValueError("Input data contains Inf values")

    _array_stats("predict_fn.raw_input", input_data)

    # Apply the same scaler used during training before model inference.
    print("DEBUG: before scaler.transform", flush=True)
    features = scaler.transform(input_data).astype(np.float32)
    print("DEBUG: after scaler.transform", flush=True)

    if np.isnan(features).any():
        raise ValueError("Scaled features contain NaN values")

    if np.isinf(features).any():
        raise ValueError("Scaled features contain Inf values")

    _array_stats("predict_fn.scaled_features", features)

    tensor = torch.tensor(features, dtype=torch.float32, device=device)

    with torch.no_grad():
        print("DEBUG: before model forward", flush=True)
        model_output = model(tensor)
        prob = float(model_output.item())
        print(f"DEBUG: raw model output tensor = {model_output.cpu().numpy().tolist()}", flush=True)
        print(f"DEBUG: after model forward, prob={prob}", flush=True)

    # Convert numeric probability into a label and rough confidence bucket.
    label = "Malware" if prob > THRESHOLD else "Benign"
    confidence = _confidence_from_threshold(prob, THRESHOLD)

    result = {
        "label": label,
        "probability": round(prob, 6),
        "confidence": confidence,
        "threshold": THRESHOLD,
    }

    print(f"DEBUG: prediction result = {result}", flush=True)
    return result


def output_fn(prediction, accept: str = "application/json"):
    """
    SageMaker output handler.
    Returns the prediction in JSON by default.
    """
    print("DEBUG: entering output_fn", flush=True)
    if accept == "application/json":
        return json.dumps(prediction), accept
    return str(prediction), accept
