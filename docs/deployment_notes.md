# Deployment Notes

This document summarizes the deployment design and endpoint behavior for the **Cloud-Based PE Malware Detection API** project.

## Overview

The trained malware classification model was deployed as an **Amazon SageMaker real-time inference endpoint**. The purpose of the deployment is to expose the trained model as a callable cloud API so that an external client application can submit feature vectors and receive malware classification results. 

---

## Deployment Components

The deployment relies on the following project files and cloud resources:

- `deployment/inference.py` — SageMaker inference script
- trained model weights — `malconv_final.pt`
- saved scaler — `scaler.pkl`
- SageMaker real-time endpoint — example name: `malconv-endpoint`
- AWS region — example: `us-east-1`

The final run guide identifies `inference.py`, `malconv_final.pt`, and `scaler.pkl` as required project components for deployment and inference. 

---

## Endpoint Contract

The deployed endpoint expects JSON input containing a `features` field with **exactly 2381 EMBER features**.

### Request format

```json
{
  "features": [2381 numeric EMBER features]
}
```

### Response format

```json
{
  "label": "Malware",
  "probability": 0.987,
  "confidence": "High"
}
```

The run guide explicitly confirms that the endpoint expects a `features` field containing **2381 EMBER features**, applies the saved scaler, and returns `label`, `probability`, and `confidence`. 

---

## Inference Flow

The deployed cloud API is used in the following sequence:

1. A PE file is uploaded through the Streamlit client.
2. The client extracts an EMBER v2 feature vector locally.
3. The client confirms that the feature vector length is **2381**.
4. The client sends the feature vector as JSON to the SageMaker endpoint.
5. The endpoint loads the scaler and trained model artifacts.
6. The endpoint applies feature scaling.
7. The model produces a classification result.
8. The endpoint returns the result as JSON with label, probability, and confidence.

---

## Why the Final API Design Uses Features Instead of Raw Bytes

An earlier Task 3 client attempted to serialize large byte arrays directly as JSON. According to the final run guide, this triggered SageMaker request-size errors such as **HTTP 413** because the payload was too large. The deployed endpoint does **not** accept padded raw byte sequences as JSON input. It expects a compact EMBER feature vector with **2381 numeric values**.

This change was important because it reduced payload size from multi-megabyte JSON requests to compact, practical request bodies measured in tens of kilobytes. 

---

## Required Runtime Environment

The deployment and client integration were validated using:

- **Python 3.12**
- `numpy==1.26.4`
- `lief==0.14.1`
- `torch`
- `torchvision`
- `joblib`
- `scikit-learn`
- `boto3`
- `streamlit`
- the **PFGimenez EMBER fork** 

The environment notes and final run guide both indicate that newer versions of Python, NumPy, and LIEF caused compatibility problems, and that the above combination was the working setup.

---

## Compatibility Work Needed for Deployment

Two categories of compatibility work were documented during setup and final testing.

### 1. NumPy compatibility

Older EMBER code references deprecated aliases such as `np.int`. Since NumPy 2.x removes those aliases, the project used `numpy==1.26.4` and, where necessary, applied runtime or direct-source patches. 

### 2. LIEF compatibility

Older EMBER code references exception names that changed in newer LIEF versions. The project pinned `lief==0.14.1` and also documented a patch that maps older exception references to `lief.lief_errors` when needed.

These fixes were necessary to keep feature extraction and inference stable across local development and demo execution. 

---

## Deployment Validation Checklist

A successful deployment should satisfy the following checks:

- the SageMaker endpoint is running and reachable
- the endpoint accepts JSON input with a `features` field
- the feature vector length is **2381**
- the endpoint returns `label`, `probability`, and `confidence`
- latency is reasonable for a live demo
- the Streamlit client can call the endpoint end to end

The course rubric specifically requires a successful SageMaker deployment, proper endpoint configuration, accessible inference functionality, appropriate input/output handling, and reasonable response time.

---

## Launch Sequence for Demo

The final run guide documented this demo sequence:

```bash
source /path/to/venv/bin/activate
export AWS_REGION=us-east-1
export SAGEMAKER_ENDPOINT_NAME=malconv-endpoint
streamlit run app/app.py
```

Then:

1. upload a PE file
2. click **Analyze File**
3. verify **Feature Count = 2381**
4. confirm the response includes prediction label, probability, confidence, payload size, and latency

---

## Troubleshooting

### HTTP 413 from SageMaker
Cause: payload too large.

Fix: send the **2381-value EMBER feature vector**, not raw byte arrays serialized to JSON.

### Endpoint receives wrong feature count
Cause: extraction failed or the wrong extractor was used.

Fix: ensure EMBER v2 extraction is used and confirm the final feature vector length is **2381** before request submission. 

### `ModuleNotFoundError: ember`
Cause: wrong package or wrong environment.

Fix: install the **PFGimenez EMBER fork** in the same virtual environment used to run the client or deployment test code.

### `module 'lief' has no attribute 'bad_format'`
Cause: old EMBER code against newer LIEF behavior.

Fix: pin `lief==0.14.1` and apply the documented compatibility patch if necessary.

### `module 'numpy' has no attribute 'int'`
Cause: EMBER code written for older NumPy.

Fix: keep `numpy==1.26.4` and apply the documented compatibility patch if needed.
