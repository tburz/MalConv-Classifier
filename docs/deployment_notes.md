# Deployment Notes

This document summarizes the deployment design and endpoint behavior for the **Cloud-Based PE Malware Detection API** project.

## Overview

The trained malware classification model was deployed as an **Amazon SageMaker real-time inference endpoint**. The purpose of the deployment is to expose the trained model as a callable cloud API so that an external client application can submit feature vectors and receive malware classification results. This deployment satisfies the Task 2 requirement to create an accessible inference endpoint with appropriate input/output handling and documented deployment steps. fileciteturn1file0

---

## Deployment Components

The deployment relies on the following project files and cloud resources:

- `deployment/inference.py` — SageMaker inference script
- trained model weights — `malconv_final.pt`
- saved scaler — `scaler.pkl`
- SageMaker real-time endpoint — example name: `malconv-endpoint`
- AWS region — example: `us-east-1` fileciteturn1file2

The final run guide identifies `inference.py`, `malconv_final.pt`, and `scaler.pkl` as required project components for deployment and inference. fileciteturn1file2

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

The run guide explicitly confirms that the endpoint expects a `features` field containing **2381 EMBER features**, applies the saved scaler, and returns `label`, `probability`, and `confidence`. fileciteturn1file2

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
8. The endpoint returns the result as JSON with label, probability, and confidence. fileciteturn1file2

---

## Why the Final API Design Uses Features Instead of Raw Bytes

An earlier Task 3 client attempted to serialize large byte arrays directly as JSON. According to the final run guide, this triggered SageMaker request-size errors such as **HTTP 413** because the payload was too large. The deployed endpoint does **not** accept padded raw byte sequences as JSON input. It expects a compact EMBER feature vector with **2381 numeric values**. fileciteturn1file2

This change was important because it reduced payload size from multi-megabyte JSON requests to compact, practical request bodies measured in tens of kilobytes. fileciteturn1file2

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
- the **PFGimenez EMBER fork** fileciteturn1file1turn1file2

The environment notes and final run guide both indicate that newer versions of Python, NumPy, and LIEF caused compatibility problems, and that the above combination was the working setup. fileciteturn1file1turn1file2

---

## Compatibility Work Needed for Deployment

Two categories of compatibility work were documented during setup and final testing.

### 1. NumPy compatibility

Older EMBER code references deprecated aliases such as `np.int`. Since NumPy 2.x removes those aliases, the project used `numpy==1.26.4` and, where necessary, applied runtime or direct-source patches. fileciteturn1file1turn1file2

### 2. LIEF compatibility

Older EMBER code references exception names that changed in newer LIEF versions. The project pinned `lief==0.14.1` and also documented a patch that maps older exception references to `lief.lief_errors` when needed. fileciteturn1file1turn1file2

These fixes were necessary to keep feature extraction and inference stable across local development and demo execution. fileciteturn1file1turn1file2

---

## Deployment Validation Checklist

A successful deployment should satisfy the following checks:

- the SageMaker endpoint is running and reachable
- the endpoint accepts JSON input with a `features` field
- the feature vector length is **2381**
- the endpoint returns `label`, `probability`, and `confidence`
- latency is reasonable for a live demo
- the Streamlit client can call the endpoint end to end fileciteturn1file0turn1file2

The course rubric specifically requires a successful SageMaker deployment, proper endpoint configuration, accessible inference functionality, appropriate input/output handling, and reasonable response time. fileciteturn1file0

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
4. confirm the response includes prediction label, probability, confidence, payload size, and latency fileciteturn1file2

---

## Troubleshooting

### HTTP 413 from SageMaker
Cause: payload too large.

Fix: send the **2381-value EMBER feature vector**, not raw byte arrays serialized to JSON. fileciteturn1file2

### Endpoint receives wrong feature count
Cause: extraction failed or the wrong extractor was used.

Fix: ensure EMBER v2 extraction is used and confirm the final feature vector length is **2381** before request submission. fileciteturn1file2

### `ModuleNotFoundError: ember`
Cause: wrong package or wrong environment.

Fix: install the **PFGimenez EMBER fork** in the same virtual environment used to run the client or deployment test code. fileciteturn1file2

### `module 'lief' has no attribute 'bad_format'`
Cause: old EMBER code against newer LIEF behavior.

Fix: pin `lief==0.14.1` and apply the documented compatibility patch if necessary. fileciteturn1file1turn1file2

### `module 'numpy' has no attribute 'int'`
Cause: EMBER code written for older NumPy.

Fix: keep `numpy==1.26.4` and apply the documented compatibility patch if needed. fileciteturn1file1turn1file2

---

## Report Notes

For the written report, the deployment section should clearly state:

- the model was deployed as an **Amazon SageMaker real-time inference endpoint**
- the endpoint expects JSON input with a `features` field containing **2381 values**
- the endpoint applies the saved scaler before inference
- the endpoint returns `label`, `probability`, and `confidence`
- the final client app performs local EMBER feature extraction and then calls the cloud endpoint
- practical compatibility work was required for Python 3.12, NumPy 1.26.4, LIEF 0.14.1, and EMBER integration fileciteturn1file2turn1file1

These details align with the assignment deliverables and help demonstrate that the deployment process, endpoint design, and client integration were fully completed. fileciteturn1file0
