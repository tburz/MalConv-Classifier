# Setup Guide

This document describes the local development environment used for the **Cloud-Based PE Malware Detection API** project and the package versions that were confirmed to work.

## Overview

This project uses:

- **Python 3.12**
- **PyTorch** for the malware classification model
- **EMBER 2017 v2** for PE feature extraction
- **Streamlit** for the Task 3 client app
- **Amazon SageMaker** for the deployed inference endpoint

The environment was tested on an **EndeavourOS** system without a dedicated NVIDIA GPU. Training ran on CPU locally, with SageMaker used for deployment. The setup notes showed that newer Python and package combinations caused compatibility problems, so the environment was pinned to versions that worked reliably.

---

## Recommended Python Version

Use **Python 3.12** in a virtual environment.

Python 3.14 caused compatibility issues during setup, while Python 3.12 worked correctly with the project dependencies and EMBER-related tooling. 

---

## Create the Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On the original system, `pyenv` was used to install and target Python 3.12 explicitly because the default shell alias pointed elsewhere. The setup notes also recommend using full paths to the virtual environment binaries when necessary. 

---

## Install Dependencies

Install packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The working dependency set for this project includes:

- `streamlit`
- `boto3`
- `numpy==1.26.4`
- `lief==0.14.1`
- `torch`
- `torchvision`
- `joblib`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`
- `jupyter`
- `ipykernel`
- the **PFGimenez EMBER fork** installed from GitHub instead of a generic PyPI package named `ember` 

If installing manually, the setup notes used commands equivalent to:

```bash
pip install streamlit boto3 numpy==1.26.4 lief==0.14.1 torch torchvision joblib scikit-learn jupyter ipykernel tqdm seaborn matplotlib
pip install git+https://github.com/PFGimenez/ember.git
```

---

## Why These Versions Matter

Two packages were especially important to pin:

### NumPy
Use:

```text
numpy==1.26.4
```

The EMBER code still references deprecated NumPy aliases such as `np.int`. NumPy 2.x removes those aliases, which breaks older EMBER-compatible code unless additional patching is applied. 

### LIEF
Use:

```text
lief==0.14.1
```

Newer versions of LIEF changed exception names used by EMBER feature extraction code. Pinning `lief==0.14.1` avoids those failures. 

---

## Jupyter Setup

The notebook for Task 1 can be run in Jupyter after registering the virtual environment as a kernel:

```bash
python -m ipykernel install --user --name=malconv --display-name "Python (malconv)"
jupyter lab
```

The setup notes specifically used `ipykernel` for notebook kernel registration. 

---

## EMBER Compatibility Fixes

The setup notes documented two compatibility issues in `ember/features.py` when using modern environments. These may require either a direct patch to the installed file or a runtime patch in the client application. 

### 1. LIEF exception compatibility

Older EMBER code references LIEF exception names that changed in newer releases. One documented fix is to replace the old exception tuple with:

```python
(lief.lief_errors, RuntimeError)
```

This was described in the environment notes and deployment run guide as a required compatibility adjustment. 

### 2. NumPy deprecated aliases

Older EMBER code may contain references such as:

- `np.int`
- `np.float`
- `np.bool`
- `np.complex`

These can be replaced with modern equivalents:

- `np.int64`
- `np.float64`
- `np.bool_`
- `np.complex128`

### Runtime patch option

The final run guide also notes that the Streamlit client can patch NumPy aliases and old LIEF exception names at runtime before importing EMBER, which avoids manually editing site-packages just to run the demo.

---

## Running the Streamlit Client

Before launching the app, export the AWS region and endpoint name:

```bash
export AWS_REGION=us-east-1
export SAGEMAKER_ENDPOINT_NAME=malconv-endpoint
streamlit run app/app.py
```

The client should:

1. accept an uploaded PE file
2. extract an EMBER v2 feature vector
3. verify the vector length is **2381**
4. send the vector to the SageMaker endpoint
5. display the returned label, probability, confidence, payload size, and latency 

---

## Troubleshooting

### `ModuleNotFoundError: ember`
Cause: wrong package or wrong environment.

Fix: install the **PFGimenez EMBER fork** in the same virtual environment used by Streamlit or Jupyter. 

### `module 'ember' has no attribute 'PEFeatureExtractor'`
Cause: import path mismatch.

Fix: use import logic that handles both:

```python
from ember import PEFeatureExtractor
```

and

```python
from ember.features import PEFeatureExtractor
```

This issue was noted in the final run guide. 

### `module 'numpy' has no attribute 'int'`
Cause: old EMBER code against newer NumPy behavior.

Fix: keep `numpy==1.26.4` and/or apply the documented compatibility patch. 

### `module 'lief' has no attribute 'bad_format'`
Cause: EMBER code written against older LIEF exception names.

Fix: use `lief==0.14.1` and apply the documented LIEF patch if needed. 

### Feature count is not 2381
Cause: extraction failed or the wrong feature extractor was used.

Fix: confirm the app is using EMBER v2 extraction and flattening the output correctly before sending it to the endpoint.
