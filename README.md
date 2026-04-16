# Cloud-Based PE Malware Detection API

End-to-end malware classification project using a MalConv-inspired neural network, the EMBER 2017 v2 dataset, Amazon SageMaker for cloud inference, and a Streamlit web client for PE file analysis. The system accepts a Portable Executable (PE) file, converts it into an EMBER-compatible feature vector, submits it to a deployed SageMaker endpoint, and returns a malware/benign prediction with model confidence.

## Project Overview

This project was completed for DSCI 6015: AI and Cybersecurity. The assignment required three connected tasks:

1. Build and train a deep learning malware classifier based on the MalConv architecture.
2. Deploy the trained model as a cloud API using Amazon SageMaker.
3. Create a client application in Streamlit or Gradio that lets a user upload a PE file and receive a classification result.

The model pipeline in this repository uses EMBER 2017 v2 features and a trained PyTorch model. The deployed SageMaker endpoint expects JSON input with a features field containing exactly 2381 EMBER features, applies the saved scaler, runs inference, and returns `label`, `probability`, and `confidence`.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebook/
│   └── malconv_midterm_local.ipynb
├── app/
│   └── app.py
├── deployment/
│   └── inference.py
├── report/
│   └── midterm_report.pdf
├── docs/
│   ├── setup.md
│   ├── deployment_notes.md
│   └── screenshots/
└── models/
    ├── scaler.pkl
    └── malconv_final.pt
```
