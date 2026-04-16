import hashlib
import json
import os
import time

import boto3
import numpy as np
import streamlit as st
import lief

# EMBER import can vary depending on package layout, so try both forms.
try:
    from ember import PEFeatureExtractor
except Exception:
    from ember.features import PEFeatureExtractor


# Read runtime configuration from environment variables.
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME", "malconv-endpoint")
DEBUG_MODE = os.getenv("DEBUG_MODE", "1") == "1"

# Create the SageMaker runtime client once and reuse it.
runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)


def patch_numpy_compat():
    """
    Restore deprecated NumPy aliases used by older EMBER code.
    This avoids editing installed packages directly.
    """
    if not hasattr(np, "int"):
        np.int = np.int64
    if not hasattr(np, "float"):
        np.float = np.float64
    if not hasattr(np, "bool"):
        np.bool = np.bool_
    if not hasattr(np, "complex"):
        np.complex = np.complex128


def patch_lief_compat():
    """
    Map older LIEF exception names to the newer unified lief.lief_errors type.
    This helps older EMBER code run against newer LIEF versions.
    """
    fallback = getattr(lief, "lief_errors", RuntimeError)

    aliases = [
        "bad_format",
        "bad_file",
        "pe_error",
        "parser_error",
        "read_out_of_bound",
        "not_found",
    ]

    for name in aliases:
        if not hasattr(lief, name):
            setattr(lief, name, fallback)


def summarize_features(features: np.ndarray) -> dict:
    """
    Produce a compact debug summary of the extracted feature vector.
    Useful for validating feature shape and checking for bad values.
    """
    features = np.asarray(features, dtype=np.float32).flatten()
    return {
        "feature_count": int(features.shape[0]),
        "dtype": str(features.dtype),
        "min": float(np.min(features)),
        "max": float(np.max(features)),
        "mean": float(np.mean(features)),
        "std": float(np.std(features)),
        "nan_count": int(np.isnan(features).sum()),
        "inf_count": int(np.isinf(features).sum()),
        "nonzero_count": int(np.count_nonzero(features)),
        # Hash a subset of features so runs can be compared without dumping everything.
        "sha256_first_512_features": hashlib.sha256(
            features[:512].tobytes()
        ).hexdigest(),
        "first_16_features": features[:16].tolist(),
    }


def extract_ember_features(file_bytes: bytes) -> np.ndarray:
    """
    Convert raw PE file bytes into a 2381-length EMBER v2 feature vector.
    Also validates shape and checks for invalid numeric values.
    """
    patch_numpy_compat()
    patch_lief_compat()

    extractor = PEFeatureExtractor(feature_version=2)
    features = extractor.feature_vector(file_bytes)
    features = np.asarray(features, dtype=np.float32).flatten()

    # The deployed endpoint expects exactly 2381 EMBER features.
    if features.shape[0] != 2381:
        raise ValueError(f"Expected 2381 features, got {features.shape[0]}")

    if np.isnan(features).any():
        raise ValueError("Feature vector contains NaN values")

    if np.isinf(features).any():
        raise ValueError("Feature vector contains Inf values")

    return features


def invoke_endpoint_with_features(features: np.ndarray):
    """
    Send the extracted feature vector to the SageMaker endpoint
    and return the prediction result, latency, and payload size.
    """
    payload = {
        "features": features.astype(float).tolist()
    }
    payload_json = json.dumps(payload)

    start = time.time()
    response = runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload_json
    )
    latency_ms = (time.time() - start) * 1000

    body = response["Body"].read().decode("utf-8")
    result = json.loads(body)

    return result, latency_ms, len(payload_json.encode("utf-8"))


def main():
    # Configure the Streamlit page layout and title.
    st.set_page_config(page_title="PE Malware Detector", layout="centered")

    st.title("PE Malware Detector")
    st.write("Upload a Windows PE file and classify whether or not it is malware.")

    # Show endpoint configuration for transparency/debugging.
    with st.expander("Endpoint Configuration", expanded=False):
        st.code(
            f"AWS_REGION={AWS_REGION}\n"
            f"SAGEMAKER_ENDPOINT_NAME={SAGEMAKER_ENDPOINT_NAME}\n"
            f"DEBUG_MODE={DEBUG_MODE}"
        )

    # Restrict uploads to common Windows PE-related file extensions.
    uploaded_file = st.file_uploader(
        "Choose a PE file",
        type=["exe", "dll", "sys", "ocx", "scr", "cpl", "drv"]
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        st.subheader("File Details")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Size:** {len(file_bytes):,} bytes")

        if st.button("Analyze File", type="primary"):
            try:
                # Step 1: Extract EMBER features from the uploaded file.
                with st.spinner("Extracting EMBER features..."):
                    features = extract_ember_features(file_bytes)
                    feature_debug = summarize_features(features)

                # Step 2: Send those features to SageMaker for inference.
                with st.spinner("Calling SageMaker endpoint..."):
                    result, latency_ms, payload_size = invoke_endpoint_with_features(features)

                label = result.get("label", "Unknown")
                probability = result.get("probability")
                confidence = result.get("confidence", "Unknown")

                st.subheader("Classification Result")

                # Use different Streamlit message styles for clearer output.
                if label == "Malware":
                    st.error(f"Prediction: {label}")
                elif label == "Benign":
                    st.success(f"Prediction: {label}")
                else:
                    st.warning(f"Prediction: {label}")

                if probability is not None:
                    st.write(f"**Probability:** {float(probability):.6f}")

                st.write(f"**Confidence:** {confidence}")
                st.write(f"**Feature Count:** {len(features)}")
                st.write(f"**Payload Size:** {payload_size:,} bytes")
                st.write(f"**API Latency:** {latency_ms:.2f} ms")

                # Optional client-side diagnostics for debugging/demo purposes.
                if DEBUG_MODE:
                    with st.expander("Client Feature Debug", expanded=False):
                        st.json(feature_debug)

                # Show the full raw response from the API.
                with st.expander("Raw API Response"):
                    st.json(result)

            except Exception as e:
                # Display the full exception in the Streamlit UI.
                st.exception(e)


if __name__ == "__main__":
    main()
