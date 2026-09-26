import argparse
import os
import subprocess

import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

# --- Safety check: never submit a paid job from uncommitted code ---
status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
if status.strip():
    raise RuntimeError("Uncommitted changes -- commit before submitting a paid job.")
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"Submitting at commit {commit}")

# --- CLI: which split to run ---
parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["train", "test"], default="train")
args = parser.parse_args()

# --- Config from environment (never hardcode account ID / bucket name) ---
ROLE_ARN = os.getenv("BER_SAGEMAKER_ROLE_ARN")
BUCKET = os.getenv("BER_S3_BUCKET")

if not ROLE_ARN or not BUCKET:
    raise RuntimeError(
        "Missing required env vars. Run:\n"
        "  export BER_SAGEMAKER_ROLE_ARN=<your-role-arn>\n"
        "  export BER_S3_BUCKET=<your-bucket-name>"
    )

# --- Resolve the correct prebuilt image URI for your actual AWS region ---
region = boto3.Session().region_name
if not region:
    raise RuntimeError(
        "No AWS region configured. Run `aws configure` or `export AWS_DEFAULT_REGION=<region>`."
    )

image_uri = sagemaker.image_uris.retrieve(
    framework="pytorch",
    region=region,
    version="2.3.0",
    py_version="py311",
    image_scope="inference",
    instance_type="ml.g4dn.xlarge",
)

# --- Submit the Processing Job ---
processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=ROLE_ARN,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    base_job_name=f"ber-embed-block-{args.split}",
)

processor.run(
    code="aws/embed_and_block.py",
    inputs=[
        ProcessingInput(
            source=f"s3://{BUCKET}/pipeline.duckdb",
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{BUCKET}/ann_candidates/",
        )
    ],
    arguments=[
        "--db-path", "/opt/ml/processing/input/pipeline.duckdb",
        "--out-dir", "/opt/ml/processing/output",
        "--split", args.split,
    ],
)

print(f"Job submitted for split='{args.split}'. Check progress in the SageMaker console under Processing jobs.")