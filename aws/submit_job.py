import subprocess
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
if status.strip():
    raise RuntimeError("Uncommitted changes -- commit before submitting a paid job.")
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"Submitting at commit {commit}")

ROLE_ARN = "arn:aws:iam::<your-account-id>:role/<your-sagemaker-execution-role>"
BUCKET = "mkn-ber-challenge"

processor = ScriptProcessor(
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.3.0-gpu-py311",
    command=["python3"],
    role=ROLE_ARN,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    base_job_name="ber-embed-block",
)

processor.run(
    code="aws/embed_and_block.py",
    inputs=[ProcessingInput(source=f"s3://{BUCKET}/pipeline.duckdb", destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{BUCKET}/ann_candidates/")],
    arguments=["--db-path", "/opt/ml/processing/input/pipeline.duckdb",
               "--out-dir", "/opt/ml/processing/output", "--split", "train"],
)