"""Создать bucket один раз; повторный запуск безопасен."""

import os

import boto3
from botocore.exceptions import ClientError

client = boto3.client("s3", endpoint_url="http://rustfs:9000", region_name="us-east-1")
bucket = os.environ["MLFLOW_BUCKET"]
try:
    client.head_bucket(Bucket=bucket)
except ClientError as exc:
    if exc.response["Error"]["Code"] not in {"404", "NoSuchBucket"}:
        raise
    client.create_bucket(Bucket=bucket)
print("Artifact bucket is ready")
