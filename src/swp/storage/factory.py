# src/swp/storage/factory.py
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from .backends import LocalByteStore, S3ByteStore, NullByteStore
from .base import ByteStore


def create_byte_store(
    url: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> ByteStore:
    cfg = os.environ if env is None else env
    raw_url = (url if url is not None else cfg.get("SWP_STORAGE_URL", "")).strip()

    parsed = urlparse(raw_url)
    scheme = parsed.scheme.lower()

    if scheme == "file":
        # Supports file:///abs/path or file://relative/path
        path_part = unquote(parsed.path)
        if parsed.netloc and parsed.netloc != "localhost":
            root_path = Path(f"//{parsed.netloc}{path_part}")
        elif parsed.netloc == "localhost":
            root_path = Path(path_part)
        else:
            root_path = Path(path_part or ".")
        return LocalByteStore(root=root_path)

    if scheme == "s3":
        bucket = parsed.netloc
        if not bucket:
            raise ValueError(f"S3 storage URL missing bucket name: {raw_url!r}")
        prefix = unquote(parsed.path).lstrip("/")

        endpoint_url = (
            cfg.get("SWP_S3_ENDPOINT_URL")
            or cfg.get("R2_ENDPOINT_URL")
            or cfg.get("AWS_ENDPOINT_URL_S3")
            or None
        )
        access_key_id = (
            cfg.get("SWP_S3_ACCESS_KEY_ID")
            or cfg.get("R2_ACCESS_KEY_ID")
            or cfg.get("AWS_ACCESS_KEY_ID")
            or None
        )
        secret_access_key = (
            cfg.get("SWP_S3_SECRET_ACCESS_KEY")
            or cfg.get("R2_SECRET_ACCESS_KEY")
            or cfg.get("AWS_SECRET_ACCESS_KEY")
            or None
        )
        region_name = (
            cfg.get("SWP_S3_REGION")
            or cfg.get("AWS_REGION")
            or cfg.get("AWS_DEFAULT_REGION")
            or ("auto" if endpoint_url else None)
        )

        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region_name,
        )
        return S3ByteStore(s3_client=client, bucket=bucket, prefix=prefix)

    if not raw_url or raw_url.lower() == "none://":
        return NullByteStore()

    raise ValueError(f"Unsupported storage URL: {raw_url!r}")