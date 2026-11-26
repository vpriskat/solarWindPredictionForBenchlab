from botocore.exceptions import ClientError
import os
import tempfile
from pathlib import Path


class LocalByteStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def put(self, key: str, data: bytes) -> None:
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, target)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def get(self, key: str) -> bytes | None:
        target = self.root / key
        return target.read_bytes() if target.exists() else None

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def list_prefix(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        out: list[str] = []
        for p in base.rglob("*"):
            if p.is_file():
                out.append(p.relative_to(self.root).as_posix())
        return sorted(out)


class NullByteStore:
    def put(self, key: str, data: bytes) -> None:
        return None

    def get(self, key: str) -> bytes | None:
        return None

    def exists(self, key: str) -> bool:
        return False

    def list_prefix(self, prefix: str) -> list[str]:
        return []

class S3ByteStore:
    def __init__(
        self,
        s3_client,
        bucket: str,
        prefix: str = "",
    ) -> None:
        self.s3 = s3_client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _full_key(self, key: str) -> str:
        clean_key = key.lstrip("/")
        return f"{self.prefix}/{clean_key}" if self.prefix else clean_key

    def _strip_prefix(self, full_key: str) -> str:
        if self.prefix and full_key.startswith(f"{self.prefix}/"):
            return full_key[len(self.prefix) + 1 :]
        return full_key

    def put(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=self._full_key(key), Body=data)

    def get(self, key: str) -> bytes | None:
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self._full_key(key))
            return resp["Body"].read()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey"}:
                return None
            raise

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"404", "NoSuchKey"}:
                return False
            raise

    def list_prefix(self, prefix: str) -> list[str]:
        full_prefix = self._full_key(prefix)
        paginator = self.s3.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for item in page.get("Contents", []):
                keys.append(self._strip_prefix(item["Key"]))
        return sorted(keys)