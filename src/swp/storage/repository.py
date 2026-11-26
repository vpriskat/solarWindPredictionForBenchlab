# src/swp/storage/repository.py
import json
from .base import ByteStore
import math
from typing import Any

def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj

class CycleRepository:
    def __init__(self, store: ByteStore) -> None:
        self.store = store

    def save_forecast(self, t0_key: str, payload: dict) -> str:
        key = f"forecasts/{t0_key}.json"
        clean_payload = _to_jsonable(payload)
        self.store.put(key, json.dumps(clean_payload, indent=2).encode("utf-8"))
        return key

    def load_forecast(self, t0_key: str) -> dict | None:
        raw = self.store.get(f"forecasts/{t0_key}.json")
        return json.loads(raw.decode("utf-8")) if raw is not None else None

    def save_run_record(self, t0_key: str, record: dict) -> str:
        key = f"runs/{t0_key}.json"
        clean_payload = _to_jsonable(record)
        self.store.put(key, json.dumps(clean_payload).encode("utf-8"))
        return key