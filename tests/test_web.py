import json

import pandas as pd

from swp import web


class FakeStore:
    def __init__(self):
        self.d = {}

    def exists(self, k):
        return k in self.d

    def get(self, k):
        return self.d[k]

    def put(self, k, b):
        self.d[k] = b

    def list_prefix(self, prefix):
        return [k for k in self.d if k.startswith(prefix)]

def test_recent_window_and_truth():
    s = FakeStore()
    for t0 in ("20261120T000000Z", "20261127T060000Z", "20261128T000000Z"):
        s.put(f"forecasts/{t0}.json", json.dumps({"forecast_speed_kms": [400.0] * 72}).encode())
        s.put(f"runs/{t0}.json", json.dumps({"level": "full"}).encode())
    s.put("truth/202611.json", json.dumps({"values": {"2026-11-27T00:00:00Z": 350.0, "2026-11-27T01:00:00Z": 351.0}}).encode())
    out = web.build_recent(s, pd.Timestamp("2026-11-28T06:00:00"), days=7, truth_days=3)
    assert [c["t0"] for c in out["cycles"]] == ["2026-11-27T06:00:00Z", "2026-11-28T00:00:00Z"]
    assert out["truth"]["values"][-1] == 351.0 and out["truth"]["values"][0] is None