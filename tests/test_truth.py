import numpy as np
from swp import truth


def test_merge_never_overwrites_with_missing():
    old = {"2026-11-01T00:00:00Z": 300.0, "2026-11-01T01:00:00Z": 310.0}
    new = {"2026-11-01T01:00:00Z": float("nan"), "2026-11-01T02:00:00Z": 320.0}
    out = truth.merge_truth(old, new)
    assert out == {"2026-11-01T00:00:00Z": 300.0, "2026-11-01T01:00:00Z": 310.0, "2026-11-01T02:00:00Z": 320.0}


def test_merge_new_value_wins():
    assert truth.merge_truth({"2026-11-01T00:00:00Z": 300.0}, {"2026-11-01T00:00:00Z": 305.0}) == {"2026-11-01T00:00:00Z": 305.0}


def test_month_key():
    assert truth.month_key("2026-11-28T06:00:00Z") == "202611"