#!/usr/bin/env python3
"""Convert strip_model_lat3.pkl -> strip_model.npz + strip_model_meta.json (no pickle inside the container).
Usage: python3 export_strip_model.py data/strip_model_lat3.pkl"""
import json, pickle, sys
import numpy as np
m = pickle.load(open(sys.argv[1], "rb"))
print("keys:", sorted(m))
np.savez("strip_model.npz", mean=np.asarray(m["mean"], float), scale=np.asarray(m["scale"], float),
         coef=np.asarray(m["coef"], float), intercept=np.asarray(m["intercept"], float),
         adj_min=np.asarray(m["adj_min"], float), adj_max=np.asarray(m["adj_max"], float))
meta = dict(strip_cols=list(m["strip_cols"]), columns=list(m["columns"]), delay_h=float(m["delay_h"]), tol_h=float(m["tol_h"]),
            latency_h=m.get("latency_h"), trained_on=str(m.get("trained_on")))
json.dump(meta, open("strip_model_meta.json", "w"), indent=1)
print("coef", np.shape(m["coef"]), "columns", len(meta["columns"]), "delay", meta["delay_h"], "tol", meta["tol_h"])
