"""Behavior descriptors and optional matched-data discrepancy, never a fake % score."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

FEATURES = ("mean_speed_mm_s", "mean_turn_rate_rad_s", "stopped_fraction",
            "feeding_fraction", "mean_feeding_bout_s")
SCALE_FLOORS = np.array([1.0, 0.2, 0.05, 0.05, 0.1])


def describe(rows):
    if len(rows) < 2:
        raise ValueError("A trajectory requires at least two samples")
    t = np.array([float(r["time_s"]) for r in rows])
    xy = np.array([[float(r["x_mm"]), float(r["y_mm"])] for r in rows])
    angles = np.array([float(r["heading_rad"]) for r in rows])
    feeding = np.array([int(r["feeding"]) for r in rows], dtype=bool)
    if not all(np.isfinite(a).all() for a in (t, xy, angles)) or np.any(np.diff(t) <= 0):
        raise ValueError("Trajectory samples must be finite and time strictly increasing")
    dt = np.diff(t)
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1)/dt
    angle_delta = np.arctan2(np.sin(np.diff(angles)), np.cos(np.diff(angles)))
    turning = abs(angle_delta)/dt
    bouts, current = [], 0.0
    for flag, duration in zip(feeding[1:], dt):
        if flag:
            current += duration
        elif current:
            bouts.append(current)
            current = 0.0
    if current:
        bouts.append(current)
    return {"mean_speed_mm_s": float(np.average(speed, weights=dt)),
            "mean_turn_rate_rad_s": float(np.average(turning, weights=dt)),
            "stopped_fraction": float(np.average(speed<1.0, weights=dt)),
            "feeding_fraction": float(np.average(feeding[1:], weights=dt)),
            "mean_feeding_bout_s": float(np.mean(bouts)) if bouts else 0.0}


class BiologicalReference:
    """A reference manifest must describe real measurements and matched conditions.

Schema and assertions are checked; this tool cannot verify the scientific
provenance or experimental matching claimed by a user-supplied manifest.
"""
    def __init__(self, manifest_path, config):
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text())
        required = {"schema", "source_url", "species", "sex", "protocol_notes",
                    "position_unit", "time_unit", "trajectories_csv",
                    "matched_forage_config"}
        if not required <= manifest.keys() or manifest["schema"] != 1:
            raise ValueError("Invalid biological reference manifest")
        if manifest["position_unit"] != "mm" or manifest["time_unit"] != "s":
            raise ValueError("Reference must explicitly use millimeters and seconds")
        if manifest["species"] != "Drosophila melanogaster":
            raise ValueError("Reference species does not match")
        if not str(manifest["source_url"]).startswith("https://") or not manifest["protocol_notes"]:
            raise ValueError("Supply a source URL and experimental protocol notes")
        if manifest["matched_forage_config"] != config.to_dict():
            raise ValueError("Reference must explicitly document matching task configuration")
        data_path = (path.parent/manifest["trajectories_csv"]).resolve()
        with data_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        groups = {}
        for row in rows:
            if row.get("feeding") not in ("0", "1"):
                raise ValueError("feeding annotations must be 0 or 1")
            groups.setdefault(row["episode_id"], []).append(row)
        if len(groups) < 5:
            raise ValueError("Supply at least five independently annotated real trials")
        self.features = [describe(sorted(group, key=lambda r:float(r["time_s"])))
                         for group in groups.values()]
        self.digest = hashlib.sha256(path.read_bytes()+data_path.read_bytes()).hexdigest()
        self.source_url = manifest["source_url"]

    def compare(self, features):
        ref = np.array([[row[key] for key in FEATURES] for row in self.features])
        samples = np.array([[row[key] for key in FEATURES] for row in features])
        scales = np.maximum(np.percentile(ref,75,axis=0)-np.percentile(ref,25,axis=0),
                            SCALE_FLOORS)
        distances = [wasserstein_distance(ref[:,i],samples[:,i])/scales[i]
                     for i in range(len(FEATURES))]
        return {"available": True, "distance": float(np.mean(distances)),
                "components": dict(zip(FEATURES,distances)),
                "reference_sha256": self.digest, "source_url": self.source_url,
                "meaning": "Normalized behavioral distribution discrepancy; lower is closer. "
                           "Not whole-animal fidelity or a biological similarity percentage."}


def no_reference():
    return {"available":False, "distance":None,
            "reason":"No experimentally matched real trajectory dataset supplied"}
