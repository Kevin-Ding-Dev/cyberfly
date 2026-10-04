"""Cyberfly: physical fly locomotion, sensory foraging and maze learning."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Keep generated caches with the project; do not alter the user's global setup.
for variable, folder in (("MPLCONFIGDIR", "matplotlib"),
                         ("FLYGYM_ASSET_CACHE_DIR", "flygym_assets")):
    path = ROOT / ".cache" / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(path))
