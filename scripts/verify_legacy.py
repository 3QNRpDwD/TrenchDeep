"""Verify retained baseline source without modifying it (Python standard library only)."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "legacy/BASELINE.json").read_text(encoding="utf-8"))
mismatches = []
for name, expected in manifest["source_sha256"].items():
    source = root / "legacy/src" / name
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        mismatches.append(name)
print(json.dumps({"baseline": manifest["commit"], "files": len(manifest["source_sha256"]), "mismatches": mismatches}, indent=2))
raise SystemExit(bool(mismatches))
