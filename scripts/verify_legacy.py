"""Verify historical baseline plus explicitly recorded, authorized corrections."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "legacy/BASELINE.json").read_text(encoding="utf-8"))
mismatches = []
corrections_path = root / "legacy/CORRECTIONS.json"
corrections = {}
if corrections_path.exists():
    record = json.loads(corrections_path.read_text(encoding="utf-8-sig"))
    if record["baseline_commit"] != manifest["commit"]:
        raise SystemExit("Correction baseline mismatch")
    for correction in record["corrections"]:
        name = correction["path"]
        if name in corrections or manifest["source_sha256"].get(name) != correction["original_sha256"]:
            raise SystemExit(f"Invalid correction provenance: {name}")
        corrections[name] = correction["corrected_sha256"]
for name, expected in manifest["source_sha256"].items():
    expected = corrections.get(name, expected)
    source = root / "legacy/src" / name
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        mismatches.append(name)
print(json.dumps({"baseline": manifest["commit"], "files": len(manifest["source_sha256"]), "authorized_corrections": list(corrections), "mismatches": mismatches}, indent=2))
raise SystemExit(bool(mismatches))
