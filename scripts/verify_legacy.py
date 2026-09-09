"""Verify integrated native sources and preserved correction provenance."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
provenance = root / "src/native_provenance"
manifest = json.loads((provenance / "BASELINE.json").read_text(encoding="utf-8"))
mismatches = []
corrections_path = provenance / "CORRECTIONS.json"
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
integrated = json.loads((provenance / "INTEGRATED.json").read_text(encoding="utf-8"))
seen = set()
for entry in integrated["files"]:
    name = entry["original"]
    if name in seen or name not in manifest["source_sha256"]:
        raise SystemExit(f"Invalid integration provenance: {name}")
    seen.add(name)
    if entry.get("removed"):
        continue
    source = (root / entry["path"]).resolve()
    if not source.is_relative_to(root / "src"):
        raise SystemExit(f"Native source outside src: {source}")
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
        mismatches.append(entry["path"])
if seen != set(manifest["source_sha256"]):
    raise SystemExit("Incomplete integration mapping")
print(json.dumps({"baseline": manifest["commit"], "mapped_files": len(seen), "retained_files": sum(not e.get("removed", False) for e in integrated["files"]), "authorized_corrections": list(corrections), "mismatches": mismatches}, indent=2))
raise SystemExit(bool(mismatches))
