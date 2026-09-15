"""Build/run product route benchmarks in separate fresh processes; stdlib only.

Default: 3 processes/route/build, 5 warmups + 30 samples, instrumented lifecycle
5 warmups + 100 batches. Instrumented seconds are deliberately not summarized.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True, encoding="utf-8").strip()


def digest(paths):
    entries = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(paths)}
    value = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"sha256": value, "files": entries}


def cpu_name():
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0]
    return platform.processor()


def build(instrumented, out):
    features = "legacyBenchmark" + (",benchmarkAlloc" if instrumented else "")
    proc = subprocess.run(["cargo", "build", "--release", "--bench", "context_diffusion",
                           "--features", features, "--message-format=json"], cwd=ROOT,
                          text=True, encoding="utf-8", capture_output=True)
    (out / f"build-{instrumented}.log").write_text(proc.stderr + proc.stdout, encoding="utf-8")
    proc.check_returncode()
    for line in proc.stdout.splitlines():
        event = json.loads(line)
        if event.get("reason") == "compiler-artifact" and event.get("executable") and event["target"]["name"] == "context_diffusion":
            executable = Path(event["executable"])
    return executable, features


def close(a, b, label, exact=False):
    if len(a) != len(b):
        raise ValueError(f"{label}: length mismatch")
    maximum = 0.0
    for i, (x, y) in enumerate(zip(a, b)):
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"{label}[{i}]: nonfinite")
        error = abs(x-y)
        if error > (0 if exact else max(1e-3, 1e-3*max(abs(x),abs(y)))):
            raise ValueError(f"{label}[{i}]: {x} != {y}")
        maximum = max(maximum, error)
    return maximum


def compare(a, b):
    errors = {}
    for key in ["initial_weights", "prediction", "gradients", "updated_weights"]:
        errors[key] = close(a["evidence"][key], b["evidence"][key], key, key == "initial_weights")
    errors["loss"] = close([a["evidence"]["loss"]], [b["evidence"]["loss"]], "loss")
    for key in ["training_losses", "training_weights", "sampling_output"]:
        errors[key] = close(a[key], b[key], key)
    for i, (x, y) in enumerate(zip(a["rng_feeds"], b["rng_feeds"])):
        assert x["t"] == y["t"]
        close(x["noise"], y["noise"], f"noise{i}", exact=True)
        close(x["loss"], y["loss"], f"rng_loss{i}")
    return errors


def summarize(data):
    result = {}
    for phase in sorted({m["phase"] for m in data["measurements"]}):
        samples = [m for m in data["measurements"] if m["phase"] == phase]
        if data["instrumented"]:
            result[phase] = {key: statistics.median(m["memory"][key] for m in samples)
                             for key in samples[0]["memory"]}
            result[phase]["peak_extra_bytes"] = statistics.median(m["peak_extra_bytes"] for m in samples)
            result[phase]["start_live_bytes"] = statistics.median(m["start_live_bytes"] for m in samples)
            result[phase]["samples"] = len(samples)
        else:
            seconds = sorted(m["seconds"] for m in samples)
            median = statistics.median(seconds)
            units = 3 if phase == "training_3_epochs" else 10 if phase == "sampling_10_steps" else 1
            result[phase] = {"median_seconds": median, "p95_seconds": seconds[math.ceil(len(seconds)*.95)-1],
                             "samples": len(seconds), "units_per_second": units/median,
                             "unit": "reverse_steps" if phase == "sampling_10_steps" else "batches" if phase == "training_3_epochs" else "calls"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "target/p1/context-baseline")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--prepared", action="store_true", help="also compare explicit static execution")
    parser.add_argument("--mode", choices=["timing", "memory", "both"], default="both")
    args = parser.parse_args()
    if args.samples < 1 or args.processes < 1:
        parser.error("positive samples/processes required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    for folder in ["src", "benches", "scripts"]:
        paths.extend(p for p in (ROOT/folder).rglob("*") if p.suffix in [".rs", ".py"])
    source = digest(paths)
    metadata = {"source":source, "head":command("git", "rev-parse", "HEAD"),
                "dirty":command("git", "status", "--porcelain"), "rustc":command("rustc", "-Vv"),
                "cargo":command("cargo", "-V"), "os":platform.platform(),
                "cpu":cpu_name(), "logical_cpus":os.cpu_count(),
                "started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "thread_policy":"serial benchmark; product default threading, no overrides",
                "profile":"release opt-level=3, no visualization/debugging",
                "warmup":5, "samples":args.samples, "processes_per_route":args.processes,
                "p95_method":"nearest-rank per process; small sample descriptive statistic",
                "command":sys.argv}
    (out/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = {"metadata":metadata, "runs":[], "equivalence":[]}
    for instrumented in ([False, True] if args.mode == "both" else [args.mode == "memory"]):
        executable, features = build(instrumented, out)
        binary_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        for process in range(args.processes):
            pair = {}
            # Alternate route order to reduce systematic thermal/order bias.
            routes = ["p1", "legacy", "prepared"] if args.prepared else ["p1", "legacy"]
            for route in (routes if process % 2 == 0 else routes[::-1]):
                name = f"{'memory' if instrumented else 'timing'}-{route}-{process}"
                print(f"Running {name}", flush=True)
                with (out/f"{name}.log").open("w", encoding="utf-8") as log:
                    subprocess.run([str(executable), route, str(out/f"{name}.json"), str(args.samples)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                data = json.loads((out/f"{name}.json").read_text())
                pair[route] = data
                fixture = {"fixture":data["fixture"], "initial_weights":data["evidence"]["initial_weights"],
                           "rng_feeds":[{"t":f["t"],"noise":f["noise"]} for f in data["rng_feeds"]]}
                fixture_digest = hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest()
                # Sidecar avoids rewriting raw evidence or mixing metadata allocation with measurements.
                run = {"name":name, "features":features, "binary_sha256":binary_digest,
                       "available_parallelism":data["available_parallelism"],
                       "source_sha256":source["sha256"], "fixture_sha256":fixture_digest,
                       "statistics":summarize(data), "memory_lifecycle":data["memory_lifecycle"]}
                summary["runs"].append(run)
                (out/f"{name}-summary.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
            summary["equivalence"].append({"instrumented":instrumented, "process":process,
                                             "maximum_absolute_errors":compare(pair["p1"],pair["legacy"])})
            if args.prepared:
                summary["equivalence"].append({"instrumented":instrumented,"process":process,"comparison":"p1/prepared",
                    "maximum_absolute_errors":compare(pair["p1"],pair["prepared"])})
            (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if digest(paths) != source:
        raise RuntimeError("source changed during measurement; discard results")
    print(out / "summary.json")


if __name__ == "__main__":
    main()
