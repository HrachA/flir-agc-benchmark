"""Verify published archives and raw timing/quality records without a GPU or dataset."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(results, dataset=None):
    specs = json.loads((ROOT / "configs/default.json").read_text())["methods"]
    names = {s["name"] for s in specs}
    manifests = {name: list(rows(results / name / "manifest.csv")) for name in ["python", "native", "cuda"]}
    identity = lambda records: {(r["split"], r["image_id"], r["tiff_sha256"], r["jpeg_sha256"]) for r in records}
    require(identity(manifests["python"]) == identity(manifests["native"]) == identity(manifests["cuda"]), "Backend input manifests differ")
    counts = Counter(r["split"] for r in manifests["python"])
    require(counts == {"val": 1144, "test": 2151}, "Published pair counts changed")
    keys = {(r["split"], r["image_id"], method) for r in manifests["python"] for method in names}
    total_quality = 0
    for backend, engines in [("python", {"python"}), ("native", {"python", "cpp_exact_1", "cpp_fast_1"}), ("cuda", {"python", "cuda"})]:
        seen = set()
        quality = results / backend / "per_image_metrics.csv.gz"
        if not quality.exists():
            quality = results / backend / "per_image_metrics.csv"
        for r in rows(quality):
            key = (r["split"], r["image_id"], r["method"])
            engine = r.get("engine", "python")
            require(key in keys and engine in engines, "Unexpected quality record")
            unique = (*key, engine)
            require(unique not in seen, "Duplicate quality record")
            seen.add(unique)
            require(math.isfinite(float(r["ssim"])) and -1 <= float(r["ssim"]) <= 1, "Invalid SSIM")
        require(len(seen) == len(keys) * len(engines), "Incomplete quality archive: " + backend)
        total_quality += len(seen)
    for backend, engines in [("native", {"cpp_exact_1", "cpp_fast_1"}), ("cuda", {"cuda"})]:
        seen = set()
        parity = results / backend / "parity_per_image.csv.gz"
        if not parity.exists():
            parity = results / backend / "parity_per_image.csv"
        for r in rows(parity):
            engine = r.get("engine", "cuda")
            key = (r["split"], r["image_id"], r["method"])
            require(key in keys and engine in engines, "Unexpected parity record")
            unique = (*key, engine)
            require(unique not in seen, "Duplicate parity record")
            seen.add(unique)
            if engine in {"cpp_exact_1", "cuda"}:
                require(int(r["different_pixels"]) == 0, "Exact backend parity failed")
        require(len(seen) == len(keys) * len(engines), "Incomplete parity archive")
    for backend in ["python", "native", "cuda"]:
        folder = results / backend
        fields = ["device_single_ms", "device_amortized_ms", "gpu_roundtrip_ms", "host_roundtrip_ms"] if backend == "cuda" else ["latency_ms"]
        sums, timing_counts, unique = defaultdict(float), Counter(), set()
        for r in rows(folder / "latency_samples.csv"):
            engine = r.get("engine", backend)
            identity_key = (engine, r["method"], r["split"], r["image_id"], r["repeat"])
            require(identity_key not in unique, "Duplicate timing observation")
            unique.add(identity_key)
            timing_counts[(engine, r["method"])] += 1
            for field in fields:
                value = float(r[field])
                require(math.isfinite(value) and value > 0, "Invalid timing")
                for split in [r["split"], "pooled"]:
                    sums[(engine, r["method"], split, field)] += value
        expected_engines = {"python", "cpp_exact_1", "cpp_exact_8", "cpp_fast_1", "cpp_fast_8"} if backend == "native" else {backend}
        require(set(timing_counts) == {(e, n) for e in expected_engines for n in names}, "Missing timing setting/profile")
        require(set(timing_counts.values()) == {640}, "Wrong timing repeat count")
        for r in rows(folder / "latency_summary.csv"):
            field = r.get("mode", "latency_ms")
            engine = r.get("engine", backend)
            actual = sums[(engine, r["method"], r["split"], field)] / int(r["samples"])
            require(math.isclose(actual, float(r["mean_ms"]), rel_tol=1e-10), "Summary disagrees with raw timing samples")
    batch_counts = Counter(r["method"] for r in rows(results / "cuda/batch_samples.csv"))
    require(set(batch_counts) == names and set(batch_counts.values()) == {160}, "Incomplete CUDA streaming timings")
    require(all(int(r["different_pixels"]) == 0 for r in rows(results / "cuda/batch_integrity.csv")), "Streaming parity failed")
    for r in rows(results / "native/thread_determinism.csv"):
        require(int(r["different_pixels_threads_1_vs_8"]) == 0, "Native thread determinism failed")
    quality = {(r["split"], r["method"]): r for r in rows(results / "python/quality_summary.csv")}
    metric_fields = [k for k in next(iter(quality.values())) if k.endswith("_mean")]
    for backend, exact in [("native", "cpp_exact_1"), ("cuda", "cuda")]:
        for r in rows(results / backend / "quality_summary.csv"):
            if r["engine"] == exact:
                base = quality[(r["split"], r["method"])]
                for field in metric_fields:
                    require(float(r[field]) == float(base[field]), "Exact backend quality differs")
    winner = max((r for r in quality.values() if r["split"] == "val"), key=lambda r: float(r["ssim_mean"]))["method"]
    require(winner == "plateau_p1000", "Validation winner changed")
    if dataset:
        for r in manifests["python"]:
            for kind in ["tiff", "jpeg"]:
                require(sha(dataset / r[kind + "_path"]) == r[kind + "_sha256"], "Source input hash changed")
    print(f"PASS: {sum(counts.values())} shared pairs; {len(names)} settings; {total_quality} complete quality records; exact C++/CUDA parity; raw timing means match summaries.")
    print("PASS: fused C++ is checked separately; streaming/thread checks passed; selection uses validation SSIM.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/published")
    parser.add_argument("--dataset-root", type=Path, help="Optionally also verify every source TIFF/JPEG hash")
    parser.add_argument("--check-export-hashes", action="store_true")
    args = parser.parse_args()
    if args.check_export_hashes:
        exported = json.loads((ROOT / "archive_import.json").read_text())
        for item in exported["files"]:
            require(sha(ROOT / item["export_path"]) == item["sha256"], "Changed exported file: " + item["export_path"])
        print("PASS: every imported archive/source-file export hash matches.")
    verify(args.results.resolve(), args.dataset_root.resolve() if args.dataset_root else None)


if __name__ == "__main__":
    main()
