"""Build the combined report, all-setting tables and figures from archived CSVs."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def collect(results):
    methods = json.loads((ROOT / "configs/default.json").read_text())["methods"]
    quality = {(r["split"], r["method"]): r for r in read(results / "python/quality_summary.csv")}
    original = {(r["split"], r["method"]): r for r in read(results / "python/latency_summary.csv")}
    native = {(r["engine"], r["split"], r["method"]): r for r in read(results / "native/latency_summary.csv")}
    cuda = {(r["mode"], r["split"], r["method"]): r for r in read(results / "cuda/latency_summary.csv")}
    batch = {r["method"]: r for r in read(results / "cuda/batch_summary.csv")}
    result = []
    for spec in methods:
        name = spec["name"]
        row = {"method": name, "family": spec["family"], "parameters_json": json.dumps({k: v for k, v in spec.items() if k not in ["name", "family"]}, sort_keys=True),
               "val_ssim": float(quality[("val", name)]["ssim_mean"]),
               "test_ssim": float(quality[("test", name)]["ssim_mean"]),
               "test_psnr_db": float(quality[("test", name)]["psnr_db_mean"]),
               "test_mae": float(quality[("test", name)]["mae_mean"]),
               "python_original_ms": float(original[("pooled", name)]["mean_ms"])}
        for engine in ["python", "cpp_exact_1", "cpp_exact_8", "cpp_fast_1", "cpp_fast_8"]:
            r = native[(engine, "pooled", name)]
            prefix = "python_same_cpu_session" if engine == "python" else engine
            for field in ["mean_ms", "median_ms", "p95_ms", "images_per_second"]:
                row[prefix + "_" + field] = float(r[field])
        for mode in ["device_single_ms", "device_amortized_ms", "gpu_roundtrip_ms", "host_roundtrip_ms"]:
            r = cuda[(mode, "pooled", name)]
            row["cuda_" + mode] = float(r["mean_ms"])
            row["cuda_" + mode.removesuffix("_ms") + "_p95_ms"] = float(r["p95_ms"])
            row["cuda_" + mode.removesuffix("_ms") + "_images_per_second"] = float(r["images_per_second"])
        row["cuda_4stream_batch_ms"] = float(batch[name]["mean_batch_ms"])
        row["cuda_4stream_amortized_ms"] = float(batch[name]["amortized_ms_per_image"])
        row["cuda_4stream_images_per_second"] = float(batch[name]["images_per_second"])
        result.append(row)
    return result


def table(headers, records):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in records])


def selected_families(data):
    return [max((r for r in data if r["family"] == family), key=lambda r: r["val_ssim"])
            for family in dict.fromkeys(r["family"] for r in data)]


def make_figures(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    output.mkdir(parents=True, exist_ok=True)
    selected = sorted(selected_families(data), key=lambda r: r["test_ssim"], reverse=True)
    y = np.arange(len(selected))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.65, 1]}, layout="constrained")
    paths = [("python_same_cpu_session_mean_ms", "Python, CPU session", "#3274a1"),
             ("cpp_exact_8_mean_ms", "C++ reference order, 8 OpenCV threads", "#e1812c"),
             ("cuda_host_roundtrip_ms", "CUDA, host round trip", "#3a923a"),
             ("cuda_device_single_ms", "CUDA, GPU-resident single", "#9467bd")]
    for offset, (key, label, color) in zip(np.linspace(-0.27, 0.27, 4), paths):
        axes[0].barh(y + offset, [r[key] for r in selected], height=0.16, color=color, label=label)
    axes[0].set(yticks=y, yticklabels=[r["method"] for r in selected], xscale="log", xlabel="Mean ms/image (log scale)", title="(a) Measured rendering paths")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.18)
    axes[0].set_axisbelow(True)
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -.15), ncol=2)
    for r, position in zip(selected, y):
        axes[1].plot([r["val_ssim"], r["test_ssim"]], [position, position], color=".75", lw=1)
    for key, label, color, marker in [("val_ssim", "Validation", "#3274a1", "o"),
                                      ("test_ssim", "Test", "#3a923a", "s")]:
        axes[1].scatter([r[key] for r in selected], y, color=color, marker=marker, label=label, zorder=3)
    axes[1].set(xlabel="Image-mean SSIM", title="(b) Reconstruction quality", xlim=(0.46, .89))
    axes[1].grid(axis="x", alpha=0.18)
    axes[1].tick_params(axis="y", left=False)
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -.15), ncol=2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "implementation_comparison.png", dpi=240)
    fig.savefig(output / "implementation_comparison.pdf")
    plt.close(fig)
    keys = ["python_same_cpu_session_mean_ms", "cpp_exact_1_mean_ms", "cpp_exact_8_mean_ms", "cuda_device_single_ms", "cuda_host_roundtrip_ms"]
    labels = ["Python\nCPU session", "C++ exact\n1 thread", "C++ exact\n8 threads", "CUDA\nGPU resident", "CUDA\nhost round trip"]
    values = np.array([[r[k] for k in keys] for r in data])
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(9.3, 9), layout="constrained")
    heat = ax.imshow(values, norm=LogNorm(values.min(), values.max()), cmap="viridis_r", aspect="auto")
    ax.set(xticks=range(len(keys)), xticklabels=labels, yticks=range(len(data)), yticklabels=[r["method"] for r in data], title="All 18 settings: mean rendering time (ms/image)")
    for i in range(len(data)):
        for j in range(len(keys)):
            ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", color="white" if heat.norm(values[i,j]) > .52 else "black", fontsize=9)
    fig.colorbar(heat, ax=ax, fraction=.035, pad=.025, label="Mean ms/image (log color scale)")
    fig.savefig(output / "all_algorithm_latency.png", dpi=240)
    fig.savefig(output / "all_algorithm_latency.pdf")
    plt.close(fig)


def create_report(results, output, figures=True):
    data = collect(results)
    output.mkdir(parents=True, exist_ok=True)
    write(output / "tables/all_algorithms.csv", data)
    long = []
    for backend in ["python", "native", "cuda"]:
        for r in read(results / backend / "latency_summary.csv"):
            long.append({"experiment": backend, "engine": r.get("engine", backend), "measurement": r.get("mode", "cpu_wall_ms"),
                         "method": r["method"], "split": r["split"], "samples": r["samples"],
                         **{key: r[key] for key in ["mean_ms", "median_ms", "p95_ms", "std_ms", "images_per_second"]}})
    write(output / "tables/latency_all_profiles.csv", long)
    write(output / "tables/quality_all_profiles.csv", [
        {"experiment": backend, "engine": r.get("engine", "python"), **{key: r[key] for key in ["split", "method", "family", "images", "ssim_mean", "ssim_std", "psnr_db_mean", "mae_mean", "rmse_mean", "correlation_mean", "hist_w1_graylevels_mean"]}}
        for backend in ["python", "native", "cuda"] for r in read(results / backend / "quality_summary.csv")])
    best = max(data, key=lambda r: r["val_ssim"])
    fused = [r for r in read(results / "native/parity_summary.csv") if r["engine"] == "cpp_fast_1"]
    changed = sum(int(r["different_pixels"]) for r in fused)
    pixels = sum(int(r["pixels"]) for r in fused)
    maxdiff = max(int(r["max_abs_difference"]) for r in fused)
    native_quality = read(results / "native/quality_summary.csv")
    nq = {(r["engine"], r["split"], r["method"]): float(r["ssim_mean"]) for r in native_quality}
    fused_ssim_delta = max(abs(nq[("cpp_fast_1", split, r["method"])] - nq[("cpp_exact_1", split, r["method"])]) for r in data for split in ["val", "test"])
    cpu_table = table(["Setting", "Python original", "Python CPU-session", "C++ exact/1", "Exact/8", "Fused/1", "Fused/8"],
                      [[r["method"], *[f"{r[k]:.4f}" for k in ["python_original_ms", "python_same_cpu_session_mean_ms", "cpp_exact_1_mean_ms", "cpp_exact_8_mean_ms", "cpp_fast_1_mean_ms", "cpp_fast_8_mean_ms"]]] for r in data])
    gpu_table = table(["Setting", "Resident single", "Resident queued", "GPU transfer interval", "Host round trip", "4-stream amortized"],
                      [[r["method"], *[f"{r[k]:.4f}" for k in ["cuda_device_single_ms", "cuda_device_amortized_ms", "cuda_gpu_roundtrip_ms", "cuda_host_roundtrip_ms", "cuda_4stream_amortized_ms"]]] for r in data])
    quality_table = table(["Family", "Validation-selected setting", "Val SSIM", "Test SSIM", "Test PSNR (dB)", "Test MAE"],
                          [[r["family"], r["method"], f"{r['val_ssim']:.4f}", f"{r['test_ssim']:.4f}", f"{r['test_psnr_db']:.2f}", f"{r['test_mae']:.2f}"] for r in selected_families(data)])
    text = f"""# FLIR AGC benchmark: Python, native C++/OpenCV and CUDA

## Main findings

Plateau HE with histogram cap 1000 and 0.1% tails (`{best['method']}`) is selected by the highest validation mean SSIM, **{best['val_ssim']:.4f}**. It reaches test SSIM **{best['test_ssim']:.4f}**, mean per-image PSNR **{best['test_psnr_db']:.2f} dB**, and MAE **{best['test_mae']:.2f}** on the 0–255 scale.

For this setting, Python takes **{best['python_same_cpu_session_mean_ms']:.4f} ms/image** in the native comparison session. Reference-order C++ takes **{best['cpp_exact_1_mean_ms']:.4f} ms** with one OpenCV thread ({best['python_same_cpu_session_mean_ms']/best['cpp_exact_1_mean_ms']:.2f}× faster) and **{best['cpp_exact_8_mean_ms']:.4f} ms** with eight OpenCV threads. CUDA takes **{best['cuda_device_single_ms']:.4f} ms** with input/output on the GPU, or **{best['cuda_host_roundtrip_ms']:.4f} ms** for the complete CPU-memory-to-CPU-memory wrapper. The resident queued-replay cost is **{best['cuda_device_amortized_ms']:.4f} ms/replay**; it is an amortized microbenchmark, not individual-request latency.

**The GPU does not win every host-image comparison.** Its plateau host round trip is slower than the preceding exact C++/8 result. CUDA is most useful when input and downstream processing remain on the device; transfer and wrapper costs can outweigh compute savings for 640×512 global mappings. CLAHE benefits more from GPU parallelism. CPU and CUDA sessions were recorded separately: their differences are descriptive and are not a newly controlled simultaneous hardware comparison.

Reference-order C++ and CUDA match Python with **zero differing pixels across all 3,295 images × 18 settings per backend**. Fused C++ is a separate implementation: **{changed:,} differing pixels ({100*changed/pixels:.8f}%)**, maximum difference **{maxdiff}** levels, and maximum absolute split-mean SSIM change **{fused_ssim_delta:.8f}**. Fused output must not be called bit-exact. One-vs-eight-thread and CUDA single-vs-four-stream outputs match on all 128 timing images for each setting.

## Dataset and experimental protocol

The public FLIR ADAS v2 validation/test files are matched by exact TIFF/JPEG stem and equal dimensions. The audit includes **1,144 validation + 2,151 test pairs**, all **640×512**. It excludes **1,598 test TIFFs stored as 8-bit**. Eligible inputs are single-channel uint16 TIFFs with observed values exceeding 255 and not exceeding 16,383. This checks storage and decoded range; it does not prove that every input contains 14 effective bits. The train split is unused. The dataset and vendor image examples are not redistributed.

Every method produces uint8 grayscale without resizing, registration, fitting against the target, or access to target JPEG pixels. Settings are fixed in `configs/default.json`; the test set does not select parameters. Similarity is image-mean Gaussian SSIM (11×11, σ=1.5, reflected borders over the full image), mean per-image PSNR, MAE, RMSE, Pearson correlation, and histogram Wasserstein-1 distance. This SSIM convention differs from valid-window MATLAB SSIM. Sequence-macro values and image standard deviations are retained in the original summaries. Correlated video frames and unequal sequence sizes limit interpretation of frame means. Similarity to a JPEG is not detection accuracy or temperature accuracy.

FLIR identifies processed JPEGs as AGC outputs but does not disclose the exact dataset settings [1]. These stateless software renderers approximate documented families; even the closest result does **not** identify or reproduce proprietary camera firmware, temporal filtering, ACE, SSO or DDE [2].

## Implementations and timing boundaries

| Implementation | Work performed | Important controls |
|---|---|---|
| Python | NumPy preprocessing/histograms; OpenCV HE8 and CLAHE | One OpenCV thread; OpenCL disabled; repeated temporary/output allocation |
| C++ reference order | Native histograms/LUTs and persistent workspace; OpenCV HE8/CLAHE | Exact reference arithmetic; profiles with 1/8 OpenCV threads; custom global loops remain single-threaded |
| C++ fused | Native statistics plus fused scale/offset conversions | 1/8 OpenCV threads; changed rounding/output checked independently |
| CUDA | CUB statistics/scans; custom point mappings and CLAHE; persistent buffers; CUDA Graphs | No fast math or FMA contraction; pinned staging; separate resident/transfer/host timing; four streams |

For each setting/profile, **64 validation + 64 test frames**, five repeats and warm-up produce **640 calls**; frame/method order is seeded and randomized. The Python pipeline is re-measured in the C++ timing session and is the baseline for C++ speedups. Original Python-session values are preserved in the first CPU column. CUDA uses exactly the same hashed subset in a later separate session. File I/O, decoding, scoring, compilation, context creation and workspace initialization are excluded. Each normal host call includes a newly allocated output array, bridge overhead and all rendering work; persistent internal workspace is reused and image statistics are recomputed each call.

CUDA resident-single timing is one compute-graph launch bracketed by events, after upload. It can include stream dispatch/idle gaps. Resident-queued timing is 16 queued compute replays on the same uploaded input divided by 16, not a batch of distinct frames. The GPU transfer interval includes H2D + compute + D2H under CUDA events, excluding CPU staging/allocation. Host round-trip timing includes allocation, Python/ctypes, pageable/pinned CPU copies, both transfers, compute and synchronization; that wall-clock call has no CUDA timer instrumentation. Four-stream groups process four distinct frames: 32 groups × five repeats = **160 batch observations/setting**. Batch time divided by four is amortized cost, not response latency.

Eight OpenCV threads accelerate supported stages such as CLAHE. They do not make custom C++ histogram/LUT loops eight-threaded; differences between those profiles for purely custom methods reflect timing variability. No fixed clocks, CPU affinity, power limits or isolated display scheduling were imposed. No energy-efficiency result is claimed.

## Reconstruction results

The following settings are chosen **within each family by validation SSIM**. Exact C++ and CUDA share these quality values. All settings/profiles, including fused quality changes, are in [quality_all_profiles.csv](tables/quality_all_profiles.csv).

{quality_table}

## CPU speed for every algorithm

All entries are mean **ms/image**. `exact` means reference-order C++; `fused` uses combined conversions. Thread counts refer to the OpenCV pool. Original Python timing and re-measured Python timing are separate columns; use the latter for within-session CPU comparisons.

{cpu_table}

## CUDA speed for every algorithm

All entries are **ms/image**, with the timing boundaries defined above. Resident queued and four-stream values are amortized. Do not compare resident compute directly with a transfer-inclusive CPU response and call the ratio an application speedup.

{gpu_table}

The [all_algorithms.csv](tables/all_algorithms.csv) file combines all 18 settings, quality, CPU profiles, CUDA timing paths, p95 values and images/s. [latency_all_profiles.csv](tables/latency_all_profiles.csv) preserves val/test/pooled mean, median, p95, standard deviation and throughput. Throughput is 1000/mean ms; four-stream throughput is 4000/mean batch ms. These are observed serial/batch rates, not detector FPS.

## Figures

![Selected-family implementation comparison](figures/implementation_comparison.png)

Figure 1. Validation-selected settings from each AGC family, ordered by descriptive test SSIM. (a) Python, reference-order C++/8, GPU-resident CUDA and host-round-trip CUDA latency; the axis is logarithmic. (b) Validation and test reconstruction quality, identical for Python, reference-order C++ and CUDA. Connectors join the two splits and are not uncertainty intervals. CPU and GPU timings are from separate sessions; GPU residency has a different boundary and is shown separately.

![All-setting timing matrix](figures/all_algorithm_latency.png)

Figure 2. All 18 fixed settings, using identical parameters across backends. Cells give mean ms/image; colors use a shared logarithmic scale. GPU-resident and host-round-trip columns measure different paths. These measurements cover native 640×512 frames only.

## Hardware, packages and build

Recorded CPU: **Intel Core i7-13700K**, 16 physical cores / 24 logical processors, approximately **31.8 GiB RAM**, Windows 11 build 26300. GPU: **NVIDIA GeForce RTX 4070 Ti SUPER**, 16 GB, compute capability **8.9**, 66 SMs, NVIDIA driver **610.88**, WDDM. This is the AGC experiment hardware; it does not describe a different detector experiment.

Recorded Python **3.12.14**; NumPy **2.5.4**, OpenCV Python **5.0.0.93** (library 5.0.0), Pillow **13.0.0**, matplotlib **3.11.2**, psutil **7.2.2**. All direct/transitive versions are pinned in `requirements-lock.txt`. C++ uses the official OpenCV **5.0.0** SDK, C++17, MSVC **19.43**, Release `/O2 /Oi /fp:precise /arch:AVX2`; wheel/SDK are different binary builds. CUDA uses NVCC/Toolkit **12.4.131**, runtime API **12040**, SM89, `-O3 --fmad=false --prec-div=true --prec-sqrt=true`; the driver's advertised API **13030** is not the compiler version.

CUDA 12.4 predates MSVC 19.43. The recorded build explicitly used `--allow-unsupported-compiler` and `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`. That combination passed tests and full parity, but is not an officially supported NVIDIA compiler pairing. The setup makes this an explicit optional override; a compatible supported toolkit/compiler is preferable. The custom CUDA renderer does not require an OpenCV CUDA build. CUB comes with the separately installed toolkit [4–6].

## Work estimates and deployment

Let N=W×H, B=16,384 histogram bins and G be the CLAHE tile count. Global histogram methods are **O(N+B)**, fixed manual mapping and auto-bright are **O(N)**, and this CLAHE pipeline is **O(N+B+256G)**, including its initial 14-bit percentile normalization. Native LUT application trades per-pixel arithmetic for one lookup after a B-entry preparation. CUDA parallelizes the image statistics, scans and mappings; CPU–GPU transfers still scale with N. Complexity files describe conceptual passes/bin visits, not exact CPU instructions, FLOPs or GPU launch counts (CUB primitives can launch multiple kernels).

Each backend retains `complexity*.csv`, analytical models at five resolutions, environment/build information, input manifests, raw timings and compressed per-image quality/parity. Only 640×512 was timed; the other dimensions are modeled. CUDA `workspace.csv` reports explicit device and pinned buffers, excluding CUDA context/driver/graph overhead, so it is not peak GPU memory. Read [the reproduction guide](../README.md) for setup and integrity checks.

For a CPU-hosted application, benchmark the C++ renderer and full CUDA host path on the target device. For a GPU-resident detector, use the CUDA device-pointer/stream API, keep AGC output on-device and separately measure normalization/channel expansion/inference. The present timings exclude those YOLO steps. Accuracy, quality and resource requirements should choose the deployment; the fastest measured setting is not necessarily the closest vendor rendering.

## References

1. Teledyne FLIR, [Free ADAS thermal dataset: formats and access](https://oem.flir.com/solutions/automotive/adas-dataset-form/).
2. FLIR, [Tau 2 Product Specification, Rev. 141](https://flir.custhelp.com/ci/fattach/get/107854/0/filename/102-PS242-40-Tau2ProductSpec_Rev141.pdf), §3.3.2.6: plateau, linear histogram, manual and auto-bright modes. These mode descriptions motivate the software variants; exact firmware equivalence is not claimed.
3. OpenCV, [histogram equalization/CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html) and [CLAHE source/license](https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/clahe.cpp).
4. NVIDIA, [CUB parallel primitives](https://nvidia.github.io/cccl/cub/).
5. NVIDIA, [CUDA 12.4.1 event API](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__EVENT.html).
6. NVIDIA, [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
7. Wang, Bovik, Sheikh and Simoncelli, [Image Quality Assessment: From Error Visibility to Structural Similarity](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf), IEEE TIP 13(4), 600–612, 2004. DOI: 10.1109/TIP.2003.819861.
"""
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    if figures:
        make_figures(data, output / "figures")
    print("Generated comparison:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/published")
    parser.add_argument("--output", type=Path, default=ROOT / "docs")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    create_report(args.results.resolve(), args.output.resolve(), not args.no_figures)


if __name__ == "__main__":
    main()
