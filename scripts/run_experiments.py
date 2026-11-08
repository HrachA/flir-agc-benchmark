"""Run the independent backends sequentially from one repository/environment."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULES = {"python": "agc_benchmark", "native": "agc_native", "cuda": "agc_cuda"}


def commands(backends, dataset, output, stage, no_cache=False):
    for backend in backends:
        if stage == "batch" and backend != "cuda":
            raise ValueError("The batch stage is available only for CUDA")
        cmd = [sys.executable, "-m", MODULES[backend] + ".run",
               "--dataset-root", str(dataset), "--output", str(output / backend),
               "--stage", stage]
        if no_cache and backend != "python":
            cmd.append("--no-cache")
        yield ROOT / "implementations" / backend, cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=MODULES, default=list(MODULES))
    parser.add_argument("--output", type=Path, default=Path("results/reproduction"))
    parser.add_argument("--stage", choices=["all", "audit", "latency", "batch", "quality", "report"], default="all")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the exact commands without executing")
    args = parser.parse_args()
    dataset, output = args.dataset_root.resolve(), args.output.resolve()
    if not dataset.is_dir():
        parser.error("Dataset directory does not exist: " + str(dataset))
    published = (ROOT / "results/published").resolve()
    if output == published or published in output.parents:
        parser.error("Use a new results directory; the published archive is read-only")
    try:
        work = list(commands(args.backends, dataset, output, args.stage, args.no_cache))
    except ValueError as exc:
        parser.error(str(exc))
    for cwd, cmd in work:
        print("\nWorking directory:", cwd, flush=True)
        print(subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=cwd, check=True)
    if args.stage == "all" and set(args.backends) == set(MODULES) and not args.dry_run:
        subprocess.run([sys.executable, str(ROOT / "scripts/report.py"),
                        "--results", str(output), "--output", str(output / "comparison")], check=True)


if __name__ == "__main__":
    main()
