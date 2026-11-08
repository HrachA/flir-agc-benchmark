import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class SuiteTests(unittest.TestCase):
    def test_all_backend_specs_and_locks_agree(self):
        expected = (ROOT / "configs/default.json").read_bytes()
        lock = (ROOT / "requirements-lock.txt").read_bytes()
        for backend in ["python", "native", "cuda"]:
            self.assertEqual((ROOT / "implementations" / backend / "configs/default.json").read_bytes(), expected)
            self.assertEqual((ROOT / "implementations" / backend / "requirements-lock.txt").read_bytes(), lock)

    def test_comparison_uses_same_cpu_session_baseline(self):
        data = module("report").collect(ROOT / "results/published")
        self.assertEqual(len(data), 18)
        selected = max(data, key=lambda r: r["val_ssim"])
        self.assertEqual(selected["method"], "plateau_p1000")
        self.assertAlmostEqual(selected["test_ssim"], 0.8625928197837)
        self.assertGreater(selected["python_same_cpu_session_mean_ms"], selected["cpp_exact_1_mean_ms"])
        self.assertGreater(selected["cuda_host_roundtrip_ms"], selected["cuda_device_single_ms"])

    def test_runner_paths_and_cuda_only_batch(self):
        runner = module("run_experiments")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp).resolve()
            work = list(runner.commands(["python", "native", "cuda"], directory, directory / "output", "all", True))
            self.assertEqual(len(work), 3)
            for cwd, cmd in work:
                self.assertTrue(cwd.is_absolute())
                self.assertIn(str(directory), cmd)
            self.assertNotIn("--no-cache", work[0][1])
            self.assertIn("--no-cache", work[2][1])
            with self.assertRaises(ValueError):
                list(runner.commands(["native"], directory, directory / "output", "batch"))

    def test_original_cache_source_hashes_survive_consolidation(self):
        import hashlib
        for backend, package in [("native", "agc_native"), ("cuda", "agc_cuda")]:
            base = ROOT / "implementations" / backend
            provenance = json.loads((base / "reference_results/provenance.json").read_text())
            for filename, recorded in [("reference_methods.py", "methods.py"), ("metrics.py", "metrics.py")]:
                self.assertEqual(hashlib.sha256((base / package / filename).read_bytes()).hexdigest(), provenance[recorded])


if __name__ == "__main__":
    unittest.main()
