"""Shared helpers for the HOOPS AI 1.1 CPU vs GPU benchmark harness.

Nothing in here imports hoops_ai, so it is safe to use from the probe phase
before a license is available.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BENCH_ROOT.parent                      # SDK install root (parent of this repo)
RESULTS_CSV = BENCH_ROOT / "results" / "results.csv"

# Default CAD formats to pick up when scanning a folder, restricted to
# formats HOOPS Exchange documents as carrying full B-rep (not just
# mesh/tessellation) -- see
# https://docs.techsoft3d.com/hoops/exchange/start/supported-formats.html
# Deliberately excluded: mesh-only formats (STL, 3MF, OBJ, FBX, glTF/GLB,
# VRML, U3D, COLLADA, 3DS) and formats the docs list as mixed/limited B-rep
# (DWG/DXF, DGN, IFC, PDF, Revit, Navisworks, I-deas, VDA, QIF) -- HOOPS AI's
# embedding pipeline needs an actual B-rep to encode, not a mesh. This is
# still not guaranteed to match exactly what your licensed HOOPS Exchange
# build can import (some formats need separate CAD interop licenses), and it
# is not necessarily the same list hoops_ai.storage.CADFileRetriever uses
# internally. Override with --extensions if it doesn't match your data /
# your license.
CAD_EXTENSIONS = {
    ".stp", ".step", ".stpz",                          # STEP
    ".stpx", ".stpxz",                                 # STEP/XML
    ".igs", ".iges",                                   # IGES
    ".x_t", ".x_b", ".xmt", ".xmt_txt", ".xmt_bin",    # Parasolid
    ".sat", ".sab",                                    # ACIS
    ".catpart", ".catproduct", ".catshape",            # CATIA V5
    ".catdrawing", ".cgr",                             # CATIA V5 (drawing / graphics rep)
    ".3dxml",                                          # CATIA V6
    ".model", ".session", ".dlv", ".exp",              # CATIA V4
    ".sldprt", ".sldasm",                              # SolidWorks
    ".ipt", ".iam",                                    # Autodesk Inventor
    ".prt", ".asm", ".neu", ".xas", ".xpr",            # NX / Creo / Pro-ENGINEER (.prt/.asm overlap between these and Solid Edge)
    ".par", ".psm", ".pwd",                            # Solid Edge
    ".jt",                                             # JT (import only)
    ".3dm",                                            # Rhino
    ".prc",                                            # PRC
}


def discover_cad_files(source_dir: Path, extensions: set[str] | None = None) -> list[str]:
    """Recursively scan source_dir for CAD files, sorted for determinism.

    Used whenever a script is pointed at a folder instead of a pre-built file
    list: every subfolder is included, and the sort order makes repeated
    scans of an unchanged folder produce identical, reproducible ordering.
    """
    source_dir = Path(source_dir)
    exts = extensions if extensions is not None else CAD_EXTENSIONS
    return sorted(
        str(p) for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def parse_extensions(raw: str | None) -> set[str] | None:
    """Parse a comma-separated --extensions value into a lowercase, dot-
    prefixed set, or None if raw is falsy (caller should fall back to
    CAD_EXTENSIONS)."""
    if not raw:
        return None
    exts = set()
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        exts.add(item if item.startswith(".") else f".{item}")
    return exts or None


def freeze_filelist(files: list[str], out_path: Path) -> Path:
    """Write a discovered file list to disk so it can be reused verbatim
    (e.g. passed to a second step, or inspected/audited later)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(files) + "\n", encoding="utf-8")
    return out_path

CSV_FIELDS = [
    "run_id",          # unique per measurement
    "timestamp",
    "phase",           # 0..5
    "step",            # dataprep | training | indexing | probe
    "env_tag",         # cpu1.1 | gpu1.1
    "accelerator",     # cpu | gpu | -   (step 2/3 only)
    "n_files",
    "param_name",      # max_workers | num_workers | batch_size | max_epochs | ...
    "param_value",
    "extra",           # JSON blob of the remaining knobs
    "wall_s",          # total wall clock of the measured call
    "sub_timings",     # JSON blob: per-task / per-phase breakdown
    "throughput",      # files/s or epochs/s where meaningful
    "peak_rss_mb",
    "peak_gpu_mb",
    "n_ok",
    "n_failed",
    "status",          # OK | FAILED | SKIPPED_BUDGET
    "note",
]


# --------------------------------------------------------------------------
# .env / license
# --------------------------------------------------------------------------
def load_dotenv() -> dict:
    """Load KEY=VALUE pairs from the first .env found, into os.environ."""
    found = {}
    for candidate in (BENCH_ROOT / ".env", REPO_ROOT / ".env",
                      REPO_ROOT / "CPU1.1" / ".env", REPO_ROOT / "GPU1.1" / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            # inline comment
            if " #" in val:
                val = val.split(" #", 1)[0].strip()
            found[key] = val
            os.environ.setdefault(key, val)
        break
    return found


def check_hoops_ai_importable() -> None:
    """Fail fast with a clear message instead of a raw ModuleNotFoundError
    traceback when a script is run with the wrong Python interpreter.

    hoops_ai is installed inside your CPU-build / GPU-build venv, not
    globally -- there is no "activate" step in this repo's workflow, you run
    each script directly with that venv's python.exe. Call this first thing
    in main(), before anything else. Uses importlib.util.find_spec, which
    only checks importability without actually importing (and therefore
    without importing torch) -- safe to call before CUDA_VISIBLE_DEVICES
    tricks that must happen before torch loads.
    """
    if importlib.util.find_spec("hoops_ai") is not None:
        return
    sys.exit(
        f"hoops_ai is not importable from this Python interpreter:\n"
        f"  {sys.executable}\n"
        "hoops_ai is installed inside your CPU/GPU venv, not globally -- run "
        "this script with THAT venv's python.exe instead, e.g.:\n"
        "  <SDK root>\\CPU1.1\\.venv\\Scripts\\python.exe bench_step1_dataprep.py ...\n"
        "(Windows: local_env.ps1 / local_env.example.ps1 record where yours "
        "live. Linux: the CPU_PY / GPU_PY env vars used by run_heavy_batch.sh "
        "point at the same venvs.)")


def require_license() -> str:
    """Make sure HOOPS_AI_LICENSE is in the environment. Does NOT import hoops_ai.

    Use this when you only need the key present (e.g. so that worker processes
    inherit it). If you are going to call hoops_ai APIs in THIS process, you also
    need apply_license().
    """
    load_dotenv()
    key = os.environ.get("HOOPS_AI_LICENSE")
    if not key:
        sys.exit("HOOPS_AI_LICENSE is not set. Put it in benchmark\\.env or the process environment.")
    return key


def apply_license(validate: bool = True, silent: bool = True) -> str:
    """Actually activate the license in this process.

    Setting the environment variable is not enough: hoops_ai requires an explicit
    hoops_ai.set_license() call, otherwise the first API use raises
    "No HOOPS AI license configured".

    bench_step1 got away without this because it imports bench_tasks, which calls
    set_license() at module import time for the worker processes. bench_step2 and
    bench_step3 talk to hoops_ai directly and must call this themselves.

    Note this imports hoops_ai (and therefore torch), so anything that has to
    happen before torch loads - such as setting CUDA_VISIBLE_DEVICES - must be
    done first.
    """
    key = require_license()
    import hoops_ai
    hoops_ai.set_license(key, validate=validate, silent=silent)
    return key


# --------------------------------------------------------------------------
# hardware / software probe
# --------------------------------------------------------------------------
def probe_hardware() -> dict:
    info = {
        "hostname": platform.node(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_brand": platform.processor(),
        "logical_cores": os.cpu_count(),
        "physical_cores": None,
        "ram_gb": None,
        "gpus": [],
        "torch_version": None,
        "torch_cuda": None,
        "cuda_available": False,
        "hoops_ai_version": None,
        "disk_free_gb": None,
    }
    try:
        import psutil
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        pass
    try:
        info["disk_free_gb"] = round(shutil.disk_usage(str(REPO_ROOT)).free / 1024 ** 3, 1)
    except Exception:
        pass
    # Prefer a real physical-core count on Windows
    if not info["physical_cores"] and platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
                capture_output=True, text=True, timeout=60)
            info["physical_cores"] = int(out.stdout.strip())
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            info["gpus"] = [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except Exception:
        pass
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["torch_gpu_name"] = torch.cuda.get_device_name(0)
            info["torch_gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
            ok, detail = cuda_usable()
            info.update(detail)
            info["cuda_usable"] = ok
        else:
            info["cuda_usable"] = False
    except Exception as exc:
        info["torch_error"] = repr(exc)
        info["cuda_usable"] = False
    try:
        from importlib.metadata import version
        info["hoops_ai_version"] = version("hoops_ai")
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------
# Is the GPU actually usable, not merely present?
# --------------------------------------------------------------------------
def cuda_usable() -> tuple[bool, dict]:
    """Launch a real kernel and see whether it runs.

    torch.cuda.is_available() only checks that a driver and a CUDA runtime are
    present. It returns True even when the installed wheel contains no code for
    the device's compute capability, in which case every kernel launch fails
    with "no kernel image is available for execution on the device".

    This matters here: torch 2.9.1+cu130 ships sm_75 and newer, so a Pascal card
    (GTX 10xx, sm_61) reports available=True and then cannot train at all.
    """
    detail: dict = {}
    try:
        import torch
    except Exception as exc:
        return False, {"cuda_error": f"torch import failed: {exc!r}"}
    if not torch.cuda.is_available():
        return False, {"cuda_error": "torch.cuda.is_available() is False"}
    try:
        major, minor = torch.cuda.get_device_capability(0)
        detail["gpu_capability"] = f"sm_{major}{minor}"
        detail["torch_arch_list"] = list(torch.cuda.get_arch_list())
    except Exception:
        pass
    try:
        a = torch.randn(128, 128, device="cuda")
        b = (a @ a).sum().item()
        torch.cuda.synchronize()
        if b != b:  # NaN
            return False, {**detail, "cuda_error": "matmul produced NaN"}
        return True, detail
    except Exception as exc:
        msg = str(exc).strip().splitlines()[0] if str(exc) else repr(exc)
        detail["cuda_error"] = msg[:300]
        arches = detail.get("torch_arch_list") or []
        cap = detail.get("gpu_capability")
        if cap and arches and cap not in arches:
            detail["cuda_hint"] = (
                f"this wheel was built for {', '.join(arches)} but the GPU is {cap}; "
                f"install a torch build that includes {cap}")
        return False, detail


# --------------------------------------------------------------------------
# resource peaks
# --------------------------------------------------------------------------
def peak_rss_mb() -> float | None:
    """Current RSS of this process *and* its children, in MB.

    Note this is an INSTANTANEOUS reading. Called after a flow finishes it
    reports almost nothing, because the worker processes have already exited.
    Use PeakSampler when you need a real peak.
    """
    try:
        import psutil
        me = psutil.Process()
        total = me.memory_info().rss
        for child in me.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                pass
        return round(total / 1024 ** 2, 1)
    except Exception:
        return None


class PeakSampler:
    """Background thread that tracks peak memory across this process tree.

    Needed because the interesting number - how much RAM 16 encoder workers use
    at once - only exists while they are running. Sampling after flow.process()
    returns measures an empty process tree.

    Usage:
        with PeakSampler() as s:
            ...work...
        s.peak_rss_mb, s.peak_children, s.system_peak_pct
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.peak_rss_mb: float | None = None
        self.peak_children: int = 0
        self.system_peak_pct: float | None = None
        self._stop = None
        self._thread = None

    def _run(self):
        import psutil
        me = psutil.Process()
        while not self._stop.is_set():
            try:
                total = me.memory_info().rss
                kids = me.children(recursive=True)
                for c in kids:
                    try:
                        total += c.memory_info().rss
                    except Exception:
                        pass
                mb = total / 1024 ** 2
                if self.peak_rss_mb is None or mb > self.peak_rss_mb:
                    self.peak_rss_mb = round(mb, 1)
                self.peak_children = max(self.peak_children, len(kids))
                pct = psutil.virtual_memory().percent
                if self.system_peak_pct is None or pct > self.system_peak_pct:
                    self.system_peak_pct = pct
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        try:
            import threading
            import psutil  # noqa: F401
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception:
            self._thread = None
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
        return False

    def as_dict(self) -> dict:
        return {"peak_rss_mb": self.peak_rss_mb,
                "peak_child_procs": self.peak_children,
                "system_mem_peak_pct": self.system_peak_pct}


def peak_gpu_mb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    except Exception:
        pass
    return None


class Timer:
    """with Timer() as t: ...   -> t.elapsed (seconds, perf_counter)."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        return False


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
def write_row(row: dict, csv_path: Path | str | None = None) -> None:
    path = Path(csv_path) if csv_path else RESULTS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    clean = {k: row.get(k, "") for k in CSV_FIELDS}
    for k, v in clean.items():
        if isinstance(v, (dict, list)):
            clean[k] = json.dumps(v, ensure_ascii=False)
    with path.open("a", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(clean)
    print(f"[bench] recorded: {clean['step']} {clean['env_tag']} "
          f"{clean['param_name']}={clean['param_value']} "
          f"n={clean['n_files']} -> {clean['wall_s']}s ({clean['status']})", flush=True)


def make_run_id(step: str, env_tag: str, param_name: str, param_value, n_files) -> str:
    return f"{step}-{env_tag}-{param_name}{param_value}-n{n_files}-{int(time.time())}"


# --------------------------------------------------------------------------
# checkpoint resolution (GPU1.1 has no packages/ dir -> fall back to CPU1.1)
# --------------------------------------------------------------------------
CKPT_NAME = "ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt"


def resolve_checkpoint(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        sys.exit(f"checkpoint not found: {p}")
    env = os.environ.get("HOOPS_AI_CKPT")
    if env and Path(env).is_file():
        return Path(env)
    for base in ("GPU1.1", "CPU1.1", "V1.1", "."):
        p = REPO_ROOT / base / "packages" / "trained_ml_models" / CKPT_NAME
        if p.is_file():
            return p
    sys.exit(
        f"Could not find {CKPT_NAME}. Searched */packages/trained_ml_models/ under {REPO_ROOT}.\n"
        "Set HOOPS_AI_CKPT or pass --ckpt.")


if __name__ == "__main__":
    load_dotenv()
    info = probe_hardware()
    out = BENCH_ROOT / "results" / f"env_{os.environ.get('BENCH_ENV_TAG', 'unknown')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"\n[bench] written -> {out}")
