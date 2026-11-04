#!/usr/bin/env python3
"""Check that this machine matches the environment the results were produced on, and say what is missing.

Usage: python3 env/check_env.py          (no GPU work)
       python3 env/check_env.py --gpu    (also runs the prototype's interleaving correctness test on 8 GPUs, ~1 min;
                                          refuses if other GPU processes are running)
Exit code 0 if every required item is OK.
"""
import ctypes
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECT = {  # what the results in results/ were produced with (2026-11)
    "gpu": "8x NVIDIA A100-SXM4-80GB, NVSwitch (NV12 between every pair)",
    "driver": "615.71.10",
    "cuda": "13.4 (nvcc at /usr/local/cuda)",
    "nccl": "2.32.3 (system libnccl2 / libnccl-dev)",
    "mpi": "Open MPI 4.1.9a1",
}
rows, failed = [], []


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def check(name, ok, found, hint="", required=True):
    rows.append((("OK" if ok else ("MISSING" if required else "optional")), name, found, "" if ok else hint))
    if required and not ok:
        failed.append(name)


# --- hardware and system ---------------------------------------------------------------------------------
rc, out = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
gpus = [l for l in out.splitlines() if l.strip()] if rc == 0 else []
check("GPUs", len(gpus) == 8 and all("A100" in g for g in gpus), f"{len(gpus)}x {gpus[0] if gpus else '-'}",
      f"expected {EXPECT['gpu']}")
check("driver", bool(gpus) and gpus[0].split(",")[-1].strip() == EXPECT["driver"],
      gpus[0].split(",")[-1].strip() if gpus else "-", f"results used {EXPECT['driver']} (other versions may work)",
      required=False)
rc, out = sh("nvidia-smi topo -m")
nv12 = out.count("NV12")
check("NVLink topology", nv12 >= 56, f"{nv12} NV12 links in topo matrix", "expected NV12 between all 8 GPUs")
cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
rc, out = sh(f"{cuda_home}/bin/nvcc --version")
check("nvcc", rc == 0 and "release 13." in out, out.splitlines()[-1] if rc == 0 else "not found",
      f"install CUDA 13.x toolkit at {cuda_home} (results used 13.4)")
try:
    lib = ctypes.CDLL("libnccl.so.2")
    v = ctypes.c_int()
    lib.ncclGetVersion(ctypes.byref(v))
    nccl = f"{v.value // 10000}.{v.value // 100 % 100}.{v.value % 100}"
except OSError:
    nccl = "not found"
check("system NCCL", nccl == "2.32.3", nccl, "apt install libnccl2=2.32.3-1+cuda13.4 libnccl-dev=2.32.3-1+cuda13.4")
check("nccl.h", Path("/usr/include/nccl.h").exists(), "/usr/include/nccl.h", "install libnccl-dev")
rc, out = sh("mpirun --version")
check("Open MPI", rc == 0, out.splitlines()[0] if rc == 0 else "not found", "needed by nccl-tests (MPI=1 build)")
for tool in ("uv", "cmake", "git", "curl"):
    check(tool, shutil.which(tool) is not None, shutil.which(tool) or "not found",
          "uv: curl -LsSf https://astral.sh/uv/install.sh | sh" if tool == "uv" else f"install {tool}")
check("nsys", Path(f"{cuda_home}/bin/nsys").exists(), f"{cuda_home}/bin/nsys", "only M10/M11 use it", required=False)

# --- Python environments -------------------------------------------------------------------------------
pins = {".venv": ("requirements-analysis.txt", "import pandas, matplotlib; print(pandas.__version__)"),
        ".venv-vllm": ("requirements-vllm.txt",
                       "import torch, vllm, flashinfer; print(torch.__version__, vllm.__version__, flashinfer.__version__)"),
        ".venv-titan": ("requirements-titan.txt", "import torch, torchtitan; print(torch.__version__)")}
for venv, (req, probe) in pins.items():
    py = ROOT / venv / "bin/python"
    if not py.exists():
        check(venv, False, "not created", f"bash env/setup_python.sh  ({req})")
        continue
    rc, frozen = sh(f"uv pip freeze --python {py} 2>/dev/null")
    want = {l.strip() for l in open(ROOT / req) if l.strip() and not l.startswith("#")}
    have = {l.strip() for l in frozen.splitlines() if l.strip()}
    diff = sorted(want - have)
    rc2, out = sh(f"{py} -c '{probe}'")
    check(venv, not diff and rc2 == 0, out.splitlines()[-1] if rc2 == 0 and out else out[-120:],
          f"{len(diff)} pins differ from {req}, e.g. {diff[:3]}; delete {venv} and rerun env/setup_python.sh")

# --- external tools ------------------------------------------------------------------------------------
tp = ROOT / "third_party"
for name, path, used in [("nccl-tests", "nccl-tests/build/all_reduce_perf", "M0-M7"),
                         ("mscclpp", "mscclpp/build/lib/libmscclpp_nccl.so", "M6, M7"),
                         ("low-latency-nccl", "low-latency-nccl/build/lib/libnccl.so.2", "M7"),
                         ("nccl-tests-llnccl", "nccl-tests-llnccl/build/all_reduce_perf", "M7"),
                         ("lmdeploy test_comm", "lmdeploy/build/bin/test_comm", "M9, M10"),
                         ("lmdeploy test_comm (forced)", "lmdeploy/build_force/bin/test_comm", "M9, M10")]:
    check(name, (tp / path).exists(), f"third_party/{path}",
          f"bash env/build_third_party.sh (needed for {used})", required=False)

# --- models and assets -----------------------------------------------------------------------------------
hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
for repo, rev in [("Qwen--Qwen2.5-32B-Instruct", "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"),
                  ("TinyLlama--TinyLlama-1.1B-Chat-v1.0", "fe8a4ea1ffedaf415f4da2f062534de366a451e6")]:
    snap = hub / f"models--{repo}" / "snapshots" / rev
    check(repo.replace("--", "/"), snap.exists(), str(snap), "bash env/get_models_assets.sh (only M5)", required=False)
for a in ("assets/c4_test/data.json", "assets/tokenizer_tinyllama/tokenizer.json"):
    check(a, (ROOT / a).exists(), a, "bash env/get_models_assets.sh (only M5 training)", required=False)

# --- optional GPU smoke test -----------------------------------------------------------------------------
if "--gpu" in sys.argv:
    rc, out = sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader")
    if out.strip():
        check("GPU smoke test", False, "skipped", "other GPU processes are running (rule R13: do not disturb them)")
    else:
        ext = Path("/tmp") / f"comm_opt_check_{os.getpid()}"
        cmd = (f"CUDA_HOME={cuda_home} PATH={cuda_home}/bin:$PATH timeout 600 {ROOT}/.venv-vllm/bin/torchrun "
               f"--nproc-per-node 8 --master-port 29699 {ROOT}/exp/M16_epoch_fix/test_interleave.py "
               f"--src {ROOT}/exp/M17_epoch_ab/fused_ar_norm_nf.cu --name check_fused_ar_norm --ext-dir {ext} "
               f"--mode graph --calls 20 --out {ext}/result.json")
        ext.mkdir(parents=True, exist_ok=True)
        rc, out = sh(cmd)
        last = [l for l in out.splitlines() if l.startswith("done:")]
        check("GPU smoke test (prototype, 20 interleaved calls)", rc == 0 and bool(last) and "20/20" in last[0],
              last[0] if last else out[-200:], "see output above; needs .venv-vllm and nvcc")

w = max(len(r[1]) for r in rows)
for status, name, found, hint in rows:
    print(f"{status:8s} {name:<{w}}  {found}" + (f"\n{'':8s} {'':<{w}}  -> {hint}" if hint else ""))
print(f"\n{'ALL REQUIRED ITEMS OK' if not failed else 'MISSING REQUIRED: ' + ', '.join(failed)}")
sys.exit(1 if failed else 0)
