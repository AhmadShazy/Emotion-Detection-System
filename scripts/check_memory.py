"""
scripts/check_memory.py
========================
Measures RAM consumption of the Humanoid Assistant project at three
key points:

    1. Baseline     — Python process before any imports
    2. Per-model    — how much RAM each model adds when loaded
    3. Runtime      — total footprint while the API is handling requests

Usage:
    # Static analysis (no server needed):
    python scripts/check_memory.py

    # Runtime analysis (while server is running in another terminal):
    python scripts/check_memory.py --runtime

Requirements:
    pip install psutil
"""

import os
import sys
import time
import argparse

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import psutil
except ImportError:
    print("❌ psutil not installed. Run: pip install psutil")
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_ram_mb() -> float:
    """Returns current process RSS memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def get_system_ram() -> dict:
    """Returns system-wide RAM stats."""
    vm = psutil.virtual_memory()
    return {
        "total_gb":     round(vm.total     / 1024 ** 3, 2),
        "available_gb": round(vm.available / 1024 ** 3, 2),
        "used_gb":      round(vm.used      / 1024 ** 3, 2),
        "percent":      vm.percent,
    }


def _bar(used_gb: float, total_gb: float, width: int = 40) -> str:
    """Renders a simple ASCII progress bar."""
    filled = int((used_gb / total_gb) * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {used_gb:.1f}/{total_gb:.1f} GB"


def _header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _row(label: str, before: float, after: float):
    delta = after - before
    bar   = "█" * min(int(delta / 10), 30)
    print(f"  {label:<28} {after:>7.1f} MB   (+{delta:>6.1f} MB)  {bar}")


# ── Static model memory analysis ──────────────────────────────────────────────

def measure_model_memory():
    _header("System RAM Status")
    sys_ram = get_system_ram()
    print(f"  Total RAM:      {sys_ram['total_gb']} GB")
    print(f"  Used:           {sys_ram['used_gb']} GB  ({sys_ram['percent']}%)")
    print(f"  Available:      {sys_ram['available_gb']} GB")
    print(f"  {_bar(sys_ram['used_gb'], sys_ram['total_gb'])}")

    _header("Per-Model RAM Consumption")
    print(f"  {'Model':<28} {'Total MB':>10}   {'Delta MB':>10}  Scale")
    print(f"  {'-'*60}")

    snapshots = {}

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline = get_ram_mb()
    snapshots["baseline"] = baseline
    print(f"\n  {'Baseline (Python only)':<28} {baseline:>7.1f} MB")

    # ── numpy + torch (shared by all models) ──────────────────────────────────
    before = get_ram_mb()
    import numpy as np
    after  = get_ram_mb()
    snapshots["numpy"] = after
    _row("numpy", before, after)

    before = get_ram_mb()
    import torch
    after  = get_ram_mb()
    snapshots["torch"] = after
    _row("torch", before, after)

    # ── RoBERTa ───────────────────────────────────────────────────────────────
    before = get_ram_mb()
    from transformers import pipeline as hf_pipeline
    roberta = hf_pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
    )
    after = get_ram_mb()
    snapshots["roberta"] = after
    _row("RoBERTa (go_emotions)", before, after)

    # ── faster-whisper tiny ───────────────────────────────────────────────────
    FASTER_WHISPER_CACHE = os.path.join(
        PROJECT_ROOT, "external", "faster_whisper"
    )
    before = get_ram_mb()
    from faster_whisper import WhisperModel
    fw_model = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8",
        download_root=FASTER_WHISPER_CACHE,
    )
    after = get_ram_mb()
    snapshots["faster_whisper"] = after
    _row("faster-whisper tiny", before, after)

    # ── Whisper base ──────────────────────────────────────────────────────────
    WHISPER_CACHE = os.path.join(PROJECT_ROOT, "external", "whisper")
    before = get_ram_mb()
    import whisper
    whisper_model = whisper.load_model("base", download_root=WHISPER_CACHE)
    after = get_ram_mb()
    snapshots["whisper"] = after
    _row("Whisper base", before, after)

    # ── SpeechBrain Wav2Vec2 ──────────────────────────────────────────────────
    SPEECHBRAIN_CACHE = os.path.join(PROJECT_ROOT, "external", "speechbrain")

    # Apply patches before SpeechBrain import
    import torchaudio
    import soundfile as sf

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"

    if not getattr(torchaudio, "_patched_by_ser_engine", False):
        def _custom_load(filepath, **kwargs):
            data, samplerate = sf.read(filepath)
            data = data.astype(np.float32)
            if data.ndim == 1:
                tensor = torch.from_numpy(data).unsqueeze(0)
            else:
                tensor = torch.from_numpy(data.transpose())
            return tensor, samplerate
        torchaudio.load = _custom_load
        torchaudio._patched_by_ser_engine = True

    import transformers
    if not hasattr(transformers, "AutoModelWithLMHead"):
        transformers.AutoModelWithLMHead = getattr(
            transformers, "AutoModelForCausalLM", transformers.AutoModel
        )

    import huggingface_hub
    if not getattr(huggingface_hub, "_patched_by_ser_engine", False):
        _orig = huggingface_hub.hf_hub_download
        def _p(*a, **kw):
            if "use_auth_token" in kw:
                kw["token"] = kw.pop("use_auth_token")
            return _orig(*a, **kw)
        huggingface_hub.hf_hub_download        = _p
        huggingface_hub._patched_by_ser_engine = True

    before = get_ram_mb()
    from speechbrain.inference.interfaces import foreign_class
    sb_model = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        savedir=SPEECHBRAIN_CACHE,
        run_opts={"device": "cpu"},
    )
    after = get_ram_mb()
    snapshots["speechbrain"] = after
    _row("SpeechBrain Wav2Vec2", before, after)

    # ── Summary ───────────────────────────────────────────────────────────────────
    _header("Summary")

    print(f"  Baseline Python process:       {snapshots['baseline']:>7.1f} MB")
    print(f"  torch + numpy overhead:        "
        f"{snapshots['torch'] - snapshots['baseline']:>7.1f} MB")
    print(f"  RoBERTa:                       "
        f"{snapshots['roberta'] - snapshots['torch']:>7.1f} MB")
    print(f"  faster-whisper tiny:           "
        f"{snapshots['faster_whisper'] - snapshots['roberta']:>7.1f} MB")
    print(f"  Whisper base:                  "
        f"{snapshots['whisper'] - snapshots['faster_whisper']:>7.1f} MB")
    print(f"  SpeechBrain Wav2Vec2:          "
        f"{snapshots['speechbrain'] - snapshots['whisper']:>7.1f} MB")
    print(f"  {'─'*45}")
    print(f"  Total (all models loaded):     "
        f"{snapshots['speechbrain']:>7.1f} MB  "
        f"({snapshots['speechbrain']/1024:.2f} GB)")

    sys_ram  = get_system_ram()
    pct_used = (snapshots["speechbrain"] / 1024) / sys_ram["total_gb"] * 100
    print(f"\n  This process uses {pct_used:.1f}% of your "
        f"{sys_ram['total_gb']} GB total RAM.")

    if snapshots["speechbrain"] > 3000:
        print("\n  ⚠️  WARNING: Over 3 GB — consider closing other apps "
            "while the API is running.")
    elif snapshots["speechbrain"] > 2000:
        print("\n  ⚠️  NOTICE: Over 2 GB — typical for this ML stack on CPU.")
    else:
        print("\n  ✅ RAM usage is within normal range.")


# ── Runtime analysis (attach to running uvicorn process) ──────────────────────

def measure_runtime_memory():
    """
    Finds the running uvicorn process and monitors its RAM usage
    every 2 seconds for 30 seconds.
    """
    _header("Runtime Memory Monitor")

    # Find uvicorn process
    uvicorn_procs = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "uvicorn" in cmdline and "api:app" in cmdline:
                uvicorn_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not uvicorn_procs:
        print("  ❌ No running uvicorn process found.")
        print("  Start the API first: uvicorn api:app --reload")
        print("  Then run: python scripts/check_memory.py --runtime")
        return

    print(f"  Found {len(uvicorn_procs)} uvicorn process(es).")
    print(f"\n  {'Time':>6}  {'PID':>8}  {'RSS MB':>10}  {'VMS MB':>10}")
    print(f"  {'-'*40}")

    samples = []
    try:
        for i in range(15):   # 15 samples × 2s = 30 seconds
            total_rss = 0
            total_vms = 0
            for proc in uvicorn_procs:
                try:
                    mem = proc.memory_info()
                    total_rss += mem.rss
                    total_vms += mem.vms
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            rss_mb = total_rss / 1024 ** 2
            vms_mb = total_vms / 1024 ** 2
            samples.append(rss_mb)

            print(f"  {i*2:>5}s  "
                  f"{'all':>8}  "
                  f"{rss_mb:>9.1f}  "
                  f"{vms_mb:>9.1f}")

            time.sleep(2)

    except KeyboardInterrupt:
        pass

    if samples:
        print(f"\n  Min RSS: {min(samples):.1f} MB")
        print(f"  Max RSS: {max(samples):.1f} MB")
        print(f"  Avg RSS: {sum(samples)/len(samples):.1f} MB")

        sys_ram = get_system_ram()
        avg_gb  = (sum(samples) / len(samples)) / 1024
        pct     = avg_gb / sys_ram["total_gb"] * 100
        print(f"\n  API uses ~{avg_gb:.2f} GB = {pct:.1f}% of your "
              f"{sys_ram['total_gb']} GB RAM at runtime.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Humanoid Assistant RAM profiler"
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Monitor a running uvicorn process instead of loading models here",
    )
    args = parser.parse_args()

    print("\n🔍 Humanoid Assistant — RAM Usage Profiler")

    if args.runtime:
        measure_runtime_memory()
    else:
        measure_model_memory()


if __name__ == "__main__":
    main()