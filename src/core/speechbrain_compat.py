"""
src/core/speechbrain_compat.py
==============================
The single definition of the compatibility patches SpeechBrain needs before it
can load in this project.

Why this module exists
----------------------
These patches were written out three times — once in ModelRegistry, once in
SEREngine, and a third, PARTIAL copy in scripts/download_models.py. The partial
copy was missing the LazyModule patch, and that omission was not theoretical:
with speechbrain 1.1.0 the downloader failed outright with

    ImportError: Lazy import of LazyModule(target=speechbrain.integrations.k2_fsa)

while the running server loaded the very same model without complaint. A build
that bakes models into an image runs the downloader, so the copy that was wrong
was the copy the deployment depended on.

The project already has tests asserting that the voice-state formula and the
silence threshold each exist in exactly one place. The most fragile code in the
codebase — monkeypatches against a PRIVATE SpeechBrain API, one of which exists
only because transformers removed AutoModelWithLMHead in v5 — was the one thing
copied three ways with no such guard. tests/test_contract.py now covers it.

Everything here is idempotent and safe to call repeatedly.
"""


def apply_patches(log=None) -> None:
    """
    Prepares the interpreter for SpeechBrain.

    Call this BEFORE importing anything from speechbrain. Several of the
    patches take effect during SpeechBrain's own import chain, so applying them
    afterwards is too late to help.

    log: optional callable for progress lines. Defaults to silence, because the
         server applies these on every model load and the downloader wants the
         narration.
    """
    def _say(msg):
        if log is not None:
            log(msg)

    # ── torchaudio ───────────────────────────────────────────────────────────
    # SpeechBrain calls torchaudio.list_audio_backends() during its import
    # chain, and recent torchaudio removed both that and get_audio_backend.
    # torchaudio.load is replaced with a soundfile implementation because the
    # dispatcher it would otherwise use is gone in the pinned version.
    import torchaudio
    import soundfile as sf
    import numpy as np
    import torch

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
        _say("Patched torchaudio.list_audio_backends")

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
        _say("Patched torchaudio.get_audio_backend")

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
        _say("Patched torchaudio.load with soundfile backend")

    # ── transformers ─────────────────────────────────────────────────────────
    # AutoModelWithLMHead was removed in transformers v5; SpeechBrain's
    # interface module still refers to it.
    import transformers
    if not hasattr(transformers, "AutoModelWithLMHead"):
        transformers.AutoModelWithLMHead = getattr(
            transformers, "AutoModelForCausalLM", transformers.AutoModel
        )
        _say("Patched transformers.AutoModelWithLMHead")

    # ── huggingface_hub ──────────────────────────────────────────────────────
    # use_auth_token was renamed to token.
    import huggingface_hub
    if not getattr(huggingface_hub, "_patched_by_ser_engine", False):
        _original = huggingface_hub.hf_hub_download

        def _patched(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return _original(*args, **kwargs)

        huggingface_hub.hf_hub_download        = _patched
        huggingface_hub._patched_by_ser_engine = True
        _say("Patched huggingface_hub.hf_hub_download")

    # ── SpeechBrain LazyModule ───────────────────────────────────────────────
    # THE ONE THE DOWNLOADER WAS MISSING.
    #
    # SpeechBrain defers optional imports through LazyModule and decides whether
    # to resolve them by inspecting the calling frame. When the caller turns out
    # to be inspect.py itself — which happens when anything walks the module,
    # pydoc included — it must refuse rather than import, because resolving an
    # optional integration like speechbrain.integrations.k2_fsa raises and the
    # whole load collapses.
    #
    # The upstream check compares paths in a way that misses some platforms, so
    # this version also matches the basename.
    try:
        from speechbrain.utils.importutils import LazyModule
        import importlib
        import warnings
        from types import ModuleType

        def _patched_ensure_module(self, stacklevel: int) -> ModuleType:
            import sys
            import os
            import inspect

            importer_frame = None
            try:
                importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
            except AttributeError:
                warnings.warn(
                    "Failed to inspect frame to check if we should ignore "
                    "importing a module lazily."
                )

            if importer_frame is not None and (
                importer_frame.filename.endswith("/inspect.py")
                or importer_frame.filename.endswith("\\inspect.py")
                or os.path.basename(importer_frame.filename) == "inspect.py"
            ):
                raise AttributeError()

            if self.lazy_module is None:
                try:
                    if self.package is None:
                        self.lazy_module = importlib.import_module(self.target)
                    else:
                        self.lazy_module = importlib.import_module(
                            f".{self.target}", self.package
                        )
                except Exception as e:
                    raise ImportError(f"Lazy import of {repr(self)} failed") from e

            return self.lazy_module

        LazyModule.ensure_module = _patched_ensure_module
        _say("Patched SpeechBrain LazyModule.ensure_module")

    except Exception as pe:
        # Never fatal on its own: a future SpeechBrain may not have LazyModule
        # at all, and the load below will report the real problem if there is
        # one.
        print(f"[speechbrain_compat] Warning: LazyModule patch failed: {pe}")


def fetch_kwargs() -> dict:
    """
    Extra keyword arguments for foreign_class(), so every caller fetches the
    model the same way.

    Asks SpeechBrain to COPY files into savedir rather than symlink them into
    the Hugging Face cache, which is its default. The default causes two
    distinct problems here:

      - On Windows, creating a symlink needs a privilege an ordinary account
        does not have, so the download dies with
        "WinError 1314: A required privilege is not held by the client".

      - In a container image it is a correctness trap. savedir ends up holding
        five links that resolve into the HF cache, so the two directories are
        only meaningful together — copy one without the other, or write them in
        separate build layers, and SER fails to load from what looks like a
        populated directory.

    Copying costs ~360 MB of duplication and removes both. Returns an empty
    dict on a SpeechBrain without LocalStrategy, so the caller still works.
    """
    try:
        from speechbrain.utils.fetching import LocalStrategy
        return {"local_strategy": LocalStrategy.COPY}
    except Exception:
        return {}
