"""Qwen3.5 Supervisor process manager adapted from SkillFlow.

Upstream reference:
``SkillFlow/training/sglang_manager.py::SGLangSupervisorManager``.

Only the runtime boundary is kept here.  Importing this module does not import
SGLang, allocate a model, or touch a GPU; the heavy dependency is loaded in the
spawned child only when :meth:`SGLangSupervisorManager.start` is called.
Training and LoRA synchronization intentionally remain outside this module.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


def _configure_worker_environment(
    *,
    cuda_version: Optional[str],
    executable: Optional[str] = None,
    cuda_root: Path = Path("/usr/local"),
) -> None:
    """Apply SkillFlow's SGLang launcher environment inside the child.

    This is the same environment boundary as
    ``SkillFlow/scripts/launch_sglang.py::build_environment``: make the
    selected Python environment's tools (notably ``ninja``) discoverable and
    select the CUDA toolkit matching ``torch.version.cuda`` when available.
    """

    executable_directory = str(Path(executable or sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join(
        [
            executable_directory,
            *[entry for entry in path_entries if entry != executable_directory],
        ]
    )
    os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "0")
    if "CUDA_HOME" not in os.environ and cuda_version is not None:
        candidate = cuda_root / f"cuda-{cuda_version}"
        if (candidate / "bin" / "nvcc").is_file():
            os.environ["CUDA_HOME"] = str(candidate)


def _sglang_worker_entry(server_args: Dict[str, Any]) -> None:
    """Launch SGLang in a spawned child, matching SkillFlow's process boundary."""

    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("SGLANG_DISABLE_CUDNN_CHECK", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    _configure_worker_environment(cuda_version=torch.version.cuda)

    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs

    launch_server(ServerArgs(**server_args))


class SGLangSupervisorManager:
    """Own the dedicated Qwen3.5-9B SGLang rollout process.

    The constructor and SGLang arguments deliberately follow SkillFlow.  The
    project-specific changes are configuration defaults, a side-effect-free
    import path, and a bounded HTTP readiness probe.
    """

    def __init__(
        self,
        *,
        model_path: str = "Qwen/Qwen3.5-9B",
        tokenizer_path: Optional[str] = None,
        port: int = 8015,
        api_key: str = "EMPTY",
        gpu_id: int = 4,
        max_lora_rank: int = 64,
        lora_target_modules: Optional[list[str]] = None,
        max_loras_per_batch: int = 1,
        max_loaded_loras: int = 2,
        mem_fraction_static: float = 0.82,
        context_length: int = 32768,
        served_model_name: str = "supervisor_theta",
        reasoning_parser: str = "qwen3",
        tool_call_parser: str = "qwen3_coder",
        host: str = "127.0.0.1",
        schedule_policy: str = "lpm",
        sampling_backend: str = "pytorch",
        enable_multimodal: bool = True,
        ready_timeout_seconds: float = 300.0,
    ) -> None:
        if not model_path.strip() or not served_model_name.strip() or not host.strip():
            raise ValueError("model_path and served_model_name must be non-empty")
        if tokenizer_path is not None and not tokenizer_path.strip():
            raise ValueError("tokenizer_path must be non-empty when provided")
        if port <= 0 or gpu_id < 0 or max_lora_rank <= 0:
            raise ValueError("port, gpu_id, and max_lora_rank are invalid")
        if not 0.0 < mem_fraction_static < 1.0:
            raise ValueError("mem_fraction_static must be between zero and one")
        if context_length <= 0 or ready_timeout_seconds <= 0:
            raise ValueError("context and readiness limits must be positive")
        if not sampling_backend.strip():
            raise ValueError("sampling_backend must be non-empty")

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.port = int(port)
        self.api_key = api_key
        self.gpu_id = int(gpu_id)
        self.max_lora_rank = int(max_lora_rank)
        self.lora_target_modules = lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
        self.max_loras_per_batch = int(max_loras_per_batch)
        self.max_loaded_loras = int(max_loaded_loras)
        self.mem_fraction_static = float(mem_fraction_static)
        self.context_length = int(context_length)
        self.served_model_name = served_model_name
        self.reasoning_parser = reasoning_parser
        self.tool_call_parser = tool_call_parser
        self.host = host.strip()
        self.schedule_policy = schedule_policy
        self.sampling_backend = sampling_backend.strip()
        self.enable_multimodal = bool(enable_multimodal)
        self.ready_timeout_seconds = float(ready_timeout_seconds)

        self._context = mp.get_context("spawn")
        self._process: Optional[mp.Process] = None
        self._ready_marker: Optional[Path] = None

    @property
    def api_base(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def server_args(self) -> Dict[str, Any]:
        """Return SkillFlow-compatible ``ServerArgs`` without starting SGLang."""

        arguments = {
            "model_path": self.model_path,
            "host": self.host,
            "port": self.port,
            "api_key": self.api_key,
            "reasoning_parser": self.reasoning_parser,
            "tool_call_parser": self.tool_call_parser,
            "trust_remote_code": True,
            "served_model_name": self.served_model_name,
            "context_length": self.context_length,
            "mem_fraction_static": self.mem_fraction_static,
            "enable_lora": True,
            "max_lora_rank": self.max_lora_rank,
            "lora_target_modules": list(self.lora_target_modules),
            "max_loras_per_batch": self.max_loras_per_batch,
            "max_loaded_loras": self.max_loaded_loras,
            "schedule_policy": self.schedule_policy,
            "sampling_backend": self.sampling_backend,
            "enable_multimodal": self.enable_multimodal,
        }
        if self.tokenizer_path is not None:
            arguments["tokenizer_path"] = self.tokenizer_path
        return arguments

    def start(self) -> None:
        if self.is_alive():
            raise RuntimeError("SGLang Supervisor is already running")

        previous_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        marker_fd, marker_name = tempfile.mkstemp(prefix="flowsteer_sglang_", suffix=".ready")
        os.close(marker_fd)
        self._ready_marker = Path(marker_name)
        try:
            self._process = self._context.Process(
                target=_sglang_worker_entry,
                args=(self.server_args(),),
                name=f"SGLang-supervisor-gpu{self.gpu_id}",
                daemon=False,
            )
            self._process.start()
        finally:
            if previous_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_devices

        try:
            self._wait_ready()
        except BaseException:
            self.stop()
            raise

    def stop(self, timeout_seconds: float = 30.0) -> None:
        process = self._process
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=max(timeout_seconds, 0.0))
            if process.is_alive():
                raise RuntimeError("SGLang Supervisor did not stop within the timeout")
        self._process = None
        if self._ready_marker is not None:
            self._ready_marker.unlink(missing_ok=True)
            self._ready_marker = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def pid(self) -> Optional[int]:
        return self._process.pid if self._process is not None else None

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout_seconds
        request = Request(
            self.api_base + "/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        while time.monotonic() < deadline:
            if self._process is None or not self._process.is_alive():
                exit_code = self._process.exitcode if self._process is not None else None
                raise RuntimeError(f"SGLang Supervisor exited before readiness (exit={exit_code})")
            try:
                with urlopen(request, timeout=3.0) as response:
                    body = response.read().decode("utf-8", errors="replace")
                if self.served_model_name in body:
                    if self._ready_marker is not None:
                        self._ready_marker.write_text("ready\n", encoding="utf-8")
                    return
            except (OSError, TimeoutError, URLError):
                pass
            time.sleep(3.0)
        raise TimeoutError(
            f"SGLang Supervisor did not become ready within {self.ready_timeout_seconds:g}s"
        )


__all__ = ["SGLangSupervisorManager"]
