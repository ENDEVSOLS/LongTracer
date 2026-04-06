"""
SLM Verifier — Lightweight generative model fallback for uncertain claims.

Called ONLY when the NLI cross-encoder cannot confidently decide (ambiguous
score zone or claims containing numbers/dates). Uses a quantized GGUF model
for fast CPU inference.

Requirements:
    pip install llama-cpp-python huggingface-hub
    OR: pip install longtracer[slm]
"""

import os
import re
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger("longtracer")

# Default model — Qwen2.5-1.5B-Instruct (Q4_K_M, ~1GB)
_DEFAULT_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
_DEFAULT_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Prompt template — minimal tokens for speed
_VERIFY_PROMPT = """Source: "{source}"
Claim: "{claim}"

Is the claim SUPPORTED or CONTRADICTED by the source? Answer with one word only."""


def _check_llama_cpp():
    """Check if llama-cpp-python is installed."""
    try:
        from llama_cpp import Llama  # noqa: F401
        return True
    except ImportError:
        return False


def _download_model(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> str:
    """Download GGUF model from HuggingFace Hub. Returns local file path."""
    from huggingface_hub import hf_hub_download

    cache = cache_dir or os.path.join(os.path.expanduser("~"), ".longtracer", "models")
    os.makedirs(cache, exist_ok=True)

    logger.info(f"SLM: Downloading {repo_id}/{filename} (~1GB, first time only)...")
    print(f"  ⏳ Downloading SLM model ({filename})... this only happens once.")

    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache,
        local_dir=cache,
    )

    logger.info(f"SLM: Model cached at {path}")
    print(f"     ✓ SLM model ready: {os.path.basename(path)}")
    return path


class SLMVerifier:
    """
    Lightweight generative model for resolving uncertain NLI claims.

    Lazy-loads on first call. Uses quantized GGUF for fast CPU inference.
    Designed to be called only for the ~5-10% of claims that fall in
    the ambiguous zone of the NLI cross-encoder.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_id: str = _DEFAULT_REPO,
        filename: str = _DEFAULT_FILE,
        n_ctx: int = 512,
        n_threads: Optional[int] = None,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.repo_id = repo_id
        self.filename = filename
        self.n_ctx = n_ctx
        self.n_threads = n_threads or min(os.cpu_count() or 4, 8)
        self.verbose = verbose
        self._llm = None  # Lazy loaded
        self._load_time_ms: float = 0.0

        self.stats = {
            "calls": 0,
            "total_ms": 0.0,
            "supported": 0,
            "contradicted": 0,
            "errors": 0,
        }

    def _ensure_loaded(self):
        """Lazy-load the GGUF model on first call."""
        if self._llm is not None:
            return

        from llama_cpp import Llama

        # Resolve model path
        if self.model_path and os.path.isfile(self.model_path):
            path = self.model_path
        else:
            # Check env var override
            env_path = os.environ.get("LONGTRACER_SLM_MODEL")
            if env_path and os.path.isfile(env_path):
                path = env_path
            else:
                path = _download_model(self.repo_id, self.filename)

        start = time.time()
        if self.verbose:
            print(f"  ⏳ Loading SLM ({os.path.basename(path)})...")

        self._llm = Llama(
            model_path=path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=0,  # CPU only
            verbose=False,
        )

        self._load_time_ms = (time.time() - start) * 1000
        if self.verbose:
            print(f"     ✓ SLM loaded in {self._load_time_ms:.0f}ms")
        logger.info(f"SLM: Model loaded in {self._load_time_ms:.0f}ms")

    def verify(self, claim: str, source: str) -> Dict:
        """
        Verify a single claim against a source using the SLM.

        Returns:
            {
                "supported": bool,
                "raw_output": str,
                "latency_ms": float,
            }
        """
        self._ensure_loaded()

        prompt = _VERIFY_PROMPT.format(
            source=source[:500],  # Truncate long sources
            claim=claim[:300],
        )

        start = time.time()
        try:
            output = self._llm(
                prompt,
                max_tokens=10,
                temperature=0.0,
                top_p=1.0,
                stop=["\n", ".", ","],
                echo=False,
            )
            raw = output["choices"][0]["text"].strip().lower()
            latency_ms = (time.time() - start) * 1000

            # Parse verdict
            supported = "support" in raw
            contradicted = "contradict" in raw

            if contradicted:
                is_supported = False
            elif supported:
                is_supported = True
            else:
                # Ambiguous output — default to not supported (safer)
                is_supported = False
                self.stats["errors"] += 1

            self.stats["calls"] += 1
            self.stats["total_ms"] += latency_ms
            if is_supported:
                self.stats["supported"] += 1
            else:
                self.stats["contradicted"] += 1

            return {
                "supported": is_supported,
                "raw_output": raw,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            self.stats["calls"] += 1
            self.stats["errors"] += 1
            self.stats["total_ms"] += latency_ms
            logger.warning(f"SLM verify failed: {e}")
            return {
                "supported": False,
                "raw_output": f"ERROR: {e}",
                "latency_ms": latency_ms,
            }

    def get_stats(self) -> Dict:
        """Return SLM usage statistics."""
        avg_ms = (self.stats["total_ms"] / self.stats["calls"]
                  if self.stats["calls"] > 0 else 0)
        return {
            **self.stats,
            "avg_ms": round(avg_ms, 1),
            "load_time_ms": round(self._load_time_ms, 1),
        }


def is_slm_available() -> bool:
    """Check if SLM fallback is available (llama-cpp-python installed)."""
    return _check_llama_cpp()
