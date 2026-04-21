"""
Config loader — reads [tool.longtracer] from pyproject.toml.

Priority chain (highest to lowest):
    Code args > Env vars > pyproject.toml > Built-in defaults

This module only handles the pyproject.toml layer. Priority merging
is done by the callers (core.py, verifier.py).
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("longtracer")

# Supported config keys and their expected types
_VALID_KEYS = {
    "project": str,
    "backend": str,
    "threshold": float,
    "verbose": bool,
    "log_level": str,
    "webhook_url": str,
    "webhook_secret": str,
    "webhook_events": list,
    "webhook_timeout": float,
}

# Module-level cache so we only read once per process
_cached_config: Dict[str, Any] | None = None


def _find_pyproject() -> Path | None:
    """Walk up from cwd to find the nearest pyproject.toml."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None


def _parse_toml(path: Path) -> Dict[str, Any]:
    """Read a TOML file and return the parsed dict."""
    try:
        # Python 3.11+ has tomllib in stdlib
        import tomllib
    except ModuleNotFoundError:
        # Python 3.10 fallback
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            logger.debug(
                "Neither tomllib (3.11+) nor tomli found. "
                "Config file support disabled. "
                "Install with: pip install tomli"
            )
            return {}

    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(*, force_reload: bool = False) -> Dict[str, Any]:
    """Load [tool.longtracer] from the nearest pyproject.toml.

    Returns:
        Dict with config values, or empty dict if no config found.
        Only keys listed in _VALID_KEYS are returned.
    """
    global _cached_config

    if _cached_config is not None and not force_reload:
        return _cached_config

    pyproject = _find_pyproject()
    if pyproject is None:
        _cached_config = {}
        return _cached_config

    try:
        data = _parse_toml(pyproject)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", pyproject, e)
        _cached_config = {}
        return _cached_config

    raw = data.get("tool", {}).get("longtracer", {})
    if not raw:
        _cached_config = {}
        return _cached_config

    # Validate and filter keys
    config: Dict[str, Any] = {}
    for key, expected_type in _VALID_KEYS.items():
        if key in raw:
            value = raw[key]
            if isinstance(value, expected_type):
                config[key] = value
            else:
                logger.warning(
                    "[tool.longtracer] %s: expected %s, got %s — ignoring",
                    key, expected_type.__name__, type(value).__name__,
                )

    # Warn about unknown keys
    unknown = set(raw.keys()) - set(_VALID_KEYS.keys())
    if unknown:
        logger.warning(
            "[tool.longtracer] unknown keys ignored: %s", ", ".join(sorted(unknown))
        )

    _cached_config = config
    logger.debug("Loaded config from %s: %s", pyproject, config)
    return _cached_config


def reset_config_cache() -> None:
    """Clear the cached config (useful for testing)."""
    global _cached_config
    _cached_config = None
