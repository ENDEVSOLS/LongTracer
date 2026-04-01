"""
LongTracer Logging Configuration.

Configures the 'longtracer' logger with optional verbose console output.
"""

import logging
import sys


def configure_logging(level: str = "INFO", verbose: bool = False):
    """
    Configure the LongTracer logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        verbose: If True, adds a console handler for detailed span summaries.
    """
    logger = logging.getLogger("longtracer")
    level_num = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level_num)

    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level_num)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console.setFormatter(formatter)
    logger.addHandler(console)

    if verbose and level_num > logging.INFO:
        logger.setLevel(logging.INFO)
        console.setLevel(logging.INFO)

def log_span(span_name: str, **kwargs):
    """Log a structured span summary in verbose mode."""
    logger = logging.getLogger("longtracer")
    if logger.isEnabledFor(logging.INFO):
        tags = " ".join([f"{k}={v}" for k, v in kwargs.items()])
        logger.info(f"[longtracer] span={span_name} {tags}")

def log_trace_id(trace_id: str):
    """Log the trace ID at the end of a root run."""
    logger = logging.getLogger("longtracer")
    if logger.isEnabledFor(logging.INFO):
        logger.info(f"[longtracer] trace_id={trace_id}")
