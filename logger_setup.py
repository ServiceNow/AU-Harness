"""Central logging setup for AudioBench.

Call ``logger_setup.configure(log_file_path)`` once (e.g. from ``evaluate.py``)
BEFORE importing other project modules that acquire loggers.  If you don’t call
it, we fall back to ``default.log`` in the project root.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

_DEFAULT_NAME = "default.log"


def _install_handlers(log_path: Path):
    """(Re)configure the root logger to write to *log_path* and stderr."""
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root_logger.addHandler(sh)


# Flag so we don’t double-configure
_configured = False


def configure(log_file: Optional[str] = None):
    """Configure root logger. If *log_file* is None, use *_DEFAULT_NAME*."""
    global _configured
    if _configured:
        return
    if log_file:
        path = Path(log_file)
    else:
        path = Path(__file__).with_name(_DEFAULT_NAME)
    _install_handlers(path)
    _configured = True
