"""Central logging setup for AudioBench.

Import this module once (e.g. at the very top of `evaluate.py`) before importing
other project modules. It configures a single root logger that overwrites
`audiobench.log` on every run and ensures no duplicate handlers are created.
"""
import logging
from pathlib import Path

_LOG_FILE = Path(__file__).with_name("audiobench.log")

# Clear existing handlers from the root logger
root_logger = logging.getLogger()
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)

root_logger.setLevel(logging.INFO)
fh = logging.FileHandler(_LOG_FILE, mode="w")  # overwrite each run
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
root_logger.addHandler(fh)

# Optional: also log to stderr at INFO level
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)
root_logger.addHandler(sh)
