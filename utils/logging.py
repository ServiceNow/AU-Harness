"""
Shared logging utilities for AudioBench.

Central logging setup and record-level logging for metrics.

For central logging setup:
  Call ``utils.logging.configure(log_file_path)`` once (e.g. from ``evaluate.py``)
  BEFORE importing other project modules that acquire loggers. If you don't call
  it, we fall back to ``default.log`` in the project root.

For metrics logging:
  Use write_record_log, write_to_run_json, and append_final_score functions
  to manage record-level and final score logging for metrics.
"""
from __future__ import annotations
import logging
import json
import re
from pathlib import Path
from itertools import zip_longest
from typing import Optional

# Default log file name for central logging
_DEFAULT_NAME = "default.log"

# Flag so we don't double-configure
_configured = False


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


def configure(log_file: Optional[str] = None):
    """Configure root logger. If *log_file* is None, use *_DEFAULT_NAME*."""
    global _configured
    if _configured:
        return
    if log_file:
        path = Path(log_file)
    else:
        path = Path(__file__).parent.parent / _DEFAULT_NAME
    _install_handlers(path)
    _configured = True


def write_record_log(self, refs, cands, scores, dataset_name, model_name, explanations=None):
    """
    Write record-level logs to a file specific to the dataset, metric, and model.
    
    Args:
        self: The metric object instance with a 'name' attribute
        refs: List of reference texts
        cands: List of candidate texts
        scores: List of scores
        dataset_name: Name of the dataset
        model_name: Name of the model
        explanations: Optional list of explanations for each score
    """
    if not refs or not scores:
        return
    
    def _slug(s):
        return re.sub(r"[^A-Za-z0-9_]+", "_", s)
    
    log_dir = Path("run_logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{_slug(dataset_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
    
    # Use provided explanations or an empty list
    if explanations is None:
        explanations = [""] * len(scores)
    
    with open(log_path, "w", encoding="utf-8") as f:
        for ref, cand, sc, expl in zip_longest(refs, cands, scores, explanations, fillvalue=None):
            entry = {"reference": ref, "candidate": cand}
            if sc is not None:
                entry["score"] = sc
            if expl:
                entry["explanation"] = expl
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    # Write to shared run.log
    write_to_run_json(self, refs, cands, scores, dataset_name, model_name, explanations)
    
    return log_path


# Flag to track if run.log has been reset for this session
_run_log_reset = False

def write_to_run_json(self, refs, cands, scores, dataset_name, model_name, explanations=None):
    """
    Write each sample's prediction to a shared run.log file that resets with every run.
    The file is truncated on the first call to this function in each program execution.
    
    Args:
        self: The metric object instance with a 'name' attribute
        refs: List of reference texts
        cands: List of candidate texts
        scores: List of scores
        dataset_name: Name of the dataset
        model_name: Name of the model
        explanations: Optional list of explanations for each score
    """
    global _run_log_reset
    run_path = Path("run_logs") / "run.log"
    run_path.parent.mkdir(exist_ok=True)
    
    # Use provided explanations or an empty list
    if explanations is None:
        explanations = [""] * len(scores)
    
    # Determine file mode: 'w' to reset file on first call, 'a' to append on subsequent calls
    file_mode = "w" if not _run_log_reset else "a"
    _run_log_reset = True
    
    # Open run.log in appropriate mode
    with open(run_path, file_mode, encoding="utf-8") as f:
        # Add entries for this metric/dataset/model
        for ref, cand, sc, expl in zip_longest(refs, cands, scores, explanations, fillvalue=None):
            entry = {
                "dataset": dataset_name,
                "metric": self.name,
                "model": model_name,
                "reference": ref,
                "candidate": cand,
            }
            if sc is not None:
                entry["score"] = sc
            if expl:
                entry["explanation"] = expl
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_final_score(self, overall, dataset_name, model_name):
    """
    Append the final aggregated score to the metric's log file.
    
    Args:
        self: The metric object instance with a 'name' attribute
        overall: Dict containing overall metrics and scores
        dataset_name: Name of the dataset
        model_name: Name of the model
    
    Returns:
        Path to the log file where the final score was appended
    """
    def _slug(s):
        return re.sub(r"[^A-Za-z0-9_]+", "_", s)
    
    log_dir = Path("run_logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{_slug(dataset_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
    
    # Append the final score to the log file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"final_score": overall}, ensure_ascii=False) + "\n")
        
    return log_path
