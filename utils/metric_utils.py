import logging

from utils.constants import metric_map

logger = logging.getLogger(__name__)


# Metric Loader
def _load_metric(name: str, language: str = "en", judge_settings: dict = None):

    if name not in metric_map:
        raise ValueError(f"Unknown metric: {name}. Available metrics: {list(metric_map.keys())}")

    module_name, class_name = metric_map[name]

    try:
        # Dynamically import the module and class
        module = __import__(module_name, fromlist=[class_name])
        MetricClass = getattr(module, class_name)

        # Handle metric-specific initialization parameters
        if "wer" in name.lower():
            metric = MetricClass(language=language)
        elif "judge" in name.lower():
            # Extract judge settings or use empty dict if None
            judge_settings = judge_settings or {}
            
            # Pass all judge settings as judge_properties
            metric = MetricClass(judge_properties=judge_settings)
        else:
            # Default initialization for other metrics
            metric = MetricClass()

        return metric
    except (ImportError, AttributeError) as e:
        logger.error(f"[_load_metric] Failed to load metric {name}: {e}")
        raise ValueError(f"Failed to load metric {name}: {e}")