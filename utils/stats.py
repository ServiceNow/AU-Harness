import json
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd

from models.model_response import ErrorTracker, ModelResponse

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Stats(ABC):
    """Abstract class for tracking statistics across entire CLAE run."""

    _instance = None

    def __new__(cls):
        """Initiates the singleton when PerfStats() is called."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initializes singleton of performance object."""
        self.stats = defaultdict(list)

    def add(self, runspec_name, model, metric, value):
        """Adds record to performance object."""
        self.stats[runspec_name].append({model: {metric: value}})

    @abstractmethod
    def add_all_stats(self, runspec_name: str, model_name: str, model_response: ModelResponse):
        """Add all measurements to the stats list for one specific llm output."""
        pass

    def clear(self, runspec_name):
        """Clears stateful performance object."""
        self.stats.pop(runspec_name, {})

    def rearranged_data(self, runspec_name):
        """Rearranges logs of performance data by model name and statistic and produces property."""
        data = self.stats[runspec_name]
        rearranged_data = defaultdict(lambda: defaultdict(list))
        for item in data:
            for model, metrics in item.items():
                for metric, value in metrics.items():
                    rearranged_data[model][metric].append(value)
        return rearranged_data

    def compute_statistics(self, data):
        """Aggregates float or bool columns with appropriate statistics."""
        statistics = defaultdict(dict)
        for model, metrics in data.items():
            statistics[model] = {}
            for metric, values in metrics.items():
                if metric == "response_code":
                    statistics[model][metric] = {
                        str(response_code): values.count(response_code) for response_code in set(values)
                    }
                    continue
                if isinstance(values, list):
                    if all(isinstance(item, list) for item in values):
                        logger.info("Selecting last value in list for computing perf stats")
                        values = values[len(values) - 1]
                    elif all(isinstance(item, (dict, str)) for item in values):
                        continue
                try:
                    np_values = np.array(values)
                except Exception as e:
                    logger.error(
                        f"Error converting values to numpy array: {values} for metric {metric} for model {model}. Exception: {e}"
                    )
                    continue
                np_values = np_values[np_values != np.array(None)]
                if len(np_values) == 0:
                    statistics[model][metric] = {"average": None}
                elif isinstance(values[0], bool):
                    statistics[model][metric] = {
                        "average": float(np.mean(np_values)),
                        "count": float(np.sum(np_values)),
                    }
                elif isinstance(values[0], float):
                    statistics[model][metric] = {
                        "min": float(np.nanmin(np_values)),
                        "max": float(np.nanmax(np_values)),
                        "average": float(np.nanmean(np_values)),
                        "quantiles": {
                            "25%": float(np.nanpercentile(np_values, 25)),
                            "50%": float(np.nanpercentile(np_values, 50)),
                            "75%": float(np.nanpercentile(np_values, 75)),
                            "95%": float(np.nanpercentile(np_values, 95)),
                            "99%": float(np.nanpercentile(np_values, 99)),
                        },
                    }
                elif isinstance(values[0], ErrorTracker):
                    statistics[model][metric] = {
                        "rate_limit (429)": values[0].rate_limit,
                        "connection_error (599)": values[0].connection_error,
                        "api_error (401)": values[0].api_error,
                        "request_timeout (408)": values[0].request_timeout,
                        "internal_server (500)": values[0].internal_server,
                        "other": values[0].other,
                    }
                else:
                    logger.debug(f"Unable to aggregate row {values[0]}")
        return statistics

    def statistics(self, runspec_name):
        """Get the statistics of the performance metrics. Produces property.

        Returns:
        -------
            dict: The statistics of the performance metrics
        """
        if runspec_name not in self.stats:
            return {}
        return self.compute_statistics(self.rearranged_data(runspec_name))

    def record_level_data(self, runspec_name):
        """Produces property for organized record level performance data."""
        return self._generate_record_level_view(self.rearranged_data(runspec_name))

    def _generate_record_level_view(self, data) -> dict[str, pd.DataFrame]:
        return {model_name: pd.DataFrame(metrics) for model_name, metrics in data.items()}

    def generate_summary(self, filename, runspec_name):
        """Generates a summary of all performance stats, as well as a record level file of performance stats.

        Args:
            filename: The filename to write performance stats to.
            runspec_name: The name of the runspec to generate the summary for.

        Returns: statistics

        """
        record_level_data = self.record_level_data(runspec_name)
        for model_name, dataframe in record_level_data.items():
            dataframe.to_csv(f"{filename}_{model_name}_record_level.csv", index=False)
        if self.statistics(runspec_name):
            with open(f"{filename}.json", "w") as f:
                json.dump(self.statistics(runspec_name), f, indent=4)
