class _SimpleMeta:  # stand-in for MetricMetadata
    def __init__(self, name, display_name=None, description=""):
        self.name = name
        self.display_name = display_name or name
        self.description = description

MetricMetadata = _SimpleMeta
class ReportingMetrics(dict):
    """Placeholder so downstream code still works."""


class AudiobenchPostprocessor(Postprocessor):
    """Postprocessor class to calculate the model scores for the model predictions."""
    def extract_model_targets(self, dataset: list[dict]) -> list:
        """Return a list of `model_target` strings, preserving dataset order.

        Args:
            dataset: A list where each element is the sample dictionary produced
                     by the pre-processor (must contain the key ``"model_target"``).

        Returns:
            List of ``model_target`` values in the same order as ``dataset``.
        """
        return [record["model_target"] for record in dataset if "model_target" in record]
    def process(self, results: list[dict]) -> dict[str, float | list]:
        """Process the results to calculate.

        The judge_response can be either 'incorrect' or 'correct'. Sum up the correct
        responses and calculate the correctness rate. Also, calculate the correctness rate
        by pattern.

        Args:
            results: results of the model predictions

        Returns:
            output: dictionary containing the correctness score

        """
        overall_sum = 0
        successful_extractions = 0
        multiplier = 100 if results[0]["judge_type"] == "binary" else 20
        for result in results:
            try:
                overall_sum += int(result["eval_response"].split()[-1])
                successful_extractions += 1
            except (ValueError, IndexError):
                continue

        success_rate = successful_extractions / len(results)
        overall_results = {
            "avg_rating": ((overall_sum / len(results)) * multiplier) / 100,
            "success_rate": success_rate,
        }
        return overall_results