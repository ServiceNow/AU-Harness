class _SimpleMeta:  # stand-in for MetricMetadata
    def __init__(self, name, display_name=None, description=""):
        self.name = name
        self.display_name = display_name or name
        self.description = description

MetricMetadata = _SimpleMeta
class ReportingMetrics(dict):
    """Placeholder so downstream code still works."""


class AudiobenchPostprocessor():
    """Postprocessor class to calculate the model scores for the model predictions."""

    def process(self, results: list[dict]) -> dict[str, float | list]:
        """Process the results to calculate.

        The judge_response can be either 'incorrect' or 'correct'. Sum up the correct
        responses and calculate the correctness rate. Also, calculate the correctness rate
        by pattern.

        Args:
        ----
            results: results of the model predictions

        Returns:
        -------
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

    def get_reporting_summary_score(self, overall_score):
        """Get the overall score to show in the result dashboard."""
        return ReportingMetrics(overall_score=overall_score["avg_rating"] / 100)

    def metadata_info(self):
        """Creates a mapping from the metric name to the metric metadata object."""
        return {
            "avg_rating": MetricMetadata(
                name="Average Rating",
                display_name="Average Rating",
                description="Average judge score rating across all predictions",
            ),
            "success_rate": MetricMetadata(
                name="Success Rate",
                display_name="Success Rate",
                description="Rate of successful extractions from the judge predictions",
            ),
        }
