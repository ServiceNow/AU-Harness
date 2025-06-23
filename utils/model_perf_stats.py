from models.model_response import ModelResponse
from utils.stats import Stats


def _iter_prompt_texts(prompt: list[dict]):
    for message in prompt:
        match message:
            case {"content": str() as text}:
                yield text
            case {"content": list() as content}:
                for block in content:
                    match block:
                        case {"text": str() as text}:
                            yield text


class PerfStats(Stats):
    """Singleton instance to track performance statistics across entire CLAE run."""

    def add_all_stats(self, runspec_name: str, model_name: str, model_response: ModelResponse, is_fix_it: bool = False):
        """Add all measurements to the stats list for one specific llm output."""
        llm_response = model_response.llm_response if 200 <= model_response.response_code < 300 else ""
        self.add(
            runspec_name,
            model_name,
            "input",
            str(model_response.input_prompt)
            if isinstance(model_response.input_prompt, list)
            else model_response.input_prompt,
        )
        self.add(runspec_name, model_name, "model_parameters", model_response.model_parameters)
        self.add(runspec_name, model_name, "llm_response", model_response.llm_response)
        self.add(runspec_name, model_name, "raw_response", model_response.raw_response)
        self.add(runspec_name, model_name, "is_empty", llm_response.strip() == "")
        self.add(runspec_name, model_name, "reasoning_response", model_response.reasoning_response)
        self.add(
            runspec_name, model_name, "reasoning_possible_to_extract", model_response.reasoning_possible_to_extract
        )
        self.add(runspec_name, model_name, "reasoning_correct_format", model_response.reasoning_correct_format)
        self.add(runspec_name, model_name, "stop_reason", model_response.stop_reason)
        self.add(runspec_name, model_name, "retrieved_fix_it_cache", is_fix_it)
        self.add(
            runspec_name,
            model_name,
            "relative_output_tokens",
            float(model_response.performance.relative_output_tokens)
            if model_response.performance
            else len(llm_response.split()),
        )
        if model_response.performance:
            input_tokens = model_response.performance.prompt_tokens
        elif isinstance(model_response.input_prompt, list):
            input_tokens = sum(len(text.split()) for text in _iter_prompt_texts(model_response.input_prompt))
        else:
            input_tokens = len(model_response.input_prompt.split())
        self.add(
            runspec_name,
            model_name,
            "input_tokens",
            float(input_tokens),
        )
        self.add(
            runspec_name,
            model_name,
            "output_tokens",
            float(model_response.performance.response_tokens)
            if model_response.performance
            else len(llm_response.split()),
        )
        self.add(
            runspec_name,
            model_name,
            "reasoning_tokens",
            model_response.performance.reasoning_tokens if model_response.performance else None,
        )
        self.add(runspec_name, model_name, "success", model_response.response_code == 200)
        self.add(
            runspec_name,
            model_name,
            "errors",
            model_response.error_tracker if model_response.error_tracker else model_response.response_code == 429,
        )
        self.add(
            runspec_name,
            model_name,
            "time_per_token",
            float(model_response.performance.time_per_token)
            if model_response.performance and model_response.performance.time_per_token
            else -1,
        )
        self.add(runspec_name, model_name, "response_code", model_response.response_code)

    def get_num_tokens(self, step, runspec_name) -> dict[str, float]:
        """Hack just to return the num_tokens statistic in performance stats until we have CLAE UI."""
        if step == 0:
            for model_name, model_stats in self.statistics(runspec_name).items():
                if (
                    model_stats
                    and "output_tokens" in model_stats
                    and "relative_output_tokens" in model_stats
                    and "reasoning_tokens" in model_stats
                ):
                    return {
                        f"{model_name}_average_output_tokens": model_stats["output_tokens"]["average"],
                        f"{model_name}_average_relative_output_tokens": model_stats["relative_output_tokens"][
                            "average"
                        ],
                        f"{model_name}_average_reasoning_tokens": model_stats["reasoning_tokens"]["average"],
                    }
        else:
            _step = str(step)
            for model_name, model_stats in self.statistics(runspec_name).items():
                if (
                    _step in model_name
                    and model_stats
                    and "output_tokens" in model_stats
                    and "relative_output_tokens" in model_stats
                    and "reasoning_tokens" in model_stats
                ):
                    return {
                        f"{model_name}_average_output_tokens": model_stats["output_tokens"]["average"],
                        f"{model_name}_average_relative_output_tokens": model_stats["relative_output_tokens"][
                            "average"
                        ],
                        f"{model_name}_average_reasoning_tokens": model_stats["reasoning_tokens"]["average"],
                    }
        return {}
