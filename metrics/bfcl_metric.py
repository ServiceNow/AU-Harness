import re
import json
from typing import List, Tuple, Dict, Optional, Union
from metrics.metrics import Metrics
from utils.custom_logging import write_record_log, append_final_score
from models.model_response import ModelResponse

class BFCLMatchScore(Metrics):
    def __init__(self):
        super().__init__()
        self.name = "bfcl_match_score"

    def __call__(
            self,
            candidates: List[dict],
            references: List[Tuple[List[str], List[Dict[str, Optional[Union[str, int]]]]]],
            *,
            instructions: Optional[List[str]] = None,
            dataset_name: Optional[str] = None,
            model_name: Optional[str] = None,
            model_responses: Optional[List[ModelResponse]] = None,
    ) -> dict[str, dict[str, float] | float]:
        # Compute record-level scores for strict outputs (binary: all instructions followed or not)
        record_scores = self.compute_record_level_scores(candidates, references)
        # Average final score over all components
        results = {"final": sum(record_scores) / len(candidates)}

        # Write detailed record-level logs (if dataset_name and model_name provided)
        if dataset_name and model_name:            
            # Very simple approach: just stringify everything
            serializable_candidates = [str(candidate) for candidate in candidates]
            serializable_refs = [str(ref[0]) for ref in references]
            
            write_record_log(
                self,
                refs=serializable_refs,
                cands=serializable_candidates,
                scores=record_scores,
                dataset_name=dataset_name,
                model_name=model_name,
                explanations=None,
                instructions=instructions
            )

            append_final_score(self, results, dataset_name, model_name)

        return results

    def _compute_outputs(
            self,
            candidates: List[dict],
            references: List[Tuple[List[str], List[Dict[str, Optional[Union[str, int]]]]]],
            strict: bool = True,
    ) -> List[dict]:
        outputs = []
        for i in range(len(candidates)):
            candidate = candidates[i]
            reference = references[i]

            llm_response = candidate["llm_response"]
            tool_response = candidate["tool_response"]
            reference_tool_response = reference[0]
            tool_required_params = reference[1]

            tool_match_list = []
            if llm_response.strip():
                tool_match_list.append(False)
            elif tool_response is None:
                tool_match_list.append(False)
            elif len(tool_response) != len(reference_tool_response):
                tool_match_list.append(False)
            elif len(tool_response) == 0 and len(reference_tool_response) == 0:
                tool_match_list.append(True)
            else:
                # Compare each tool call one-by-one
                for idx, (tool_call, ref_call) in enumerate(zip(tool_response, reference_tool_response)):
                    if not isinstance(tool_call, dict) or not isinstance(ref_call, dict):
                        tool_match_list.append(False)
                        continue
                    # Extract tool names (expecting only one per call)
                    tool_name = list(tool_call.keys())[0]
                    ref_tool_name = list(ref_call.keys())[0]

                    # Tool name mismatch
                    if re.sub(r"\.", "_", tool_name) != re.sub(r"\.", "_", ref_tool_name):
                        tool_match_list.append(False)
                        continue

                    tool_params = tool_call[tool_name]
                    ref_params = ref_call[ref_tool_name]
                    required_params = tool_required_params[idx]  # Parameters required for this tool call

                    # Check if all required params match
                    all_match = True
                    for param in required_params:
                        if param not in tool_params or param not in ref_params:
                            all_match = False
                            break

                        tool_value = tool_params[param]
                        ref_value = ref_params[param]

                        # Normalize reference value if it's a list with a single value
                        if isinstance(ref_value, list) and len(ref_value) == 1:
                            ref_value = ref_value[0]

                        # Final comparison
                        if tool_value != ref_value:
                            all_match = False
                            break

                    tool_match_list.append(all_match)

            outputs.append({
                "tool_responses_result": tool_match_list,
            })
        return outputs

    def compute_record_level_scores(
            self,
            candidates: List[str],
            references: List[Tuple[List[str], List[Dict[str, Optional[Union[str, int]]]]]],
    ) -> List[float]:
        outputs = self._compute_outputs(candidates, references)
        return [float(all(out["tool_responses_result"])) for out in outputs]
