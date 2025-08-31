import re
from typing import List, Tuple, Dict, Optional, Union

from metrics.metrics import Metrics
from models.model_response import ModelResponse
from utils.custom_logging import write_record_log, append_final_score

#### Constants ####
PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}


class BFCLMatchScore(Metrics):
    """BFCL function calling match score metric.

    Supports parallel function calls (order-insensitive). Rejects unexpected
    parameters and checks required arguments.
    """

    def __init__(self):
        super().__init__()
        self.name = "bfcl_match_score"

    def __call__(
            self,
            candidates: List[dict],
            references: List[Tuple[List[str], List[Dict[str, Optional[Union[str, int]]]]]],
            *,
            instructions: Optional[List[str]] = None,
            task_name: Optional[str] = None,
            model_name: Optional[str] = None,
            model_responses: Optional[List[ModelResponse]] = None,
    ) -> dict[str, dict[str, float] | float]:
        # Compute record-level scores for strict outputs (binary: all instructions followed or not)
        record_scores = self.compute_record_level_scores(candidates, references)
        # Average final score over all components
        results = {"final": sum(record_scores) / len(candidates) if candidates else 0.0}

        # Write detailed record-level logs (if task_name and model_name provided)
        if task_name and model_name:            
            # Very simple approach: just stringify everything
            serializable_candidates = [str(candidate) for candidate in candidates]
            serializable_refs = [str(ref[0]) for ref in references]
            write_record_log(
                self,
                refs=serializable_refs,
                cands=serializable_candidates,
                scores=record_scores,
                task_name=task_name,
                model_name=model_name,
                explanations=None,
                instructions=instructions,
                model_responses=model_responses,
            )

            append_final_score(self, results, task_name, model_name, model_responses)

        return results

    # ----------------- Helpers -----------------

    def _standardize_value(self, val):
        if isinstance(val, str):
            regex_string = r"[ \,\.\/\-\_\*\^]"
            return re.sub(regex_string, "", val).lower().replace("'", '"')
        elif callable(val):
            return repr(val)
        elif isinstance(val, list):
            return [self._standardize_value(v) for v in val]
        elif isinstance(val, dict):
            return {k: self._standardize_value(v) for k, v in val.items()}
        return val

    def _compare_dicts(self, tool_dict: dict, ref_dict: dict):
        for k in tool_dict:
            if k not in ref_dict:
                return False, f"Unexpected dict key: {k}"
        for k, allowed in ref_dict.items():
            if k not in tool_dict and allowed not in ("", None):
                return False, f"Missing dict key: {k}"
        for k, v in tool_dict.items():
            tv = self._standardize_value(v)
            rv = self._standardize_value(ref_dict[k])
            if isinstance(rv, list):
                rv_std = [self._standardize_value(x) for x in rv]
                if tv not in rv_std:
                    return False, f"Dict value mismatch at '{k}': {tv} ∉ {rv_std}"
            else:
                if tv != rv:
                    return False, f"Dict value mismatch at '{k}': {tv} != {rv}"
        return True, ""

    def _compare_tool_call(self, tool_call, ref_call, tool_required_params):
        """Compare one tool call against its reference."""
        if not isinstance(tool_call, dict) or not isinstance(ref_call, dict):
            return False, ["Tool/Reference call is not a dict."]

        tool_name = list(tool_call.keys())[0]
        ref_tool_name = list(ref_call.keys())[0]

        if re.sub(r"\.", "_", tool_name) != re.sub(r"\.", "_", ref_tool_name):
            return False, [f"Function name mismatch: {tool_name} vs {ref_tool_name}"]

        tool_params = tool_call[tool_name]
        ref_params = ref_call[ref_tool_name]

        # --- Reject unexpected top-level parameters ---
        for k in tool_params:
            if k not in ref_params:
                return False, [f"Unexpected parameter: {k}"]

        required_params = tool_required_params.get(tool_name) or tool_required_params.get(
            ref_tool_name
        )
        if required_params is None:
            return False, [f"Missing required-params metadata for tool '{tool_name}'"]

        all_match = True
        errors = []

        for param, param_type in required_params:
            python_type = PYTHON_TYPE_MAPPING.get(param_type, str)

            if param not in tool_params or param not in ref_params:
                errors.append(f"Missing required parameter '{param}'.")
                all_match = False
                break

            tool_value = self._standardize_value(tool_params[param])
            ref_value = self._standardize_value(ref_params[param])

            # Empty/None means no constraint
            if ref_value in ("", None):
                continue

            # Normalize numeric float/int
            if python_type == float and isinstance(tool_value, int):
                tool_value = float(tool_value)

            # ---- dict parameter handling ----
            if python_type == dict:
                if isinstance(ref_value, dict):
                    ok, msg = self._compare_dicts(tool_value, ref_value)
                    if not ok:
                        errors.append(msg)
                        all_match = False
                elif isinstance(ref_value, list):
                    # list of dict-templates: succeed if any template matches
                    dict_templates = [x for x in ref_value if isinstance(x, dict)]
                    if len(dict_templates) == len(ref_value) and dict_templates:
                        matched_any = False
                        last_msg = ""
                        for tmpl in dict_templates:
                            ok, msg = self._compare_dicts(tool_value, tmpl)
                            if ok:
                                matched_any = True
                                break
                            else:
                                last_msg = msg
                        if not matched_any:
                            errors.append(
                                f"Dict value for '{param}' did not match any allowed templates. Last error: {last_msg}"
                            )
                            all_match = False
                    else:
                        # Fallback: treat as allowed list of whole dicts (exact match)
                        if tool_value not in ref_value:
                            errors.append(
                                f"Value for '{param}' not in allowed list: {tool_value} ∉ {ref_value}"
                            )
                            all_match = False
                else:
                    errors.append(
                        f"Type mismatch for '{param}': expected dict semantics, got {type(ref_value).__name__} in reference"
                    )
                    all_match = False

            elif python_type == list:
                if isinstance(ref_value, list):
                    # Case: list of possible lists (list-of-lists)
                    if all(isinstance(x, list) for x in ref_value):
                        matched_any = False
                        last_msg = ""
                        for candidate_list in ref_value:
                            if len(tool_value) != len(candidate_list):
                                continue
                            elementwise_ok = True
                            for tv, rv in zip(tool_value, candidate_list):
                                if isinstance(rv, dict) and isinstance(tv, dict):
                                    ok, msg = self._compare_dicts(tv, rv)
                                    if not ok:
                                        elementwise_ok = False
                                        last_msg = msg
                                        break
                                else:
                                    if tv != rv:
                                        elementwise_ok = False
                                        last_msg = f"List element mismatch: {tv} != {rv}"
                                        break
                            if elementwise_ok:
                                matched_any = True
                                break
                        if not matched_any:
                            errors.append(
                                f"List value for '{param}' did not match any allowed options. Last error: {last_msg}"
                            )
                            all_match = False
                    else:
                        # Simple allowed-values check
                        if tool_value not in ref_value:
                            errors.append(
                                f"Value for '{param}' not in allowed list: {tool_value} ∉ {ref_value}"
                            )
                            all_match = False
                else:
                    if tool_value != ref_value:
                        errors.append(
                            f"Mismatch for '{param}': {tool_value} != {ref_value}"
                        )
                        all_match = False

        return all_match, errors

    # ----------------- Core compute -----------------

    def _compute_outputs(
        self,
        candidates: List[dict],
        references: List[
            Tuple[List[Dict[str, dict]], Dict[str, List[Tuple[str, str]]]]
        ],
    ) -> List[dict]:

        outputs = []

        for i, candidate in enumerate(candidates):
            tool_response = candidate.get("tool_response")
            reference_tool_response, tool_required_params = references[i]

            if tool_response is None:
                outputs.append(
                    {
                        "valid": False,
                        "results": [False],
                        "errors": ["tool_response is None."],
                    }
                )
                continue

            if len(tool_response) != len(reference_tool_response):
                outputs.append(
                    {
                        "valid": False,
                        "results": [False],
                        "errors": [
                            f"Wrong number of tool calls: got {len(tool_response)}, expected {len(reference_tool_response)}."
                        ],
                        "tool_response": str(tool_response),
                        "reference_tool_response": str(reference_tool_response),
                        "tool_required_params": str(tool_required_params)
                    }
                )
                continue

            if len(tool_response) == 0 and len(reference_tool_response) == 0:
                outputs.append({"valid": True, "results": [True], "errors": []})
                continue

            unmatched_refs = reference_tool_response[:]
            try:
                unmatched_cands = tool_response[:]
                results, errors = [], []

                for ref_call in unmatched_refs:
                    matched = False
                    for j, cand_call in enumerate(unmatched_cands):
                        ok, err = self._compare_tool_call(
                            cand_call, ref_call, tool_required_params
                        )
                        if ok:
                            results.append(True)
                            unmatched_cands.pop(j)
                            matched = True
                            break
                    if not matched:
                        results.append(False)
                        errors.append(f"Could not find match for reference call: {ref_call}")
            except Exception as e:
                outputs.append({"valid": False, "results": [False], "errors": [e]})

            outputs.append(
                {"valid": all(results), "results": results, "errors": errors, "tool_response": str(tool_response),
                 "reference_tool_response": str(reference_tool_response),
                 "tool_required_params": str(tool_required_params)}
            )

        return outputs

    def compute_record_level_scores(
            self,
            candidates: List[str],
            references: List[Tuple[List[str], List[Dict[str, Optional[Union[str, int]]]]]],
    ) -> List[float]:
        outputs = self._compute_outputs(candidates, references)
        return [float(out["valid"]) for out in outputs]
