"""Voice bench instruction following evaluation score metrics."""
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from models.model_response import ModelResponse
from metrics.metrics import Metrics
from utils.custom_logging import write_record_log, append_final_score
from utils import util
from .instruction_following_eval import instructions_registry


class InstructionFollowingScore(Metrics):
    """Evaluates instruction following capabilities in voice benchmarks."""

    def __init__(self):
        super().__init__()
        self.name = "instruction_following"

    def __call__(
            self,
            candidates: List[str],
            references: List[Tuple[List[str], List[Dict[str, Optional[Union[str, int]]]]]],
            *,
            instructions: Optional[List[str]] = None,
            task_name: Optional[str] = None,
            model_name: Optional[str] = None,
            model_responses: Optional[List[ModelResponse]] = None,
    ) -> dict[str, dict[str, float] | float]:
        # Compute strict and loose scores
        strict_outputs = self._compute_outputs(candidates, references, strict=True)
        loose_outputs = self._compute_outputs(candidates, references, strict=False)

        # Compute accuracy reports
        strict_report = self._compute_accuracy_report(strict_outputs)
        loose_report = self._compute_accuracy_report(loose_outputs)

        results = {
            "strict-prompt": util.smart_round(strict_report["prompt"] * 100.0 , 2),
            "strict-instruction": util.smart_round(strict_report["instruction"] * 100.0, 2),
            "loose-prompt": util.smart_round(loose_report["prompt"] * 100.0 , 2),
            "loose-instruction": util.smart_round(loose_report["instruction"] * 100.0, 2),
        }

        # Average final score over all components
        results["final"] = util.smart_round(float(np.mean(list(results.values()))), 2)

        # Compute record-level scores for strict outputs (binary: all instructions followed or not)
        record_scores = [float(all(out["follow_instruction_list"])) for out in strict_outputs]

        # Write detailed record-level logs (if task_name and model_name provided)
        if task_name and model_name:
            write_record_log(
                self,
                refs=[ref[2] for ref in references],
                cands=candidates,
                scores=record_scores,
                task_name=task_name,
                model_name=model_name,
                explanations=None,
                instructions=instructions,
                model_responses=model_responses
            )
            append_final_score(self, results, task_name, model_name, model_responses)
        return results

    def _compute_outputs(
            self,
            candidates: List[str],
            references: List[Tuple[List[str], List[Dict[str, Optional[Union[str, int]]]]]],
            strict: bool = True,
    ) -> List[dict]:
        outputs = []
        for i, candidate in enumerate(candidates):
            instruction_id_list, kwargs_list, _ = references[i]

            is_following_list = []
            for idx, instruction_id in enumerate(instruction_id_list):
                instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
                instruction = instruction_cls(instruction_id)

                kwargs = {k: v for k, v in kwargs_list[idx].items() if v is not None}
                instruction.build_description(**kwargs)

                args = instruction.get_instruction_args()
                if args and "prompt" in args:
                    instruction.build_description(prompt=candidate)

                if strict:
                    follows = (bool(candidate.strip()) and
                               bool(instruction.check_following(candidate)))
                else:
                    follows = False
                    for variant in self._generate_loose_variants(candidate):
                        if (bool(variant.strip()) and
                                bool(instruction.check_following(variant))):
                            follows = True
                            break

                is_following_list.append(follows)

            outputs.append({
                "instruction_id_list": instruction_id_list,
                "follow_instruction_list": is_following_list,
            })
        return outputs

    def _generate_loose_variants(self, response: str) -> List[str]:
        lines = response.split("\n")
        variants = [
            response,
            response.replace("*", ""),
            "\n".join(lines[1:]).strip(),
            "\n".join(lines[:-1]).strip(),
            "\n".join(lines[1:-1]).strip(),
        ]

        # Include those with asterisks removed
        variants += [v.replace("*", "") for v in variants if "*" in v]
        return list(set(variants))  # Deduplicate

    def _compute_accuracy_report(self, outputs: List[dict]) -> dict[str, float]:
        prompt_total = len(outputs)
        prompt_correct = 0
        instruction_total = 0
        instruction_correct = 0

        for example in outputs:
            follows = example["follow_instruction_list"]

            if all(follows):
                prompt_correct += 1

            instruction_total += len(follows)
            instruction_correct += sum(follows)

        return {
            "prompt": prompt_correct / prompt_total if prompt_total > 0 else 0.0,
            "instruction": (instruction_correct / instruction_total 
                            if instruction_total > 0 else 0.0),
        }

    def compute_record_level_scores(
            self,
            candidates: List[str],
            references: List[Tuple[List[str], List[Dict[str, Optional[Union[str, int]]]]]],
    ) -> List[float]:
        """Compute record-level scores for instruction following."""
        outputs = self._compute_outputs(candidates, references, strict=True)
        return [float(all(out["follow_instruction_list"])) for out in outputs]
