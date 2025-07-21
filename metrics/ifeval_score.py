from metrics.metrics import Metrics
from utils.custom_logging import write_record_log, append_final_score
from metrics.ifeval import instructions_registry


class IfevalScore(Metrics):
    def __call__(self, candidates, references, instructions=None, *, dataset_name: str | None = None,
                 model_name: str | None = None):
        # Store instructions for potential later use
        self.instructions = instructions
        overall = self.compute_record_level_scores(candidates, references, instructions)
        if dataset_name and model_name:
            scores = overall.get(self.name, [])
            # write_record_log will also write to run.log internally
            write_record_log(self, references, candidates, scores, dataset_name, model_name,
                             instructions=self.instructions)
            # Directly call append_final_score
            append_final_score(self, {"if_eval_strict": overall["overall"]}, dataset_name, model_name)
        return overall

    def __init__(self):
        super().__init__()
        self.name = "ifeval_score"

    def compute_record_level_scores(self, candidates: list, references: list, instructions: list) -> dict[str, list | None]:
        """Compute the scores that should be saved in the record level file.

        Args:
            candidates: Generated text from the model
            references: Reference text from the dataset

        Returns:
            Scores for each record. The keys should be the column names that will be saved in the record level file.
        """

        # TODO: Optimizing for batch processing (more efficient with GPU) later
        from tqdm import tqdm
        score_list = []
        for i in tqdm(range(len(candidates)), desc="IFEVALSCORE"):

            reference, candidate = references[i], candidates[i]
            instruction_id_list = instructions[i]['instruction_id_list']
            instructions_kwargs = instructions[i]['kwargs']
            is_following_list = []
            for index, instruction_id in enumerate(instruction_id_list):
                instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
                instruction = instruction_cls(instruction_id)
                # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
                kwargs = {k: v for k, v in instructions_kwargs[index].items() if v}
                instruction.build_description(**kwargs)
                args = instruction.get_instruction_args()
                if args and "prompt" in args:
                    instruction.build_description(prompt=candidate)

                if reference.strip() and instruction.check_following(reference):
                    is_following_list.append(True)
                else:
                    is_following_list.append(False)
            score_list.append(1 if all(is_following_list) else 0)

        overall_score = sum(score_list) / len(score_list)

        return {self.name: score_list, "overall": overall_score}
