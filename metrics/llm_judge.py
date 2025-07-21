from __future__ import annotations
import asyncio, json, yaml, os, re
from pathlib import Path
from openai import AsyncAzureOpenAI, APIConnectionError
from tqdm import tqdm
from typing import List, Tuple, Optional
import httpx
import logging
logger = logging.getLogger(__name__)
from metrics.metrics import Metrics
from utils.custom_logging import write_record_log, append_final_score

# ---------------------------------------------------------------------------
# Helper to load prompt templates shipped with the package
# ---------------------------------------------------------------------------

_template_cache: dict[str, str] | None = None
PROMPT_FILE_PATH = Path(__file__).resolve().parents[1] / "prompts/judge_prompts.yaml"
def _get_prompt(kind: str) -> str:
    """Load and return the prompt string for *kind* every call (no caching)."""
    data = yaml.safe_load(PROMPT_FILE_PATH.read_text()) or {}
    if kind not in data:
        raise KeyError(f"Prompt '{kind}' not found in {PROMPT_FILE_PATH}")
    return data[kind]

# ---------------------------------------------------------------------------
# Base LLM judge – uses gpt-4o via async OpenAI SDK
# ---------------------------------------------------------------------------

_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_MAX_CONCURRENCY = 5

class _BaseLLMJudge(Metrics):
    """Common LLM-as-judge scaffolding."""

    def __init__(self, max_concurrency: int | None = None, model: str | None = None, *_, **__):
        super().__init__()
        # If not supplied, fall back to defaults
        self._max_concurrency = max_concurrency or _DEFAULT_MAX_CONCURRENCY
        self._model = model or _DEFAULT_OPENAI_MODEL
        # Azure OpenAI async client
        self._client = AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_KEY"),  # set in your shell
            api_version=os.environ.get("AZURE_OPENAI_VERSION", "2025-01-01-preview"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://corelmm-gpt-4t.openai.azure.com"),
        )

    async def _score_once(self, system_prompt: str, user_prompt: str) -> float | dict | None:

        max_retries = 8
        for attempt in range(max_retries):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
                content = resp.choices[0].message.content.strip()
                try:
                    return json.loads(content)
                except Exception:
                    return content
            except (APIConnectionError, httpx.ConnectError, httpx.HTTPError) as e:
                logger.warning(f"API connection failed (attempt {attempt+1}/{max_retries}): {e}")
                await asyncio.sleep(2)  # Wait before retrying
            except Exception as e:
                error_message = str(e)
                # Handle content policy violations separately
                if "content management policy" in error_message and attempt < max_retries - 1:
                    logger.warning(f"Content filter triggered (attempt {attempt+1}/{max_retries}). Modifying prompt...")
                    
                    # Apply progressively stronger modifications to avoid content filter
                    if attempt == 0:
                        # First retry: Add a safety prefix to the system prompt
                        system_prompt = f"Please provide an academic evaluation only. Avoid any harmful, unethical, or inappropriate content. {system_prompt}"
                    else:
                        # Second retry: Add a safety prefix to the user prompt
                        user_prompt = f"For academic evaluation purposes only: {user_prompt}"
                    
                    # Continue to retry with modified prompts
                    await asyncio.sleep(1)
                    continue
                else:
                    # For other unexpected errors, log and potentially break
                    logger.error(f"Unexpected error in _score_once: {e}")
                    break
        logger.error(f"All {max_retries} attempts failed for this sample. Skipping.")
        return None

    async def _judge_all(
        self,
        candidates: list[str],
        references: list[str],
    ) -> list:
        """Run the LLM judge over *candidates* vs *references*.

        *prompt_add_on* is extra text appended to the base system prompt.  Each
        concrete metric can inject task-specific instructions without changing
        the core helper.
        """
        sem = asyncio.Semaphore(self._max_concurrency)
        sys_prompt_template = _get_prompt(self._prompt_key)

        async def _run(cand, ref):
            # Use the prompt template as system prompt
            sys_prompt = sys_prompt_template
            # Format candidate and reference as user prompt
            user_prompt = f"candidate: {cand}\nreference: {ref}"
            async with sem:
                return await self._score_once(sys_prompt, user_prompt)  # Use sys_prompt_template as system prompt

        # Create tasks for all candidate-reference pairs and track their indices
        tasks_with_indices = [(i, _run(c, r)) for i, (c, r) in enumerate(zip(candidates, references))]
        pending = {asyncio.create_task(coro): i for i, coro in tasks_with_indices}
        results = [None] * len(tasks_with_indices)  # Pre-allocate results list
        
        # Use tqdm to monitor progress across asynchronous completions
        with tqdm(total=len(pending), desc=f"LLM Judge ({self._prompt_key})") as progress:
            while pending:
                done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    index = pending.pop(task)
                    result = task.result()
                    results[index] = result  # Store result at the correct index
                    progress.update(1)
        return results

    # ---------------------------------------------------
    # Internal helper for per-record logging
    # ---------------------------------------------------

#All of these judges always return on a scale of 100

# ---------------------------------------------------------------------------
# Binary (yes/no) judge
# ---------------------------------------------------------------------------

class BinaryLLMJudgeMetric(_BaseLLMJudge):  # noqa: D401
    name: str = "llm_judge_binary"
    _prompt_key: str = "binary_judge_prompt"

    def __call__(self, candidates, references, instructions=None, *, dataset_name: str | None = None, model_name: str | None = None, model_responses=None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []
        
        overall = super().get_score(candidates, references)
        if self.name in overall:
            overall[self.name] *= 100
        if dataset_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            # write_record_log will also write to run.log internally
            explanations = getattr(self, "explanations", None)
            write_record_log(self, references, candidates, scores, dataset_name, model_name, explanations, 
                           instructions=self.instructions, model_responses=self.model_responses)
            # Directly call append_final_score
            append_final_score(self, overall, dataset_name, model_name)
        return overall


    def compute_record_level_scores(self, candidates: list, references: list):
        # Here we can use self.instructions if needed
        raw_scores = asyncio.run(self._judge_all(candidates, references))
        # Expect {"score": number, "explanation": str}
        scores = [float(r.get("score", 0)) if isinstance(r, dict) else 0.0 for r in raw_scores]
        self.explanations = [r.get("explanation", "") if isinstance(r, dict) else "" for r in raw_scores]
        return {self.name: scores}

class DetailedLLMJudgeMetric(_BaseLLMJudge):
    """Detailed LLM judge metric.
    On a scale of 0 to 5, how well does the candidate text match the reference text?
    Returns:
        float: Overall score
    """
    name: str = "llm_judge_detailed"
    _prompt_key: str = "detailed_judge_prompt"

    def __call__(self, candidates, references, instructions=None, *, dataset_name: str | None = None, model_name: str | None = None, model_responses=None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        # Store instructions and model_responses for potential later use
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []
        
        overall = super().get_score(candidates, references)
        if self.name in overall:
            # From 0-5 scale to 0-100 scale
            overall[self.name] *= 20
        if dataset_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            # write_record_log will also write to run.log internally
            explanations = getattr(self, "explanations", None)
            write_record_log(self, references, candidates, scores, dataset_name, model_name, explanations, 
                          instructions=self.instructions, model_responses=self.model_responses)
            # Directly call append_final_score
            append_final_score(self, overall, dataset_name, model_name)
        return overall

    def compute_record_level_scores(self, candidates: list, references: list):
        # Here we can use self.instructions if needed
        raw_scores = asyncio.run(self._judge_all(candidates, references))
        # Expect {"score": number, "explanation": str}
        scores = [float(r.get("score", 0)) if isinstance(r, dict) else 0.0 for r in raw_scores]
        self.explanations = [r.get("explanation", "") if isinstance(r, dict) else "" for r in raw_scores]
        return {self.name: scores}

class CallHomeLLMJudgeMetric(_BaseLLMJudge):
    name: str = "llm_judge_callhome"
    _prompt_key: str = "callhome_judge_prompt"

    def __call__(self, candidates, references, instructions=None, *, dataset_name: str | None = None, model_name: str | None = None, model_responses=None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        # Store instructions and model_responses for potential later use
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []
        
        overall = super().get_score(candidates, references)
        print(overall)
        if self.name in overall:
            overall[self.name] += 1
            overall[self.name] *= 10
        if dataset_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            # write_record_log will also write to run.log internally
            explanations = getattr(self, "explanations", None)
            write_record_log(self, references, candidates, scores, dataset_name, model_name, explanations, 
                      instructions=self.instructions, model_responses=self.model_responses)
            # Directly call append_final_score
            append_final_score(self, overall, dataset_name, model_name)
        return overall

    def compute_record_level_scores(self, candidates: list, references: list):
        # Here we can use self.instructions if needed
        raw_scores = asyncio.run(self._judge_all(candidates, references))
        # Expect {"score": number, "explanation": str}
        scores = [float(r.get("score", 0)) if isinstance(r, dict) else 0.0 for r in raw_scores]
        self.explanations = [r.get("explanation", "") if isinstance(r, dict) else "" for r in raw_scores]
        return {self.name: scores}
# ---------------------------------------------------------------------------
# Original LLM judge
# ---------------------------------------------------------------------------

class LLMJudgeMetric(Metrics):  # noqa: D401
    name: str = "llm_judge"

    def __init__(self) -> None:
        self._model = None

    def __call__(self, ref: str, hyp: str, instructions=None):
        # Store instructions for potential later use
        self.instructions = instructions
        return float(ref.strip() == hyp.strip())


class BigBenchAudioLLMJudgeMetric(_BaseLLMJudge):
    """
    A judge metric for evaluating BigBenchAudio predictions using an LLM to determine correctness.

    This class compares model predictions (candidates) against references (transcript, official_answer pairs)
    using a prompt-based LLM, which returns either "CORRECT" or "INCORRECT" for each comparison.
    """
    name: str = "llm_judge_big_bench_audio"
    _prompt_key: str = "big_bench_audio_judge_prompt"

    def __call__(
        self,
        candidates: List[str],
        references: List[Tuple[str, str]],
        instructions=None,
        *,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
        model_responses=None
    ) -> dict:
        """
        Evaluate the predictions using LLM-based judgment and return overall accuracy.

        Args:
            candidates (List[str]): List of model predictions.
            references (List[Tuple[str, str]]): List of (transcript, official_answer) pairs.
            dataset_name (Optional[str]): Optional dataset name for logging purposes.
            model_name (Optional[str]): Optional model name for logging purposes.

        Returns:
            dict: Evaluation result with accuracy and counts for correct, incorrect, and failed responses.
        """
        # Store instructions and model_responses for potential later use
        self.instructions = instructions
        self.model_responses = model_responses if model_responses else []
        
        scores = self.compute_record_level_scores(candidates, references)
        all_scores = scores[self.name]

        num_correct = sum(1 for s in all_scores if s == 1.0)
        num_incorrect = sum(1 for s in all_scores if s == 0.0)
        total = num_correct + num_incorrect

        overall = {
            self.name: num_correct / total if total > 0 else 0.0,
            "num_correct": num_correct,
            "num_incorrect": num_incorrect,
            "num_failed": len(all_scores) - total,
        }

        if dataset_name and model_name:
            # write_record_log will also write to run.log internally
            write_record_log(self, references, candidates, all_scores, dataset_name, model_name, 
                       instructions=self.instructions, model_responses=self.model_responses)
            # Directly call append_final_score
            append_final_score(self, overall, dataset_name, model_name)

        return overall

    def compute_record_level_scores(
        self,
        candidates: List[str],
        references: List[Tuple[str, str]]
    ) -> dict:
        """
        Calls the LLM to judge each (transcript, prediction, official_answer) triple and returns scores.

        Args:
            candidates (List[str]): List of model predictions.
            references (List[Tuple[str, str]]): List of (transcript, official_answer) pairs.

        Returns:
            dict: A mapping from metric name to list of 1.0 (correct), 0.0 (incorrect), or None (failed).
        """
        # LLM input: (transcript, prediction, official_answer)
        raw_responses = asyncio.run(self._judge_all(candidates, references))
        scores = []

        for response in raw_responses:
            if isinstance(response, str):
                normalized = response.strip().upper()
                if normalized == "CORRECT":
                    scores.append(1.0)
                elif normalized == "INCORRECT":
                    scores.append(0.0)
                else:
                    scores.append(None)
            else:
                scores.append(None)

        self.explanations = raw_responses  # Save raw LLM responses for inspection/logging
        return {self.name: scores}

    def _append_final_score(self, overall: dict, dataset_name: str, model_name: str) -> None:
        """
        Appends the final score summary to a structured log file.

        Args:
            overall (dict): Final evaluation scores and metadata.
            dataset_name (str): Dataset identifier.
            model_name (str): Model identifier.
        """
        def _slug(text: str) -> str:
            return re.sub(r"[^A-Za-z0-9_]+", "_", text)

        log_path = Path(".") / f"{_slug(dataset_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"final_score": overall}, ensure_ascii=False) + "\n")