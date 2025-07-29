from __future__ import annotations
import asyncio, json, yaml, os, re, math
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
    """Common LLM-as-judge class."""

    def __init__(self, max_concurrency: int | None = None, model: str | None = None, *_, **__):
        super().__init__()
        # If not supplied, fall back to defaults
        self._max_concurrency = max_concurrency or _DEFAULT_MAX_CONCURRENCY
        self._model = model or _DEFAULT_OPENAI_MODEL
        self._request_manager = None # Set in Engine
        # Azure OpenAI async client
        self._client = AsyncAzureOpenAI(
            api_key=os.environ.get("AZURE_OPENAI_KEY"),  # set in your shell
            api_version=os.environ.get("AZURE_OPENAI_VERSION", "2025-01-01-preview"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://corelmm-gpt-4t.openai.azure.com"),
        )
    
    def set_request_manager(self, manager):
        """Set the request manager and register the model type."""
        self._request_manager = manager
        if self._request_manager is not None:
            # Register the model type directly with the central controller
            self._request_manager.central_controller.register_model_type(
                self._model, self._max_concurrency
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
        dataset_name: str | None = None,
        model_name: str | None = None,
    ) -> list:
        """Run the LLM judge over *candidates* vs *references*.

        *prompt_add_on* is extra text appended to the base system prompt. Each
        concrete metric can inject task-specific instructions without changing
        the core helper.
        """
        if self._request_manager is None:
            raise ValueError("Request manager must be set before calling _judge_all")
        
        # Setup for token management
        token_sem = asyncio.Semaphore(0)  # Start with 0 tokens
        pending_samples = list(range(len(candidates)))
        completed_samples = set()
        results = [None] * len(candidates)
        sys_prompt_template = _get_prompt(self._prompt_key)
        
        # Generate unique evaluator instance ID
        evaluator_id = f"llm_judge_{self._prompt_key}_{id(self)}"
        
        # Continuously ask for tokens based on pending samples
        # Have dynamic wait times for by dataset size - this "layering" of Engine priority gives models in the same Engine similar priority, so they don't wait on each other as often
        async def token_manager():
            request_count = 0
            # Calculate wait times based on dataset size
            dataset_size = len(candidates)
            
            # Calculate a scale factor from 0 to 1 based on dataset size
            scale_factor = min(1.0, max(0.0, math.log10(dataset_size + 10) / 4.0))
            
            # Scale the no-token wait time between 0.5s and 2s
            no_token_wait = scale_factor * 2.0
            
            # Double the wait time when tokens are granted
            token_wait = no_token_wait * 2.0
            
            while len(pending_samples) > 0:
                request_count += 1
                # Request as many tokens as needed for pending samples, up to max_concurrency
                request_amount = min(self._max_concurrency, len(pending_samples))
                
                if request_amount > 0:
                    granted = await self._request_manager.request_tokens(
                        self._model, evaluator_id, request_amount)
                    
                    if granted > 0:
                        # Remove samples from pending list based on granted tokens
                        # This doesn't affect the pending_samples list we iterate over in other functions
                        for _ in range(granted):
                            if pending_samples:
                                # Release semaphore permits for each granted token
                                token_sem.release()
                        # Wait based on dataset size when tokens were granted
                        await asyncio.sleep(token_wait)
                    else:
                        # Backoff when no tokens were granted, based on dataset size
                        # Apply a small multiplier for repeated failures, but cap it
                        backoff_multiplier = min(3.0, 1.0 + (request_count / 10))
                        await asyncio.sleep(no_token_wait * backoff_multiplier)
                else:
                    # This shouldn't happen with the loop condition, but just in case
                    break
        
        # Start the token management task
        token_task = asyncio.create_task(token_manager())
        
        # Process each candidate-reference pair with token management
        async def _evaluate_with_token_mgmt(idx, cand, ref):
            # Acquire a token
            await token_sem.acquire()
            
            try:
                # Use the prompt template as system prompt
                sys_prompt = sys_prompt_template
                # Format candidate and reference as user prompt
                user_prompt = f"candidate: {cand}\nreference: {ref}"
                
                # Call the scoring function
                result = await self._score_once(sys_prompt, user_prompt)
                
                # Add to completed set and remove from pending
                completed_samples.add(idx)
                if idx in pending_samples:
                    pending_samples.remove(idx)
                
                # Return token to model's pool
                await self._request_manager.return_tokens(self._model, evaluator_id, 1)
                
                return idx, result
            except Exception as e:
                # Make sure to return token on error
                logger.error(f"[_BaseLLMJudge._evaluate_with_token_mgmt] Error processing sample {idx}: {e}")
                completed_samples.add(idx)
                if idx in pending_samples:
                    pending_samples.remove(idx)
                await self._request_manager.return_tokens(self._model, evaluator_id, 1)
                return idx, None
        
        # Create tasks for all samples
        tasks = [_evaluate_with_token_mgmt(i, c, r) for i, (c, r) in enumerate(zip(candidates, references))]
        
        # Use tqdm to show progress
        desc = f"{self._prompt_key}"
        if dataset_name and model_name:
            desc = f"Evaluating {dataset_name} | {model_name} | {desc}"
        with tqdm(total=len(tasks), desc=desc) as progress_bar:
            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                results[idx] = result
                progress_bar.update(1)
        
        # Cancel the token_manager task when all samples are processed
        token_task.cancel()
        try:
            await token_task
        except asyncio.CancelledError:
            pass
        
        return results


#All of these judges always return on a scale of 100

# ---------------------------------------------------------------------------
# Binary (yes/no) judge
# ---------------------------------------------------------------------------

class BinaryLLMJudgeMetric(_BaseLLMJudge):  # noqa: D401
    name: str = "llm_judge_binary"
    _prompt_key: str = "binary_judge_prompt"

    async def __call__(self, candidates, references, instructions=None, *, dataset_name: str | None = None, model_name: str | None = None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        self.instructions = instructions
        overall = await super().get_score(candidates, references, dataset_name, model_name)
        if self.name in overall:
            overall[self.name] *= 100
        if dataset_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            # write_record_log will also write to run.log internally
            explanations = getattr(self, "explanations", None)
            write_record_log(self, references, candidates, scores, dataset_name, model_name, explanations, instructions=self.instructions)
            # Directly call append_final_score
            append_final_score(self, overall, dataset_name, model_name)
        return overall

    async def compute_record_level_scores(self, candidates: list, references: list, dataset_name: str | None = None, model_name: str | None = None):
        # Here we can use self.instructions if needed
        raw_scores = await self._judge_all(candidates, references, dataset_name, model_name)
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

    async def __call__(self, candidates, references, instructions=None, *, dataset_name: str | None = None, model_name: str | None = None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        # Store instructions for potential later use
        self.instructions = instructions
        overall = await super().get_score(candidates, references, dataset_name, model_name)
        if self.name in overall:
            overall[self.name] *= 20
        if dataset_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            # write_record_log will also write to run.log internally
            explanations = getattr(self, "explanations", None)
            write_record_log(self, references, candidates, scores, dataset_name, model_name, explanations, instructions=self.instructions)
            # Directly call append_final_score
            append_final_score(self, overall, dataset_name, model_name)
        return overall

    async def compute_record_level_scores(self, candidates: list, references: list, dataset_name: str | None = None, model_name: str | None = None):
        # Here we can use self.instructions if needed
        raw_scores = await self._judge_all(candidates, references, dataset_name, model_name)
        # Expect {"score": number, "explanation": str}
        scores = [float(r.get("score", 0)) if isinstance(r, dict) else 0.0 for r in raw_scores]
        self.explanations = [r.get("explanation", "") if isinstance(r, dict) else "" for r in raw_scores]
        return {self.name: scores}


class CallHomeLLMJudgeMetric(_BaseLLMJudge):
    name: str = "llm_judge_callhome"
    _prompt_key: str = "callhome_judge_prompt"

    async def __call__(self, candidates, references, instructions=None, *, dataset_name: str | None = None, model_name: str | None = None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        # Store instructions for potential later use
        self.instructions = instructions
        overall = await super().get_score(candidates, references, dataset_name, model_name)
        if self.name in overall:
            overall[self.name] += 1
            overall[self.name] *= 10
        if dataset_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            # write_record_log will also write to run.log internally
            explanations = getattr(self, "explanations", None)
            write_record_log(self, references, candidates, scores, dataset_name, model_name, explanations, instructions=self.instructions)
            # Directly call append_final_score
            append_final_score(self, overall, dataset_name, model_name)
        return overall
        
    async def compute_record_level_scores(self, candidates: list, references: list, dataset_name: str | None = None, model_name: str | None = None):
        # Here we can use self.instructions if needed
        raw_scores = await self._judge_all(candidates, references, dataset_name, model_name)
        # Expect {"score": number, "explanation": str}
        scores = [float(r.get("score", 0)) if isinstance(r, dict) else 0.0 for r in raw_scores]
        self.explanations = [r.get("explanation", "") if isinstance(r, dict) else "" for r in raw_scores]
        return {self.name: scores}


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

    async def __call__(
        self,
        candidates: List[str],
        references: List[Tuple[str, str]],
        instructions=None,
        *,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None
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
        # Store instructions for potential later use
        self.instructions = instructions

        scores = await self.compute_record_level_scores(candidates, references, dataset_name, model_name)
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
            write_record_log(self, references, candidates, all_scores, dataset_name, model_name, instructions=self.instructions)
            # Directly call append_final_score
            append_final_score(self, overall, dataset_name, model_name)

        return overall

    async def compute_record_level_scores(
        self,
        candidates: List[str],
        references: List[Tuple[str, str]],
        dataset_name: str | None = None,
        model_name: str | None = None
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
        raw_responses = await self._judge_all(candidates, references, dataset_name, model_name)
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