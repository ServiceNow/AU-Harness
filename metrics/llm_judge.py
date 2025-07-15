from __future__ import annotations

import asyncio, json, yaml, os, re
from pathlib import Path
from pathlib import Path

from openai import AsyncAzureOpenAI
from tqdm import tqdm  # progress bar
import logging
logger = logging.getLogger(__name__)

from metrics.metrics import Metrics

# ---------------------------------------------------------------------------
# Helper to load prompt templates shipped with the package
# ---------------------------------------------------------------------------
from pathlib import Path

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
            api_version="2025-01-01-preview",
            azure_endpoint="https://corelmm-gpt-4t.openai.azure.com",
        )

    async def _score_once(self, system_prompt: str, user_prompt: str) -> float | dict | None:
        import asyncio
        import httpx
        import openai
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
            except (openai.APIConnectionError, httpx.ConnectError, httpx.HTTPError) as e:
                logger.warning(f"API connection failed (attempt {attempt+1}/{max_retries}): {e}")
                await asyncio.sleep(2)  # Wait before retrying
            except Exception as e:
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
            user_prompt = sys_prompt_template.format(prediction=cand, reference=ref)
            async with sem:
                return await self._score_once("", user_prompt)  # no base system prompt

        # Create tasks for all candidate-reference pairs
        tasks = [_run(c, r) for c, r in zip(candidates, references)]
        results = []
        # Use tqdm to monitor progress across asynchronous completions
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"LLM Judge ({self._prompt_key})"):
            result = await coro
            results.append(result)
        return results

    # ---------------------------------------------------
    # Internal helper for per-record logging
    # ---------------------------------------------------
    def _write_record_log(self, refs, cands, scores, dataset_name, model_name):
        if not refs or not scores:
            return
        def _slug(s):
            return re.sub(r"[^A-Za-z0-9_]+", "_", s)
        log_dir = Path("run_logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{_slug(dataset_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
        explanations = getattr(self, "explanations", [""] * len(scores))
        with open(log_path, "w", encoding="utf-8") as f:
            for ref, cand, sc, expl in zip(refs, cands, scores, explanations):
                f.write(json.dumps({"reference": ref, "candidate": cand, "score": sc, "explanation": expl}, ensure_ascii=False) + "\n")
        
        # Write to shared run.json
        self._write_to_run_json(refs, cands, scores, dataset_name, model_name)
        
    def _write_to_run_json(self, refs, cands, scores, dataset_name, model_name):
        """Write each sample's prediction to a shared run.log file that resets with every run."""
        import json
        from pathlib import Path
        
        run_path = Path("run_logs") / "run.log"
        
        # Get explanations if available
        explanations = getattr(self, "explanations", [""] * len(scores))
        
        # Open run.log in append mode
        with open(run_path, "a", encoding="utf-8") as f:
            # Add entries for this metric/dataset/model
            for ref, cand, sc, expl in zip(refs, cands, scores, explanations):
                entry = {
                    "dataset": dataset_name,
                    "metric": self.name,
                    "model": model_name,
                    "reference": ref,
                    "candidate": cand,
                    "score": sc,
                    "explanation": expl
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


#All of these judges always return on a scale of 100

# ---------------------------------------------------------------------------
# Binary (yes/no) judge
# ---------------------------------------------------------------------------

class BinaryLLMJudgeMetric(_BaseLLMJudge):  # noqa: D401
    name: str = "llm_judge_binary"
    _prompt_key: str = "binary_judge_prompt"

    def __call__(self, candidates, references, *, dataset_name: str | None = None, model_name: str | None = None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        overall = super().get_score(candidates, references)
        if self.name in overall:
            overall[self.name] *= 100
        if dataset_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            self._write_record_log(references, candidates, scores, dataset_name, model_name)
            self._append_final_score(overall, dataset_name, model_name)
        return overall

    def _append_final_score(self, overall, dataset_name, model_name):
        import json, re
        from pathlib import Path
        def _slug(s):
            return re.sub(r"[^A-Za-z0-9_]+", "_", s)
        log_dir = Path("run_logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f"{_slug(dataset_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"final_score": overall}, ensure_ascii=False) + "\n")
    def compute_record_level_scores(self, candidates: list, references: list):
        raw_scores = asyncio.run(self._judge_all(candidates, references))
        # Expect {"score": number, "explanation": str}
        scores = [float(r.get("score", 0)) if isinstance(r, dict) else 0.0 for r in raw_scores]
        self.explanations = [r.get("explanation", "") if isinstance(r, dict) else "" for r in raw_scores]
        return {self.name: scores}

# ---------------------------------------------------------------------------
# Detailed judge – returns a 0-5 score and rationale
# ---------------------------------------------------------------------------

class DetailedLLMJudgeMetric(_BaseLLMJudge):  # noqa: D401
    name: str = "llm_judge_detailed"
    _prompt_key: str = "detailed_judge_prompt"

    def __call__(self, candidates, references, *, dataset_name: str | None = None, model_name: str | None = None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        overall = super().get_score(candidates, references)
        if self.name in overall:
            overall[self.name] *= 20
        if dataset_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            self._write_record_log(references, candidates, scores, dataset_name, model_name)
            self._append_final_score(overall, dataset_name, model_name)
        return overall

    def _append_final_score(self, overall, dataset_name, model_name):
        import json, re
        from pathlib import Path
        def _slug(s):
            return re.sub(r"[^A-Za-z0-9_]+", "_", s)
        log_path = Path(".") / f"{_slug(dataset_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"final_score": overall}, ensure_ascii=False) + "\n")   
    def compute_record_level_scores(self, candidates: list, references: list):
        raw_scores = asyncio.run(self._judge_all(candidates, references))
        # Expect {"score": number, "explanation": str}
        scores = [float(r.get("score", 0)) if isinstance(r, dict) else 0.0 for r in raw_scores]
        self.explanations = [r.get("explanation", "") if isinstance(r, dict) else "" for r in raw_scores]
        return {self.name: scores}

class CallHomeLLMJudgeMetric(_BaseLLMJudge):
    name: str = "llm_judge_callhome"
    _prompt_key: str = "callhome_judge_prompt"

    def __call__(self, candidates, references, *, dataset_name: str | None = None, model_name: str | None = None):
        """Return overall average dict and record-level details. Write per-record log if dataset/model provided."""
        overall = super().get_score(candidates, references)
        print(overall)
        if self.name in overall:
            overall[self.name] += 1
            overall[self.name] *= 10
        if dataset_name and model_name:
            scores = self.record_level_scores.get(self.name, [])
            self._write_record_log(references, candidates, scores, dataset_name, model_name)
            self._append_final_score(overall, dataset_name, model_name)
        return overall

    def _append_final_score(self, overall, dataset_name, model_name):
        import json, re
        from pathlib import Path
        def _slug(s):
            return re.sub(r"[^A-Za-z0-9_]+", "_", s)
        log_path = Path(".") / f"{_slug(dataset_name)}_{_slug(self.name)}_{_slug(model_name)}.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"final_score": overall}, ensure_ascii=False) + "\n")   
    def compute_record_level_scores(self, candidates: list, references: list):
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

    def __call__(self, ref: str, hyp: str) -> float:
        return float(ref.strip() == hyp.strip())