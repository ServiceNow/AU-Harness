from __future__ import annotations

import asyncio, json, yaml, os
from pathlib import Path

from openai import AsyncAzureOpenAI
from tqdm import tqdm  # progress bar

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

    async def _score_once(self, system_prompt: str, user_prompt: str) -> float | dict:
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

# ---------------------------------------------------------------------------
# Binary (yes/no) judge
# ---------------------------------------------------------------------------

class BinaryLLMJudgeMetric(_BaseLLMJudge):  # noqa: D401
    name: str = "llm_judge_binary"
    _prompt_key: str = "binary_judge_prompt"

    def __call__(self, candidates, references):
        # Binary judge returns a 0/1 style score list, we expose record-level directly
        return self.compute_record_level_scores(candidates, references)
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

    def __call__(self, candidates, references):
        # Detailed judge returns 0-5 score list with explanations
        return self.compute_record_level_scores(candidates, references)
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