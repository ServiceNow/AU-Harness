from __future__ import annotations

import asyncio, json, yaml, os
from pathlib import Path

import openai

from metrics.metrics import Metrics

# ---------------------------------------------------------------------------
# Helper to load prompt templates shipped with the package
# ---------------------------------------------------------------------------
from importlib.resources import files

_template_cache: dict[str, str] | None = None

def _get_prompt(kind: str) -> str:
    """Return the prompt string for *kind* from `chains/audiobench_chain.yaml`."""
    global _template_cache
    if _template_cache is None:
        yaml_path = files("audiobench.chains").joinpath("audiobench_chain.yaml")
        _template_cache = yaml.safe_load(yaml_path.read_text())
    try:
        return _template_cache[kind]
    except KeyError:
        raise ValueError(f"Prompt kind '{kind}' not found in {yaml_path}")

# ---------------------------------------------------------------------------
# Base LLM judge – uses gpt-4o via async OpenAI SDK
# ---------------------------------------------------------------------------

_OPENAI_MODEL = os.getenv("AUDIOBENCH_JUDGE_MODEL", "gpt-4o-mini")
_MAX_CONCURRENCY = int(os.getenv("AUDIOBENCH_JUDGE_CONCURRENCY", 5))

class _BaseLLMJudge(Metrics):
    """Common LLM-as-judge scaffolding."""

    def __init__(self, *_, **__):
        super().__init__()
        self._client = openai.AsyncOpenAI()  # uses env OPENAI_API_KEY

    async def _score_once(self, system_prompt: str, user_prompt: str) -> float | dict:
        resp = await self._client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        try:
            return json.loads(content)
        except Exception:
            return content

    async def _judge_all(self, candidates: list[str], references: list[str]) -> list:
        sem = asyncio.Semaphore(_MAX_CONCURRENCY)
        sys_prompt = _get_prompt(self._prompt_key)
        async def _run(cand, ref):
            user_prompt = sys_prompt.format(prediction=cand, reference=ref)
            async with sem:
                return await self._score_once("You are an evaluation assistant.", user_prompt)
        return await asyncio.gather(*[_run(c, r) for c, r in zip(candidates, references)])

# ---------------------------------------------------------------------------
# Binary (yes/no) judge
# ---------------------------------------------------------------------------

class BinaryLLMJudgeMetric(_BaseLLMJudge):  # noqa: D401
    name: str = "llm_judge_binary"
    _prompt_key: str = "binary_judge_prompt"

    def compute_record_level_scores(self, candidates: list, references: list):
        loop = asyncio.get_event_loop()
        raw_scores = loop.run_until_complete(self._judge_all(candidates, references))
        # Expect each response to be {"correct": true/false}
        bools = [bool(r.get("correct", False)) if isinstance(r, dict) else False for r in raw_scores]
        return {self.name: [1.0 if b else 0.0 for b in bools]}

# ---------------------------------------------------------------------------
# Detailed judge – returns a 0-10 score and rationale
# ---------------------------------------------------------------------------

class DetailedLLMJudgeMetric(_BaseLLMJudge):  # noqa: D401
    name: str = "llm_judge_detailed"
    _prompt_key: str = "detailed_judge_prompt"

    def compute_record_level_scores(self, candidates: list, references: list):
        loop = asyncio.get_event_loop()
        raw_scores = loop.run_until_complete(self._judge_all(candidates, references))
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