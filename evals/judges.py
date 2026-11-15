"""judges - LLM judges, deliberately from a different model family.

Every judgement is a single ``judge.*`` root span in the same Langfuse project as
the run it judges, carrying ``run_id`` and ``judge_version``. Sharing the
project is what links a verdict to its trace; the span name is what filters it
out when you are debugging an answer.

A judge's numbers are only quotable once kappa >= 0.6 against human labels.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from datetime import datetime, timezone

from evals.dataset import Dataset, Expected, Scenario
from src import llm
from src.config import JUDGE_MODEL, JUDGE_TEMPERATURE, CORPUS_DIR
from src.documents import load_documents
from src.models.run_record import RunRecord
from src.observability import span

logger = logging.getLogger(__name__)

JUDGE_VERSION = f"{JUDGE_MODEL}-v3-whole-answer"
KAPPA_GATE = 0.6


def judge_version(metric: str) -> str:
    return JUDGE_VERSION + ("-citation-policy-context-v2" if metric == "citation_support" else "")

METRICS = ("answer_correctness", "groundedness", "citation_support")

_RUBRICS = {
    "answer_correctness": """Evaluate the entire answer for the CURRENT QUESTION in its conversation.
The policy evidence is authoritative. The reference is a fallible hint, not a
mandatory checklist. If a reference mentions a person or rule absent from the
policy, do not penalise a policy-correct answer for omitting that reference detail.
For 'summarise that', resolve 'that' from the immediately preceding discussion;
do not require the summary to cover unrelated topics in the reference.

Perform three checks independently:
1. answers_question: Does the answer satisfy the actual question? A plain
acknowledgement satisfies a user fact statement. A warranted refusal satisfies
an out-of-scope request. Exact wording and an offer of help are unnecessary.
2. claims_supported: Are ALL material additional claims supported by policy or
explicit user facts? Audit the extra details as well as the main answer.
A main fact being correct does not excuse an invented deadline or estimate.
3. conditions_preserved: Does the answer avoid treating unknown circumstances
as settled? Different outcomes depending on an unstated contract type or data
access must be conditional. General summaries need not enumerate every exception,
but applying a conditional rule to this user's specific case requires its facts.

The final score is 1 ONLY when all three checks pass; otherwise 0.
In evidence_checks, identify the most consequential extra claims and quote the
supporting policy passage, or say no supporting passage exists. This is an
evidence audit, not a request for private reasoning.

CURRENT QUESTION: {question}
RECENT CONVERSATION:
{conversation}
POLICY EVIDENCE:
{policy_evidence}
FALLIBLE REFERENCE (policy and actual conversational scope take precedence):
{reference}
REFUSAL WARRANTED: {refusal_warranted}
ANSWER TO EVALUATE:
{response}

Return ONLY JSON:
{{"checks": {{"answers_question": true, "claims_supported": true, "conditions_preserved": true}},
"evidence_checks": [{{"claim": "...", "policy_quote_or_missing": "..."}}],
"score": 0 or 1, "reason": "one sentence consistent with the checks"}}""",
    "groundedness": """You are checking whether an assistant's answer is supported by the retrieved context.

Score 1 only if every factual claim in the answer (numbers, dates, names,
conditions) appears in the context below. Score 0 if any claim is unsupported or
contradicted by the context. A refusal is grounded only if the context truly
lacks the answer.
User facts and conversation history in the context are valid evidence for
conversational claims. When refusal is warranted, a scope redirect does not
need retrieved policy evidence. Keep the score consistent with the explanation.
Refusal warranted: {refusal_warranted}

Question: {question}
Retrieved context:
{context}

Assistant answer: {response}

Return ONLY AND NOTHING ELSE a JSON object: {{"score": 0 or 1, "reason": "one sentence"}}""",
    "citation_support": """You are checking whether an assistant's cited documents actually support its answer.

Score 1 if every citation points at a document that contains the claim it is
attached to. Score 0 if a citation is missing for the key claim, or points at a
document that does not contain it.

Question: {question}
Answer: {response}
Cited documents: {citations}
Actual cited policy text:
{policy_evidence}
Agent context:
{context}
If refusal is warranted and the answer makes no policy claims, a citation is not required.
An acknowledgement of facts explicitly given by the user also needs no policy
citation. Return 1 for these not-applicable cases, not 0 for missing citations.
For substantive answers, evaluate all material claims and their attribution;
do not overlook unsupported extra claims because the principal claim is cited.
Refusal warranted: {refusal_warranted}

Return ONLY JSON: {{"score": 0 or 1, "reason": "one sentence"}}""",
}


@dataclass
class JudgeResult:
    record_id: str
    run_id: str
    metric: str
    score: int | None
    reason: str
    judge_version: str
    model: str
    created_at: str
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _policy_documents() -> dict[str, str]:
    return {doc["doc_id"]: doc["body"] for doc in load_documents(CORPUS_DIR)}


def _prompt(metric: str, record: RunRecord, expected: Expected | None) -> str:
    citations = ", ".join(record.output.citations) or "(none)"
    documents = _policy_documents()
    ids = list(dict.fromkeys(record.output.citations + (expected.docs if expected else [])))
    evidence = "\n\n".join(f"[Source: {doc_id}]\n{documents[doc_id]}" for doc_id in ids if doc_id in documents)
    context = record.prompt or "(no context retrieved)"
    if metric == "citation_support":
        # Citation support uses authoritative full policies above. Groundedness
        # separately assesses everything the agent actually retrieved.
        context = re.sub(r"=== DOCUMENTS ===.*?(?=\n=== CONVERSATION ===|\n==== CURRENT ====|\Z)",
                         "", context, flags=re.DOTALL)
    return _RUBRICS[metric].format(
        question=record.question,
        reference=(expected.reference_answer if expected else "") or "(none provided)",
        response=record.output.response or "(empty)",
        context=context,
        citations=citations,
        policy_evidence=evidence or "(no matching policy text)",
        refusal_warranted=expected.refusal_warranted if expected else False,
        conversation=record.prompt.partition("=== CONVERSATION ===")[2].partition("==== CURRENT ===")[0].strip() or "(no previous messages)",
    )


def _parse_verdict(text: str) -> tuple[int | None, str, str | None]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None, "", f"judge returned non-JSON: {text[:160]}"
    if not isinstance(parsed, dict):
        return None, "", f"judge returned {type(parsed).__name__}"
    raw = parsed.get("score")
    if type(raw) is not int or raw not in (0, 1):
        return None, str(parsed.get("reason", "")), f"unusable score: {raw!r}"
    if "checks" in parsed:
        checks = parsed["checks"]
        keys = {"answers_question", "claims_supported", "conditions_preserved"}
        if not isinstance(checks, dict) or not keys <= checks.keys() or any(type(checks[k]) is not bool for k in keys):
            return None, str(parsed.get("reason", "")), "invalid correctness checks"
        if raw != int(all(checks[k] for k in keys)):
            return None, str(parsed.get("reason", "")), "score contradicts correctness checks"
    return raw, str(parsed.get("reason", "")), None


def judge_record(
    record: RunRecord,
    scenario: Scenario,
    *,
    expected: Expected | None = None,
    metric: str,
    model: str = JUDGE_MODEL,
) -> JudgeResult:
    created = datetime.now(timezone.utc).isoformat()
    expected = expected or scenario.expected
    prompt = _prompt(metric, record, expected)

    with span(
        f"judge.{metric}",
        **{
            "run_id": record.run_id,
            "record_id": record.record_id,
            "judge_version": judge_version(metric),
            "judge.model": model,
            "judge.metric": metric,
        },
    ) as annotate:
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=JUDGE_TEMPERATURE,
            json_mode=True,
        )
        annotate(
            **{
                "judge.prompt_tokens": response.usage.prompt_tokens,
                "judge.completion_tokens": response.usage.completion_tokens,
            }
        )

    score, reason, error = _parse_verdict(response.text)
    return JudgeResult(
        record_id=record.record_id,
        run_id=record.run_id,
        metric=metric,
        score=score,
        reason=reason,
        judge_version=judge_version(metric),
        model=model,
        created_at=created,
        error=error,
    )


def judge_records(
    records: list[RunRecord],
    dataset: Dataset,
    *,
    metrics: tuple[str, ...] = METRICS,
    model: str = JUDGE_MODEL,
    only_unjudged: set[tuple[str, str]] | None = None,
    on_result=None,
) -> list[JudgeResult]:
    """Judge every record on every requested metric. Skips refusals on correctness."""
    results: list[JudgeResult] = []
    for record in records:
        if record.error is not None:
            continue
        try:
            scenario = dataset.by_id(record.scenario_id)
        except KeyError:
            continue
        expected = scenario.expected_for(record.turn_index)
        if expected is None:
            continue
        for metric in metrics:
            if only_unjudged and (record.record_id, metric) in only_unjudged:
                continue
            result = judge_record(record, scenario, expected=expected, metric=metric, model=model)
            if on_result:
                on_result(result)
            results.append(result)
    return results


def write_judgements(path, results: list[JudgeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.as_dict()) + "\n")


def append_judgement(path, result: JudgeResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.as_dict()) + "\n")


def read_judgements(path) -> list[JudgeResult]:
    if not path.exists():
        return []
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            results.append(JudgeResult(**json.loads(line)))
    return results
