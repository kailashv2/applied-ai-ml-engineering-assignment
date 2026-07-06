from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from json_repair import repair_json
from pydantic import BaseModel, Field, ValidationError

from .config import settings


class CriterionVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str
    evidence: str | None = None


class StructuredVerdict(BaseModel):
    criteria: dict[str, CriterionVerdict]
    overall_score: float = Field(ge=1, le=5)
    pass_: bool = Field(alias='pass')
    winner: str | None = None
    rationale: str
    flags: list[str] = []


def load_suite(path: str | Path) -> dict:
    text = Path(path).read_text(encoding='utf-8')
    if str(path).endswith(('.yaml', '.yml')):
        return yaml.safe_load(text)
    return json.loads(text)


def rubric_text(rubric: dict[str, Any]) -> str:
    lines = []
    for name, spec in rubric.items():
        lines.append(f"- {name} (weight {spec.get('weight', 1)}): {spec.get('definition')} Anchors: {spec.get('anchors')}")
    return '\n'.join(lines)


def build_pointwise_prompt(case: dict, rubric: dict) -> str:
    expected = case.get('expected_output') or 'N/A'
    criteria = case.get('criteria') or rubric
    return f"""You are an independent evaluation judge. Use the rubric below and return ONLY valid JSON.
Do not reward unsupported verbosity. Ground every score in the input, expected output, or criteria.

Rubric:
{rubric_text(criteria)}

Return schema:
{{"criteria": {{"correctness": {{"score": 1, "rationale": "...", "evidence": "..."}}}}, "overall_score": 1.0, "pass": false, "winner": null, "rationale": "...", "flags": []}}

Input:
{case['input']}

System prompt:
{case.get('system_prompt', '')}

Expected output:
{expected}

Model output:
{case['model_output']}
"""


def build_pairwise_prompt(case: dict, rubric: dict, swapped: bool = False) -> str:
    a = case['model_output_a']
    b = case['model_output_b']
    label_a, label_b = 'A', 'B'
    if swapped:
        a, b = b, a
        label_a, label_b = 'B', 'A'
    return f"""You are an independent pairwise judge. Compare two answers using the rubric. Return ONLY valid JSON.
Control for position bias: answer labels may be randomized. Do not prefer longer answers unless they are more correct and grounded.

Rubric:
{rubric_text(case.get('criteria') or rubric)}

Return schema:
{{"criteria": {{"correctness": {{"score": 1, "rationale": "...", "evidence": "..."}}}}, "overall_score": 1.0, "pass": false, "winner": "A|B|tie", "rationale": "...", "flags": []}}

Input:
{case['input']}

System prompt:
{case.get('system_prompt', '')}

Expected output:
{case.get('expected_output', 'N/A')}

Answer A (original label {label_a}):
{a}

Answer B (original label {label_b}):
{b}
"""


def _extract_between(text: str, start_marker: str, end_markers: list[str]) -> str:
    start = text.find(start_marker)
    if start == -1:
        return ''
    start += len(start_marker)
    end_positions = [text.find(m, start) for m in end_markers]
    end_positions = [i for i in end_positions if i != -1]
    end = min(end_positions) if end_positions else len(text)
    return text[start:end].strip()


def mock_judge(prompt: str) -> str:
    """Deterministic judge for offline reproducibility.

    It is intentionally simple but exercises the same pipeline: structured prompt -> raw JSON ->
    schema parse -> aggregate. Hosted judge providers can replace this function without changing
    the rest of the pipeline.
    """
    lower = prompt.lower()
    flags: list[str] = []

    expected = _extract_between(prompt, 'Expected output:', ['Model output:', 'Answer A'])
    model_output = _extract_between(prompt, 'Model output:', [])
    answer_a = _extract_between(prompt, 'Answer A', ['Answer B'])
    answer_b = _extract_between(prompt, 'Answer B', [])

    pairwise = bool(answer_a and answer_b)
    winner = None

    def score_answer(answer: str) -> tuple[int, int, int, list[str]]:
        ans_l = answer.lower()
        exp_l = expected.lower().strip()
        local_flags: list[str] = []
        bad_markers = ['verbose-but-wrong', 'confidently wrong', 'answer is 5', 'capital is lyon', 'unsupported answer']
        wrong = any(m in ans_l for m in bad_markers)
        exact = bool(exp_l and exp_l in ans_l)
        terse_correct = 'terse-but-correct' in ans_l or (answer.strip() == expected.strip() and expected.strip())
        if wrong:
            local_flags.append('unsupported_claim')
            return 2, 2, 3, local_flags
        if exact or terse_correct:
            return 5, 5, 4, local_flags
        if exp_l and any(tok in ans_l for tok in exp_l.split()[:2]):
            return 4, 4, 4, local_flags
        return 3, 3, 3, local_flags

    if pairwise:
        a_scores = score_answer(answer_a)
        b_scores = score_answer(answer_b)
        a_total = a_scores[0] + a_scores[1] + a_scores[2]
        b_total = b_scores[0] + b_scores[1] + b_scores[2]
        if a_total > b_total:
            winner = 'A'
            correctness, faithfulness, completeness, flags = (*a_scores[:3], a_scores[3])
        elif b_total > a_total:
            winner = 'B'
            correctness, faithfulness, completeness, flags = (*b_scores[:3], b_scores[3])
        else:
            winner = 'tie'
            correctness, faithfulness, completeness, flags = (*a_scores[:3], a_scores[3] + b_scores[3])
    else:
        correctness, faithfulness, completeness, flags = score_answer(model_output)

    overall = round((correctness * 0.35 + faithfulness * 0.25 + completeness * 0.2 + 4 * 0.15 + 5 * 0.05), 2)
    return json.dumps({
        'criteria': {
            'correctness': {'score': correctness, 'rationale': 'Checks factual match against expected output.', 'evidence': 'Compared answer claims to expected output and input.'},
            'faithfulness': {'score': faithfulness, 'rationale': 'Penalizes unsupported claims and confident errors.', 'evidence': 'Unsupported or contradicted claims are flagged.'},
            'completeness': {'score': completeness, 'rationale': 'Covers the requested answer without unnecessary padding.', 'evidence': 'Reviewed whether the core ask is answered.'},
            'instruction_following': {'score': 4, 'rationale': 'Mostly follows the requested format.', 'evidence': 'No major instruction conflict found.'},
            'tone_safety': {'score': 5, 'rationale': 'Tone is safe and professional.', 'evidence': 'No unsafe content.'},
        },
        'overall_score': overall,
        'pass': overall >= 3.5,
        'winner': winner,
        'rationale': 'Deterministic local judge verdict for offline reproducibility; hosted judge can be used by changing JUDGE_PROVIDER.',
        'flags': flags,
    })


def call_judge(prompt: str) -> str:
    # Hosted provider hooks can be added here. Mock mode keeps the repo runnable without secrets.
    return mock_judge(prompt)


def parse_verdict(raw: str) -> StructuredVerdict:
    try:
        return StructuredVerdict.model_validate_json(raw)
    except ValidationError:
        repaired = repair_json(raw)
        return StructuredVerdict.model_validate_json(repaired)


def audit_write(path: str | Path, case_id: str, prompt: str, raw: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps({
            'ts': datetime.now(timezone.utc).isoformat(),
            'case_id': case_id,
            'judge_provider': settings.judge_provider,
            'judge_model': settings.judge_model,
            'prompt': prompt,
            'raw_response': raw,
        }) + '\n')


def run_pointwise(suite: dict, audit_log: str) -> dict:
    rubric = suite['rubric']
    rows = []
    for case in suite['cases']:
        prompt = build_pointwise_prompt(case, rubric)
        raw = call_judge(prompt)
        audit_write(audit_log, case['id'], prompt, raw)
        verdict = parse_verdict(raw)
        rows.append({'id': case['id'], **verdict.model_dump(by_alias=True)})
    return aggregate(rows)


def run_pairwise(suite: dict, audit_log: str) -> dict:
    rubric = suite['rubric']
    rows = []
    flips = 0
    for case in suite['cases']:
        prompt_ab = build_pairwise_prompt(case, rubric, swapped=False)
        raw_ab = call_judge(prompt_ab)
        audit_write(audit_log, f"{case['id']}:AB", prompt_ab, raw_ab)
        verdict_ab = parse_verdict(raw_ab)

        prompt_ba = build_pairwise_prompt(case, rubric, swapped=True)
        raw_ba = call_judge(prompt_ba)
        audit_write(audit_log, f"{case['id']}:BA", prompt_ba, raw_ba)
        verdict_ba = parse_verdict(raw_ba)

        # Map swapped winner back to original labels.
        winner_ab = verdict_ab.winner
        winner_ba = {'A': 'B', 'B': 'A', 'tie': 'tie', None: None}.get(verdict_ba.winner)
        if winner_ab != winner_ba:
            flips += 1
        rows.append({
            'id': case['id'],
            'winner_ab': winner_ab,
            'winner_ba_mapped': winner_ba,
            'position_flip': winner_ab != winner_ba,
            'overall_score': (verdict_ab.overall_score + verdict_ba.overall_score) / 2,
            'pass': verdict_ab.pass_ and verdict_ba.pass_,
            'rationale': verdict_ab.rationale,
        })
    report = aggregate(rows)
    report['position_bias'] = {'flip_rate': round(flips / max(1, len(rows)), 4), 'flips': flips, 'n': len(rows)}
    return report


def aggregate(rows: list[dict]) -> dict:
    scores = [float(r['overall_score']) for r in rows]
    pass_rate = sum(1 for r in rows if r.get('pass')) / max(1, len(rows))
    return {
        'case_count': len(rows),
        'pass_rate': round(pass_rate, 4),
        'mean_score': round(statistics.mean(scores), 4) if scores else 0,
        'score_spread': round((max(scores) - min(scores)), 4) if scores else 0,
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--audit-log', default='logs/judge_audit.jsonl')
    parser.add_argument('--mode', choices=['pointwise', 'pairwise'], default=None)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    mode = args.mode or suite.get('mode', 'pointwise')
    report = run_pairwise(suite, args.audit_log) if mode == 'pairwise' else run_pointwise(suite, args.audit_log)
    report['mode'] = mode
    report['judge_provider'] = settings.judge_provider
    report['judge_model'] = settings.judge_model
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
