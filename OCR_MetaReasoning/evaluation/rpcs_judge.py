import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from OCR_MetaReasoning.evaluation.answer_utils import extract_json_text, mean
from OCR_MetaReasoning.evaluation.api_client import ChatCompletionClient, extract_message_text
from OCR_MetaReasoning.evaluation.config import config
from OCR_MetaReasoning.evaluation.dataset_processor import image_to_data_url


RPCS_CRITERIA = [
    "capability_match",
    "groundedness",
    "step_completeness",
    "non_hallucination",
]

CAPABILITY_DEFINITIONS = {
    "meta_deductive": (
        "Meta-Deductive Reasoning follows H + R -> O. A compliant process uses explicit "
        "rules, clauses, formulas, legends, field dependencies, or visible structural "
        "constraints; proposes or selects a testable candidate/case; binds the rule to "
        "image-grounded evidence; propagates consequences; performs consistency checking "
        "or self-correction when needed; and derives the uniquely rule-supported conclusion."
    ),
    "meta_inductive": (
        "Meta-Inductive Reasoning follows H + O -> R. A compliant process aligns multiple "
        "visible local observations, repeated instances, structural templates, rows/columns, "
        "or visual-text mappings; abstracts an unstated stable pattern or rule; tests the "
        "candidate pattern against the support instances; and generalizes it to a missing, "
        "unlabeled, unseen, or future target."
    ),
    "meta_abductive": (
        "Meta-Abductive Reasoning follows O + R -> H. A compliant process starts from an "
        "observed result, anomaly, goal, consequence, discrepancy, or masked/hidden element; "
        "uses visible rules, constraints, dependencies, totals, or local evidence to trace "
        "backward; generates and tests hidden hypotheses; checks coverage and consistency; "
        "and recovers the unique or minimal hidden premise, field, condition, event, cause, "
        "or explanation."
    ),
}

REFERENCE_COMPARISON_GUIDANCE = {
    "meta_deductive": (
        "The reference path should show explicit rule binding, candidate/condition checking, "
        "and elimination or verification against the rule. Use it to recognize the main process "
        "moves expected for this sample."
    ),
    "meta_inductive": (
        "The reference path should show multi-instance comparison, pattern abstraction, and "
        "generalization from observed examples. Use it to recognize the visible support instances "
        "and the intended pattern-generalization process."
    ),
    "meta_abductive": (
        "The reference path should begin from the observed result/anomaly/goal/consequence, "
        "trace visible constraints backward, recover the hidden cause or missing premise, "
        "and validate that the explanation covers the key observations."
    ),
}


def parse_meta_reasoning_types(values: List[str]) -> List[str]:
    if not values or "all" in values:
        return list(config.META_REASONING_TYPES)
    invalid = [value for value in values if value not in config.META_REASONING_TYPES]
    if invalid:
        raise ValueError(f"Invalid meta reasoning type(s): {invalid}")
    return values


def load_result_payload(result_file: Path) -> Dict[str, Any]:
    with result_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload
    if isinstance(payload, list):
        return {"results": payload}
    raise ValueError(f"Unsupported result file format: {result_file}")


def load_reference_samples(meta_reasoning_type: str) -> Dict[str, Dict[str, Any]]:
    dataset_file = config.dataset_file(meta_reasoning_type)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    reference_map: Dict[str, Dict[str, Any]] = {}
    with dataset_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            sample_id = sample.get("sample_id") or sample.get("id")
            if sample_id is not None:
                reference_map[str(sample_id)] = sample
    return reference_map


def load_existing_judgments(output_file: Path) -> Dict[str, Dict[str, Any]]:
    if not output_file.exists():
        return {}
    try:
        with output_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}

    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return {
            item.get("sample_id", item.get("id")): item
            for item in payload["results"]
            if item.get("sample_id", item.get("id")) is not None
        }
    if isinstance(payload, list):
        return {
            item.get("sample_id", item.get("id")): item
            for item in payload
            if item.get("sample_id", item.get("id")) is not None
        }
    return {}


def build_judge_content(sample: Dict[str, Any], include_image: bool) -> List[Dict[str, Any]]:
    prompt = build_judge_prompt(sample)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    if include_image:
        image_path = config.image_path(sample["meta_reasoning_type"], sample["image"])
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})
    return content


def build_judge_prompt(sample: Dict[str, Any]) -> str:
    meta_reasoning_type = sample.get("meta_reasoning_type", "")
    capability_definition = CAPABILITY_DEFINITIONS.get(
        meta_reasoning_type,
        "Use the sample's declared meta_reasoning_type as the target process.",
    )
    reference_guidance = REFERENCE_COMPARISON_GUIDANCE.get(
        meta_reasoning_type,
        "Compare the model process against the provided reference reasoning steps.",
    )
    reference_reasoning_steps = format_reasoning_steps(sample.get("reference_reasoning_steps", []))
    model_process = strip_final_answer_lines(sample.get("model_output", ""))
    return f"""You are an impartial judge for OCR-MetaReasoning.

Your task is to evaluate only the model's visible solution process. Do not evaluate whether the final answer is correct. Do not compare against a ground-truth answer, predicted answer, main task score, or reference answer; those answer fields are intentionally not provided to you. A process can receive a high RPCS even if its final answer is wrong, and a process can receive a low RPCS even if its final answer is correct.

Important: judge the explicit reasoning in model_output against the target meta-reasoning definition and the reference_reasoning_steps below. The reference steps are a process outline for this sample, not an answer key. A model should receive credit for an equivalent evidence-grounded process even when it compresses, reorders, or phrases the steps differently. Do not require exact canonical wording or an exact final answer match.
Scoring policy:
- Score only process quality: reasoning direction, evidence grounding, coverage of the required inferential moves, and absence of invented process evidence.
- Ignore the final answer line itself. Use it only if it contains additional reasoning claims; never score the value/label as right or wrong.
- Allow compressed, reordered, or paraphrased reasoning when the process still makes the required inferential moves clear.
- Follow the benchmark documents' RPCS definition: capability_match checks the dominant meta-reasoning direction; groundedness checks whether key reasoning claims trace to image evidence or question constraints; step_completeness checks whether necessary intermediate reasoning nodes are covered; non_hallucination checks whether the process avoids image-irrelevant, nonexistent, or unverifiable information.

Target meta_reasoning_type: {meta_reasoning_type}
Target process definition: {capability_definition}
Reference comparison guidance: {reference_guidance}

Score the four binary criteria below. Use only 0 or 1 for each criterion.

1. capability_match:
   Award 1 when the dominant reasoning direction in the solution process matches the target meta_reasoning_type and includes the essential inferential operations needed for this sample, as indicated by the target definition and reference_reasoning_steps.
   - meta_deductive requires an H + R -> O process: candidate/case proposal or selection, explicit rule binding, consequence propagation, and consistency checking or self-correction.
   - meta_inductive requires an H + O -> R process: multi-instance alignment, pattern abstraction, candidate-rule testing, generalization, and validation.
   - meta_abductive requires an O + R -> H process: starting from a result/anomaly/goal/consequence, reverse/backward reasoning through visible constraints, hidden-premise recovery, coverage checking, and minimal-explanation selection.

2. groundedness:
   Award 1 when the key reasoning claims are traceable to image evidence or question constraints. This covers process-relevant numbers, labels, rows, columns, fields, rules, relations, comparisons, observed results, and hidden-target constraints.
   The process may use reference_reasoning_steps to clarify what evidence should be bound, but the visible reasoning should still connect its key claims to the image/question evidence.

3. step_completeness:
   Award 1 when the output covers the necessary intermediate reasoning nodes for the sample's meta-reasoning type.
   Use reference_reasoning_steps as anchors for necessary nodes, not as a required script. Compressed, merged, or synonymous steps can receive credit when they retain the key evidence bindings and reasoning direction.
   Do not penalize solely because the final answer is missing or incorrect.

4. non_hallucination:
   Award 1 when the process avoids introducing information that is unrelated to the image, nonexistent in the image, or unverifiable from the image/question/reference process.
   This includes invented rules, fields, instances, labels, values, entities, explanations, shortcut assumptions, alternative rules, and unsupported evidence. Do not count a bare final-answer mismatch as hallucination; only process claims and evidence claims matter.

Judging guidance:
- Judge the model_output as written and focus on the reasoning it explicitly presents.
- Use the image, question, and reference_reasoning_steps as the evidence context for understanding the process.
- Ignore formatting style unless it prevents understanding.
- Use the reference_reasoning_steps to identify required process anchors, but accept compressed, merged, or synonymous process paths that preserve the key evidence bindings and reasoning direction.
- Do not use answer correctness, answer mismatch, predicted_answer, ground_truth, or main_score in any criterion.
- The final JSON must be machine parseable and must not be wrapped in markdown.

Return exactly this JSON schema:
{{
  "capability_match": 0,
  "groundedness": 0,
  "step_completeness": 0,
  "non_hallucination": 0,
  "rpcs": 0,
  "rationale": {{
    "capability_match": "one concise sentence",
    "groundedness": "one concise sentence",
    "step_completeness": "one concise sentence",
    "non_hallucination": "one concise sentence"
  }}
}}

Sample metadata:
- sample_id: {sample.get("sample_id", sample.get("id", ""))}
- reasoning_taxonomy: {sample.get("reasoning_taxonomy", "")}
- answer_type: {sample.get("answer_type", "")}
- metric: {sample.get("metric", "")}
- reference_reasoning_steps:
{reference_reasoning_steps}

Question:
{sample.get("question", "")}

Model output to judge:
{model_process}
"""


def strip_final_answer_lines(text: str) -> str:
    """Remove standalone final-answer lines so RPCS judges the reasoning process only."""
    if not isinstance(text, str):
        return ""
    lines = []
    answer_line_pattern = re.compile(
        r"^\s*(?:final\s+answer|answer|答案|最终答案)\s*[:：].*$",
        flags=re.IGNORECASE,
    )
    for line in text.splitlines():
        if answer_line_pattern.match(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def format_reasoning_steps(steps: Any) -> str:
    if isinstance(steps, list):
        if not steps:
            return "(none)"
        return "\n".join(f"{idx}. {str(step).strip()}" for idx, step in enumerate(steps, start=1))
    if isinstance(steps, str):
        text = steps.strip()
        return text if text else "(none)"
    return "(none)"


def parse_judge_response(text: str) -> Dict[str, Any]:
    json_text = extract_json_text(text) or extract_first_json_object(text)
    if not json_text:
        raise ValueError(f"Judge response did not contain JSON: {text[:500]}")
    data = json.loads(json_text)
    if not isinstance(data, dict):
        raise ValueError("Judge JSON must be an object.")

    parsed: Dict[str, Any] = {}
    for criterion in RPCS_CRITERIA:
        value = data.get(criterion)
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, str) and value.strip() in {"0", "1"}:
            value = int(value.strip())
        if value not in {0, 1}:
            raise ValueError(f"Invalid {criterion} score: {value!r}")
        parsed[criterion] = int(value)

    expected_total = sum(parsed[criterion] for criterion in RPCS_CRITERIA)
    rpcs = data.get("rpcs", expected_total)
    if isinstance(rpcs, str) and rpcs.strip().isdigit():
        rpcs = int(rpcs.strip())
    if rpcs != expected_total:
        rpcs = expected_total
    parsed["rpcs"] = expected_total

    rationale = data.get("rationale", {})
    parsed["rationale"] = rationale if isinstance(rationale, dict) else {}
    return parsed


def extract_first_json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0).strip() if match else ""


class RPCSJudge:
    def __init__(
        self,
        source_model_name: str,
        judge_model_config: Dict[str, Any],
        include_image: bool = True,
    ) -> None:
        self.source_model_name = source_model_name
        self.judge_model_name = judge_model_config["model_name"]
        self.temperature = judge_model_config.get("temperature", 0.0)
        self.top_p = judge_model_config.get("top_p")
        self.top_k = judge_model_config.get("top_k")
        self.repetition_penalty = judge_model_config.get("repetition_penalty")
        self.presence_penalty = judge_model_config.get("presence_penalty")
        self.max_tokens = judge_model_config.get("max_tokens")
        self.include_image = include_image
        self.client = ChatCompletionClient(
            api_key=judge_model_config["api_key"],
            base_url=judge_model_config["base_url"],
            model_name=judge_model_config["model_name"],
            max_retries=judge_model_config.get("max_retries", 3),
            timeout=judge_model_config.get("timeout", 600.0),
        )
        self.output_dir = config.rpcs_result_dir(source_model_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def source_result_file(self, meta_reasoning_type: str) -> Path:
        return config.RESULT_ROOT / f"{self.source_model_name}_result" / f"{meta_reasoning_type}_infer_result.json"

    def output_file(self, meta_reasoning_type: str) -> Path:
        return self.output_dir / f"{meta_reasoning_type}_rpcs_judge_result.json"

    def judge_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = sample.get("sample_id") or sample.get("id")
        base = build_result_base(sample, self.judge_model_name)

        try:
            if sample.get("status") != "success":
                return {
                    **base,
                    "rpcs": 0,
                    "criterion_scores": {criterion: 0 for criterion in RPCS_CRITERIA},
                    "rationale": {},
                    "usage": {},
                    "judge_status": "skipped",
                    "error_msg": "source inference status is not success",
                }
            if not sample.get("model_output", "").strip():
                return {
                    **base,
                    "rpcs": 0,
                    "criterion_scores": {criterion: 0 for criterion in RPCS_CRITERIA},
                    "rationale": {},
                    "usage": {},
                    "judge_status": "skipped",
                    "error_msg": "empty model_output",
                }
            if not strip_final_answer_lines(sample.get("model_output", "")).strip():
                return {
                    **base,
                    "rpcs": 0,
                    "criterion_scores": {criterion: 0 for criterion in RPCS_CRITERIA},
                    "rationale": {},
                    "usage": {},
                    "judge_status": "skipped",
                    "error_msg": "model_output contains no visible reasoning process",
                }

            messages = [{"role": "user", "content": build_judge_content(sample, self.include_image)}]
            response = self.client.create(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                presence_penalty=self.presence_penalty,
            )
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            judge_output = extract_message_text(message).strip()
            parsed = parse_judge_response(judge_output)

            return {
                **base,
                "rpcs": parsed["rpcs"],
                "criterion_scores": {
                    criterion: parsed[criterion] for criterion in RPCS_CRITERIA
                },
                "rationale": parsed.get("rationale", {}),
                "judge_output": judge_output,
                "usage": response.get("usage", {}),
                "judge_status": "success",
            }
        except Exception as exc:
            return {
                **base,
                "rpcs": 0,
                "criterion_scores": {criterion: 0 for criterion in RPCS_CRITERIA},
                "rationale": {},
                "usage": {},
                "judge_status": "fail",
                "error_msg": str(exc),
            }

    def process_meta_reasoning_type(
        self,
        meta_reasoning_type: str,
        max_workers: int,
        batch_size: int,
        sample_num: Optional[int],
        retry_all: bool,
    ) -> Dict[str, int]:
        source_file = self.source_result_file(meta_reasoning_type)
        if not source_file.exists():
            raise FileNotFoundError(f"Source result file not found: {source_file}")

        source_payload = load_result_payload(source_file)
        reference_map = load_reference_samples(meta_reasoning_type)
        samples = [
            attach_reference_reasoning_steps(item, reference_map)
            for item in source_payload.get("results", [])
            if item.get("meta_reasoning_type") == meta_reasoning_type
        ]
        output_file = self.output_file(meta_reasoning_type)
        existing = {} if retry_all else load_existing_judgments(output_file)

        to_process = []
        for sample in samples:
            sample_id = sample.get("sample_id") or sample.get("id")
            previous = existing.get(sample_id)
            if retry_all or not previous or previous.get("judge_status") != "success":
                to_process.append(sample)
        if sample_num is not None and sample_num > 0:
            to_process = to_process[:sample_num]

        if not to_process:
            print(f"{meta_reasoning_type}: no samples to judge.")
            return count_judge_status(existing)

        print(f"{meta_reasoning_type}: judging {len(to_process)} samples.")
        result_map = dict(existing)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {
                executor.submit(self.judge_single_sample, sample): sample
                for sample in to_process
            }
            with tqdm(total=len(to_process), desc=f"{meta_reasoning_type} RPCS", unit="sample") as progress:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    sample_id = sample.get("sample_id") or sample.get("id")
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            **build_result_base(sample, self.judge_model_name),
                            "rpcs": 0,
                            "criterion_scores": {criterion: 0 for criterion in RPCS_CRITERIA},
                            "rationale": {},
                            "usage": {},
                            "judge_status": "fail",
                            "error_msg": str(exc),
                        }

                    with self._lock:
                        result_map[result.get("sample_id", sample_id)] = result
                        completed += 1
                        if completed % max(batch_size, 1) == 0:
                            write_judgments(
                                output_file=output_file,
                                source_model_name=self.source_model_name,
                                judge_model_name=self.judge_model_name,
                                meta_reasoning_type=meta_reasoning_type,
                                include_image=self.include_image,
                                result_map=result_map,
                            )
                    progress.update(1)

        with self._lock:
            write_judgments(
                output_file=output_file,
                source_model_name=self.source_model_name,
                judge_model_name=self.judge_model_name,
                meta_reasoning_type=meta_reasoning_type,
                include_image=self.include_image,
                result_map=result_map,
            )
        return count_judge_status(result_map)


def build_result_base(sample: Dict[str, Any], judge_model_name: str) -> Dict[str, Any]:
    return {
        "sample_id": sample.get("sample_id") or sample.get("id"),
        "image": sample.get("image"),
        "question": sample.get("question"),
        "meta_reasoning_type": sample.get("meta_reasoning_type"),
        "reasoning_taxonomy": sample.get("reasoning_taxonomy"),
        "answer_type": sample.get("answer_type"),
        "metric": sample.get("metric"),
        "ground_truth": sample.get("ground_truth", sample.get("answer", "")),
        "predicted_answer": sample.get("predicted_answer", ""),
        "main_score": sample.get("score", 0.0),
        "model_output": sample.get("model_output", ""),
        "source_status": sample.get("status", ""),
        "judge_model_name": judge_model_name,
        "reference_reasoning_steps": sample.get("reference_reasoning_steps", []),
    }


def attach_reference_reasoning_steps(
    sample: Dict[str, Any],
    reference_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    sample_id = str(sample.get("sample_id") or sample.get("id") or "")
    reference = reference_map.get(sample_id, {})
    attached = dict(sample)
    attached["reference_reasoning_steps"] = reference.get("reasoning_steps", [])
    attached["reference_answer"] = reference.get("answer", sample.get("ground_truth", sample.get("answer", "")))
    return attached


def write_judgments(
    output_file: Path,
    source_model_name: str,
    judge_model_name: str,
    meta_reasoning_type: str,
    include_image: bool,
    result_map: Dict[str, Dict[str, Any]],
) -> None:
    ordered_results = sorted(result_map.values(), key=lambda item: item.get("sample_id", ""))
    payload = {
        "source_model_name": source_model_name,
        "judge_model_name": judge_model_name,
        "meta_reasoning_type": meta_reasoning_type,
        "result_count": len(ordered_results),
        "include_image": include_image,
        "rpcs_summary": build_rpcs_summary(ordered_results),
        "results": ordered_results,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_rpcs_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scorable = [
        item
        for item in results
        if item.get("judge_status") in {"success", "skipped"}
    ]
    successful = [item for item in results if item.get("judge_status") == "success"]
    rpcs_values = [float(item.get("rpcs", 0.0)) for item in scorable]
    normalized_values = [value / 4.0 for value in rpcs_values]
    return {
        "sample_count": len(results),
        "success_count": len(successful),
        "fail_count": sum(1 for item in results if item.get("judge_status") == "fail"),
        "skipped_count": sum(1 for item in results if item.get("judge_status") == "skipped"),
        "mean_rpcs": mean(rpcs_values),
        "mean_rpcs_normalized": mean(normalized_values),
        "criterion_pass_rate": build_criterion_pass_rate(scorable),
        "reasoning_taxonomy_mean_rpcs": grouped_mean_rpcs(scorable, "reasoning_taxonomy"),
    }


def build_criterion_pass_rate(results: List[Dict[str, Any]]) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for criterion in RPCS_CRITERIA:
        rates[criterion] = mean(
            float((item.get("criterion_scores") or {}).get(criterion, 0))
            for item in results
        )
    return rates


def grouped_mean_rpcs(results: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    groups: Dict[str, List[float]] = {}
    for item in results:
        groups.setdefault(str(item.get(key, "unknown")), []).append(float(item.get("rpcs", 0.0)))
    return {name: mean(values) for name, values in sorted(groups.items())}


def count_judge_status(results: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    counts = {"success": 0, "fail": 0, "skipped": 0}
    for item in results.values():
        status = item.get("judge_status", "fail")
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_overall_summary(output_dir: Path, source_model_name: str, judge_model_name: str) -> Dict[str, Any]:
    all_results: List[Dict[str, Any]] = []
    per_type: Dict[str, Dict[str, Any]] = {}
    for meta_reasoning_type in config.META_REASONING_TYPES:
        output_file = output_dir / f"{meta_reasoning_type}_rpcs_judge_result.json"
        if not output_file.exists():
            continue
        with output_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        results = payload.get("results", [])
        all_results.extend(results)
        per_type[meta_reasoning_type] = payload.get("rpcs_summary", build_rpcs_summary(results))

    scorable = [
        item
        for item in all_results
        if item.get("judge_status") in {"success", "skipped"}
    ]
    successful = [item for item in all_results if item.get("judge_status") == "success"]
    ability_mean = grouped_mean_rpcs(scorable, "meta_reasoning_type")
    return {
        "source_model_name": source_model_name,
        "judge_model_name": judge_model_name,
        "sample_count": len(all_results),
        "success_count": len(successful),
        "fail_count": sum(1 for item in all_results if item.get("judge_status") == "fail"),
        "skipped_count": sum(1 for item in all_results if item.get("judge_status") == "skipped"),
        "mean_rpcs": mean(float(item.get("rpcs", 0.0)) for item in scorable),
        "mean_rpcs_normalized": mean(float(item.get("rpcs", 0.0)) / 4.0 for item in scorable),
        "meta_reasoning_type_mean_rpcs": ability_mean,
        "meta_reasoning_type_mean_rpcs_normalized": {
            key: value / 4.0 for key, value in ability_mean.items()
        },
        "criterion_pass_rate": build_criterion_pass_rate(scorable),
        "reasoning_taxonomy_mean_rpcs": grouped_mean_rpcs(scorable, "reasoning_taxonomy"),
        "per_meta_reasoning_type": per_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge Reasoning Process Compliance Score (RPCS) for OCR-MetaReasoning outputs."
    )
    parser.add_argument(
        "--source_model_name",
        default="gpt-5.4-mini",
        help="Model whose inference files are read from OCR_MetaReasoning/evaluation/result/<model>_result.",
    )
    parser.add_argument("--judge_model_name", default=config.JUDGE_MODEL_NAME)
    parser.add_argument("--api_key", default=config.API_KEY)
    parser.add_argument("--base_url", default=config.BASE_URL)
    parser.add_argument(
        "--meta_reasoning_types",
        nargs="+",
        default=["all"],
        help="Choose from meta_deductive meta_inductive meta_abductive or all.",
    )
    parser.add_argument("--sample_num", type=int, default=0, help="Number of samples per selected ability. 0 means all remaining samples.")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=config.DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--repetition_penalty", type=float, default=config.DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--presence_penalty", type=float, default=config.DEFAULT_PRESENCE_PENALTY)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_retries", type=int, default=config.DEFAULT_MAX_RETRIES)
    parser.add_argument("--timeout", type=float, default=config.DEFAULT_TIMEOUT)
    parser.add_argument("--retry_all", action="store_true")
    parser.add_argument(
        "--no_image",
        action="store_true",
        help="Judge from text fields only. Default is to include the original image.",
    )
    args = parser.parse_args()

    judge_model_config = config.infer_config(
        model_name=args.judge_model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )
    judge = RPCSJudge(
        source_model_name=args.source_model_name,
        judge_model_config=judge_model_config,
        include_image=not args.no_image,
    )

    selected_types = parse_meta_reasoning_types(args.meta_reasoning_types)
    sample_num = args.sample_num if args.sample_num > 0 else None
    status_summary: Dict[str, Dict[str, int]] = {}
    for meta_reasoning_type in selected_types:
        status_summary[meta_reasoning_type] = judge.process_meta_reasoning_type(
            meta_reasoning_type=meta_reasoning_type,
            max_workers=args.workers,
            batch_size=args.batch_size,
            sample_num=sample_num,
            retry_all=args.retry_all,
        )

    overall_summary = build_overall_summary(
        output_dir=judge.output_dir,
        source_model_name=args.source_model_name,
        judge_model_name=args.judge_model_name,
    )
    overall_file = judge.output_dir / "rpcs_summary.json"
    with overall_file.open("w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({"status": status_summary, "summary_file": str(overall_file)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
