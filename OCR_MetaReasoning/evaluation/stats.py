import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from OCR_MetaReasoning.evaluation.answer_utils import extract_final_answer, mean, score_prediction
from OCR_MetaReasoning.evaluation.config import config
from OCR_MetaReasoning.evaluation.dataset_processor import (
    build_ability_taxonomy_subscore,
    grouped_answer_form_score,
    grouped_score,
)


def load_results(model_name: str) -> List[Dict[str, Any]]:
    result_dir = config.RESULT_ROOT / f"{model_name}_result"
    all_results: List[Dict[str, Any]] = []
    for meta_reasoning_type in config.META_REASONING_TYPES:
        result_file = result_dir / f"{meta_reasoning_type}_infer_result.json"
        if not result_file.exists():
            continue
        with result_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        all_results.extend(payload.get("results", []))
    return all_results


def maybe_rescore_results(
    results: List[Dict[str, Any]],
    rescore: bool,
    reextract: bool = False,
) -> List[Dict[str, Any]]:
    if not rescore and not reextract:
        return results

    rescored: List[Dict[str, Any]] = []
    for item in results:
        updated = dict(item)
        if updated.get("status") == "success":
            if reextract:
                updated["predicted_answer"] = extract_final_answer(
                    updated.get("model_output", ""),
                    updated.get("answer_type", "string"),
                )
            updated["score"] = score_prediction(
                updated.get("predicted_answer", ""),
                updated.get("ground_truth", ""),
                updated.get("metric", "exact_match"),
                updated.get("answer_type", "string"),
            )
        rescored.append(updated)
    return rescored


def build_summary(
    results: List[Dict[str, Any]],
    rescore: bool = False,
    reextract: bool = False,
) -> Dict[str, Any]:
    results = maybe_rescore_results(results, rescore, reextract)
    successful = [item for item in results if item.get("status") == "success"]
    ability_scores = grouped_score(successful, "meta_reasoning_type")
    ability_values = [
        ability_scores.get(meta_reasoning_type, 0.0)
        for meta_reasoning_type in config.META_REASONING_TYPES
    ]

    return {
        "sample_count": len(results),
        "success_count": len(successful),
        "fail_count": len(results) - len(successful),
        "MRMS": mean(ability_values),
        "overall_micro_score": mean(float(item.get("score", 0.0)) for item in successful),
        "meta_reasoning_type_score": ability_scores,
        "answer_type_score": grouped_answer_form_score(successful),
        "answer_type_detail_score": grouped_score(successful, "answer_type"),
        "reasoning_taxonomy_score": grouped_score(successful, "reasoning_taxonomy"),
        "ability_taxonomy_subscore": build_ability_taxonomy_subscore(successful),
        "rescored": rescore or reextract,
        "reextracted": reextract,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OCR-MetaReasoning inference scores.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_file", default="")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Recompute scores from predicted_answer and ground_truth with the current strict scorers.",
    )
    parser.add_argument(
        "--reextract",
        action="store_true",
        help="Re-extract predicted_answer from model_output before recomputing scores.",
    )
    args = parser.parse_args()

    results = load_results(args.model_name)
    summary = build_summary(results, rescore=args.rescore, reextract=args.reextract)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = config.RESULT_ROOT / f"{args.model_name}_result" / "score_summary.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
