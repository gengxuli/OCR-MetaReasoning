import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from OCR_MetaReasoning.evaluation.answer_utils import extract_final_answer, score_prediction
from OCR_MetaReasoning.evaluation.api_client import (
    ChatCompletionClient,
    extract_message_text,
    extract_reasoning_content,
)
from OCR_MetaReasoning.evaluation.config import config
from OCR_MetaReasoning.evaluation.dataset_processor import DatasetProcessor, build_multimodal_content


class OpenAIMultimodalInfer(DatasetProcessor):
    def __init__(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        thinking_enabled: bool = False,
        reasoning_effort: str = "",
        max_image_pixels: int = 0,
        max_image_bytes: int = 0,
        image_jpeg_quality: int = 90,
    ) -> None:
        super().__init__(model_name=model_name)
        self.temperature = model_config.get("temperature", 0.0)
        self.top_p = model_config.get("top_p")
        self.top_k = model_config.get("top_k")
        self.repetition_penalty = model_config.get("repetition_penalty")
        self.presence_penalty = model_config.get("presence_penalty")
        self.max_tokens = model_config.get("max_tokens")
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = reasoning_effort or None
        self.max_image_pixels = max_image_pixels
        self.max_image_bytes = max_image_bytes
        self.image_jpeg_quality = image_jpeg_quality
        self.client = ChatCompletionClient(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"],
            model_name=model_config["model_name"],
            max_retries=model_config.get("max_retries", 3),
            timeout=model_config.get("timeout", 5.0),
        )

    def process_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = sample.get("sample_id") or sample.get("id")
        meta_reasoning_type = sample["meta_reasoning_type"]
        answer_type = sample.get("answer_type", "string")
        metric = sample.get("metric", "exact_match")
        image_path = config.image_path(meta_reasoning_type, sample["image"])
        result_base = build_result_base(sample, sample_id, meta_reasoning_type, answer_type, metric)

        try:
            content = build_multimodal_content(
                sample["question"],
                image_path,
                answer_type,
                max_image_pixels=self.max_image_pixels,
                max_image_bytes=self.max_image_bytes,
                image_jpeg_quality=self.image_jpeg_quality,
            )
            messages = [{"role": "user", "content": content}]
            response = self.client.create(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                presence_penalty=self.presence_penalty,
                thinking_enabled=self.thinking_enabled,
                reasoning_effort=self.reasoning_effort,
            )
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            model_output = extract_message_text(message).strip()
            reasoning_content = extract_reasoning_content(response)
            usage = response.get("usage", {})
            finish_reason = choice.get("finish_reason")
            if not model_output:
                return {
                    **result_base,
                    "model_output": "",
                    "predicted_answer": "",
                    "reasoning_content": reasoning_content,
                    "score": 0.0,
                    "usage": usage,
                    "finish_reason": finish_reason,
                    "status": "fail",
                    "error_msg": "empty model output",
                    "raw_message": message,
                }

            predicted_answer = extract_final_answer(model_output, answer_type)
            score = score_prediction(predicted_answer, sample.get("answer", ""), metric, answer_type)

            return {
                **result_base,
                "model_output": model_output,
                "predicted_answer": predicted_answer,
                "reasoning_content": reasoning_content,
                "score": score,
                "usage": usage,
                "finish_reason": finish_reason,
                "status": "success",
            }
        except Exception as exc:
            return {
                **result_base,
                "model_output": "",
                "predicted_answer": "",
                "reasoning_content": "",
                "score": 0.0,
                "usage": {},
                "status": "fail",
                "error_msg": str(exc),
            }


def build_result_base(
    sample: Dict[str, Any],
    sample_id: Any,
    meta_reasoning_type: str,
    answer_type: str,
    metric: str,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "image": sample.get("image"),
        "question": sample.get("question"),
        "meta_reasoning_type": meta_reasoning_type,
        "reasoning_taxonomy": sample.get("reasoning_taxonomy"),
        "answer_type": answer_type,
        "metric": metric,
        "ground_truth": sample.get("answer", ""),
    }


def parse_meta_reasoning_types(values: List[str]) -> List[str]:
    if not values or "all" in values:
        return list(config.META_REASONING_TYPES)
    invalid = [value for value in values if value not in config.META_REASONING_TYPES]
    if invalid:
        raise ValueError(f"Invalid meta reasoning type(s): {invalid}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR-MetaReasoning inference.")
    parser.add_argument("--model_name", type=str, default=config.MODEL_NAME)
    parser.add_argument("--api_key", type=str, default=config.API_KEY)
    parser.add_argument("--base_url", type=str, default=config.BASE_URL)
    parser.add_argument(
        "--meta_reasoning_types",
        nargs="+",
        default=["all"],
        help="Choose from meta_deductive meta_inductive meta_abductive or all.",
    )
    parser.add_argument("--sample_num", type=int, default=0, help="Number of samples per selected ability. 0 means all remaining samples.")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=config.DEFAULT_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=config.DEFAULT_TOP_P)
    parser.add_argument("--top_k", type=int, default=config.DEFAULT_TOP_K)
    parser.add_argument("--repetition_penalty", type=float, default=config.DEFAULT_REPETITION_PENALTY)
    parser.add_argument("--presence_penalty", type=float, default=config.DEFAULT_PRESENCE_PENALTY)
    parser.add_argument("--max_tokens", type=int, default=config.DEFAULT_MAX_TOKENS)
    parser.add_argument("--max_retries", type=int, default=config.DEFAULT_MAX_RETRIES)
    parser.add_argument("--timeout", type=float, default=config.DEFAULT_TIMEOUT)
    parser.add_argument("--thinking_enabled", action="store_true")
    parser.add_argument("--reasoning_effort", type=str, default="")
    parser.add_argument(
        "--max_image_pixels",
        type=int,
        default=0,
        help="Resize input images whose pixel count exceeds this value. 0 keeps original pixels.",
    )
    parser.add_argument(
        "--max_image_bytes",
        type=int,
        default=0,
        help="Compress/resize encoded input images whose byte size exceeds this value. 0 keeps original size.",
    )
    parser.add_argument(
        "--image_jpeg_quality",
        type=int,
        default=90,
        help="Initial JPEG quality used when image byte limiting is enabled.",
    )
    parser.add_argument("--retry_all", action="store_true")
    args = parser.parse_args()

    model_config = config.infer_config(
        model_name=args.model_name,
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
    infer_engine = OpenAIMultimodalInfer(
        model_name=args.model_name,
        model_config=model_config,
        thinking_enabled=args.thinking_enabled,
        reasoning_effort=args.reasoning_effort,
        max_image_pixels=args.max_image_pixels,
        max_image_bytes=args.max_image_bytes,
        image_jpeg_quality=args.image_jpeg_quality,
    )

    selected_types = parse_meta_reasoning_types(args.meta_reasoning_types)
    sample_num = args.sample_num if args.sample_num > 0 else None
    summary: Dict[str, Dict[str, int]] = {}
    for meta_reasoning_type in selected_types:
        status_count = infer_engine.process_meta_reasoning_type(
            meta_reasoning_type=meta_reasoning_type,
            max_workers=args.workers,
            batch_size=args.batch_size,
            sample_num=sample_num,
            retry_all=args.retry_all,
        )
        summary[meta_reasoning_type] = status_count

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
