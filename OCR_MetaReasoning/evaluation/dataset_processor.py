import base64
import io
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps
from tqdm import tqdm

from OCR_MetaReasoning.evaluation.answer_utils import mean
from OCR_MetaReasoning.evaluation.config import config
from OCR_MetaReasoning.utils.image_transfer import pillow_to_base64_data_url


class DatasetProcessor:
    def __init__(self, model_name: str, status_field: str = "status") -> None:
        self.model_name = model_name
        self.status_field = status_field
        self.results_dir = config.RESULT_ROOT / f"{model_name}_result"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def process_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def load_samples(self, meta_reasoning_type: str) -> List[Dict[str, Any]]:
        dataset_file = config.dataset_file(meta_reasoning_type)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        samples = []
        with dataset_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def result_file(self, meta_reasoning_type: str) -> Path:
        return self.results_dir / f"{meta_reasoning_type}_infer_result.json"

    def load_existing_results(self, result_file: Path) -> Dict[str, Dict[str, Any]]:
        if not result_file.exists():
            return {}
        try:
            with result_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}

        if isinstance(data, dict) and isinstance(data.get("results"), list):
            return {item.get("sample_id", item.get("id")): item for item in data["results"]}
        if isinstance(data, list):
            return {item.get("sample_id", item.get("id")): item for item in data}
        return {}

    def filter_samples(
        self,
        samples: List[Dict[str, Any]],
        existing_results: Dict[str, Dict[str, Any]],
        sample_num: Optional[int],
        retry_all: bool,
    ) -> List[Dict[str, Any]]:
        if retry_all:
            selected = samples
        else:
            selected = []
            for sample in samples:
                sample_id = sample.get("sample_id") or sample.get("id")
                previous = existing_results.get(sample_id)
                if not previous or previous.get(self.status_field) != "success":
                    selected.append(sample)

        if sample_num is not None and sample_num > 0:
            selected = selected[:sample_num]
        return selected

    def write_results(
        self,
        result_file: Path,
        meta_reasoning_type: str,
        result_map: Dict[str, Dict[str, Any]],
    ) -> None:
        ordered_results = sorted(result_map.values(), key=lambda item: item.get("sample_id", ""))
        payload = {
            "model_name": self.model_name,
            "meta_reasoning_type": meta_reasoning_type,
            "result_count": len(ordered_results),
            "score_summary": build_score_summary(ordered_results),
            "results": ordered_results,
        }
        with result_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def process_meta_reasoning_type(
        self,
        meta_reasoning_type: str,
        max_workers: int = 4,
        batch_size: int = 10,
        sample_num: Optional[int] = None,
        retry_all: bool = False,
    ) -> Dict[str, int]:
        samples = self.load_samples(meta_reasoning_type)
        result_file = self.result_file(meta_reasoning_type)
        existing_results = {} if retry_all else self.load_existing_results(result_file)
        to_process = self.filter_samples(samples, existing_results, sample_num, retry_all)

        if not to_process:
            print(f"{meta_reasoning_type}: no samples to process.")
            return self.count_status(existing_results)

        print(f"{meta_reasoning_type}: processing {len(to_process)} samples.")
        result_map = dict(existing_results)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {executor.submit(self.process_single_sample, sample): sample for sample in to_process}
            with tqdm(total=len(to_process), desc=meta_reasoning_type, unit="sample") as progress:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    sample_id = sample.get("sample_id") or sample.get("id")
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "sample_id": sample_id,
                            "meta_reasoning_type": meta_reasoning_type,
                            "status": "fail",
                            "error_msg": str(exc),
                        }

                    with self._lock:
                        result_map[result.get("sample_id", sample_id)] = result
                        completed += 1
                        if completed % max(batch_size, 1) == 0:
                            self.write_results(result_file, meta_reasoning_type, result_map)
                    progress.update(1)

        with self._lock:
            self.write_results(result_file, meta_reasoning_type, result_map)
        return self.count_status(result_map)

    def count_status(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        counts = {"success": 0, "fail": 0}
        for item in results.values():
            status = item.get(self.status_field, "fail")
            counts[status] = counts.get(status, 0) + 1
        return counts


def build_multimodal_content(
    question: str,
    image_path: Path,
    answer_type: str,
    max_image_pixels: int = 0,
    max_image_bytes: int = 0,
    image_jpeg_quality: int = 90,
) -> List[Dict[str, Any]]:
    prompt = build_prompt(question, answer_type)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(
                    image_path,
                    max_image_pixels=max_image_pixels,
                    max_image_bytes=max_image_bytes,
                    image_jpeg_quality=image_jpeg_quality,
                )
            },
        }
    )
    return content


def build_prompt(question: str, answer_type: str) -> str:
    format_hint = build_answer_format_hint(answer_type)
    return (
        "You are given one text-rich image and one question.\n"
        "Solve the problem step by step based on the image.\n"
        "First output the solution steps in numbered form, then output the final answer on its own line.\n"
        "Use this format strictly:\n"
        "Step 1: ...\n"
        "Step 2: ...\n"
        f"{format_hint}\n"
        "Do not add explanation, units, citations, or markdown after the final answer line unless the requested answer itself requires them.\n"
        f"Question: {question}"
    )


def build_answer_format_hint(answer_type: str) -> str:
    if answer_type == "json":
        return "Final Answer: <valid minified JSON only>"
    if answer_type in {"integer", "float"}:
        return "Final Answer: <number only>"
    return "Final Answer: <short answer only>"


def image_to_data_url(
    image_path: Path,
    max_image_pixels: int = 0,
    max_image_bytes: int = 0,
    image_jpeg_quality: int = 90,
) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as image:
        if max_image_pixels <= 0 and max_image_bytes <= 0:
            image_format = "JPEG" if image.format == "JPEG" else "PNG"
            return pillow_to_base64_data_url(
                image.convert("RGB") if image_format == "JPEG" else image,
                image_format=image_format,
            )

        image = ImageOps.exif_transpose(image)
        image_format = "JPEG" if image.format == "JPEG" else "PNG"
        prepared = resize_to_max_pixels(image, max_image_pixels)
        image_bytes, image_format = encode_limited_image(
            prepared,
            image_format=image_format,
            max_image_bytes=max_image_bytes,
            image_jpeg_quality=image_jpeg_quality,
        )
        return image_bytes_to_data_url(image_bytes, image_format)


def resize_to_max_pixels(image: Image.Image, max_image_pixels: int) -> Image.Image:
    if max_image_pixels <= 0 or image.width * image.height <= max_image_pixels:
        return image.copy()

    scale = math.sqrt(max_image_pixels / (image.width * image.height))
    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def encode_limited_image(
    image: Image.Image,
    image_format: str,
    max_image_bytes: int,
    image_jpeg_quality: int,
) -> Tuple[bytes, str]:
    image_jpeg_quality = min(95, max(1, image_jpeg_quality))
    image_bytes = encode_image_bytes(image, image_format, image_jpeg_quality)
    if max_image_bytes <= 0 or len(image_bytes) <= max_image_bytes:
        return image_bytes, image_format

    working = image
    jpeg_quality_floor = 60
    for quality in range(image_jpeg_quality, jpeg_quality_floor - 1, -5):
        image_bytes = encode_image_bytes(working, "JPEG", quality)
        if len(image_bytes) <= max_image_bytes:
            return image_bytes, "JPEG"

    quality = jpeg_quality_floor
    while len(image_bytes) > max_image_bytes and working.width > 1 and working.height > 1:
        scale = math.sqrt(max_image_bytes / len(image_bytes)) * 0.95
        width = max(1, int(working.width * scale))
        height = max(1, int(working.height * scale))
        if width == working.width and height == working.height:
            width = max(1, working.width - 1)
            height = max(1, working.height - 1)
        working = working.resize((width, height), Image.Resampling.LANCZOS)
        image_bytes = encode_image_bytes(working, "JPEG", quality)

    return image_bytes, "JPEG"


def encode_image_bytes(image: Image.Image, image_format: str, image_jpeg_quality: int) -> bytes:
    buffer = io.BytesIO()
    if image_format == "JPEG":
        output = image.convert("RGB") if image.mode not in {"RGB", "L"} else image
        output.save(buffer, format="JPEG", quality=image_jpeg_quality, optimize=True)
    else:
        image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def image_bytes_to_data_url(image_bytes: bytes, image_format: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    mime = "image/jpeg" if image_format == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


def build_score_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [item for item in results if item.get("status") == "success"]
    all_scores = [float(item.get("score", 0.0)) for item in successful]
    ability_scores = grouped_score(successful, "meta_reasoning_type")
    missing_meta_reasoning_types = [
        meta_reasoning_type
        for meta_reasoning_type in config.META_REASONING_TYPES
        if meta_reasoning_type not in ability_scores
    ]

    summary: Dict[str, Any] = {
        "sample_count": len(results),
        "success_count": len(successful),
        "fail_count": len(results) - len(successful),
        "MRMS": None
        if missing_meta_reasoning_types
        else mean(
            ability_scores[meta_reasoning_type]
            for meta_reasoning_type in config.META_REASONING_TYPES
        ),
        "ability_macro_score": mean(ability_scores.values()),
        "missing_meta_reasoning_types": missing_meta_reasoning_types,
        "overall_micro_score": mean(all_scores),
        "meta_reasoning_type_score": ability_scores,
        "answer_type_score": grouped_answer_form_score(successful),
        "answer_type_detail_score": grouped_score(successful, "answer_type"),
        "reasoning_taxonomy_score": grouped_score(successful, "reasoning_taxonomy"),
        "ability_taxonomy_subscore": build_ability_taxonomy_subscore(successful),
    }

    return summary


def grouped_score(results: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    groups: Dict[str, List[float]] = {}
    for item in results:
        score = float(item.get("score", 0.0))
        groups.setdefault(str(item.get(key, "unknown")), []).append(score)
    return {name: mean(scores) for name, scores in sorted(groups.items())}


def grouped_answer_form_score(results: List[Dict[str, Any]]) -> Dict[str, float]:
    groups: Dict[str, List[float]] = {name: [] for name in config.ANSWER_FORMS}
    for item in results:
        groups.setdefault(answer_form(item.get("answer_type", "unknown")), []).append(
            float(item.get("score", 0.0))
        )
    return {name: mean(scores) for name, scores in sorted(groups.items()) if scores}


def answer_form(answer_type: str) -> str:
    if answer_type in {"integer", "float"}:
        return "numeric"
    if answer_type == "json":
        return "json"
    if answer_type == "string":
        return "string"
    return "unknown"


def build_ability_taxonomy_subscore(results: List[Dict[str, Any]]) -> Dict[str, float]:
    matrix: Dict[str, List[float]] = {
        f"{ability}::{taxonomy}": []
        for ability in config.META_REASONING_TYPES
        for taxonomy in config.REASONING_TAXONOMIES
    }
    for item in results:
        key = f"{item.get('meta_reasoning_type', 'unknown')}::{item.get('reasoning_taxonomy', 'unknown')}"
        matrix.setdefault(key, []).append(float(item.get("score", 0.0)))
    return {key: mean(scores) for key, scores in sorted(matrix.items()) if scores}
