import os
from pathlib import Path
from typing import Dict, List


class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATASET_ROOT = PROJECT_ROOT / "dataset"
    RESULT_ROOT = PROJECT_ROOT / "OCR_MetaReasoning" / "evaluation" / "result"
    RPCS_RESULT_ROOT = PROJECT_ROOT / "OCR_MetaReasoning" / "evaluation" / "rpcs_result"

    META_REASONING_TYPES: List[str] = [
        "meta_deductive",
        "meta_inductive",
        "meta_abductive",
    ]

    REASONING_TAXONOMIES: List[str] = [
        "transaction_analysis_reasoning",
        "data_interpretation_reasoning",
        "field_dependency_reasoning",
        "document_logic_reasoning",
        "layout_semantics_reasoning",
    ]

    ANSWER_FORMS: List[str] = [
        "string",
        "numeric",
        "json",
    ]

    # Keep environment names provider-neutral for public release.
    API_KEY = (
        os.getenv("MODEL_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    )
    BASE_URL = os.getenv("MODEL_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://openrouter.ai/"
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
    JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", MODEL_NAME)

    DEFAULT_TIMEOUT = float(os.getenv("MODEL_TIMEOUT", "600"))
    DEFAULT_MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "3"))
    DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0"))
    DEFAULT_TOP_P = float(os.getenv("MODEL_TOP_P", "1"))
    DEFAULT_TOP_K = int(os.getenv("MODEL_TOP_K", "0"))
    DEFAULT_REPETITION_PENALTY = float(os.getenv("MODEL_REPETITION_PENALTY", "1"))
    DEFAULT_PRESENCE_PENALTY = float(os.getenv("MODEL_PRESENCE_PENALTY", "0"))
    DEFAULT_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "32768"))

    @classmethod
    def dataset_file(cls, meta_reasoning_type: str) -> Path:
        return cls.DATASET_ROOT / meta_reasoning_type / f"benchmark_{meta_reasoning_type}.jsonl"

    @classmethod
    def image_path(cls, meta_reasoning_type: str, image: str) -> Path:
        return cls.DATASET_ROOT / meta_reasoning_type / image

    @classmethod
    def infer_config(
        cls,
        model_name: str,
        api_key: str = "",
        base_url: str = "",
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        repetition_penalty: float = None,
        presence_penalty: float = None,
        max_tokens: int = None,
        max_retries: int = None,
        timeout: float = None,
    ) -> Dict:
        return {
            "model_name": model_name or cls.MODEL_NAME,
            "api_key": api_key or cls.API_KEY,
            "base_url": base_url or cls.BASE_URL,
            "temperature": cls.DEFAULT_TEMPERATURE if temperature is None else temperature,
            "top_p": cls.DEFAULT_TOP_P if top_p is None else top_p,
            "top_k": cls.DEFAULT_TOP_K if top_k is None else top_k,
            "repetition_penalty": (
                cls.DEFAULT_REPETITION_PENALTY if repetition_penalty is None else repetition_penalty
            ),
            "presence_penalty": cls.DEFAULT_PRESENCE_PENALTY if presence_penalty is None else presence_penalty,
            "max_tokens": cls.DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens,
            "max_retries": cls.DEFAULT_MAX_RETRIES if max_retries is None else max_retries,
            "timeout": cls.DEFAULT_TIMEOUT if timeout is None else timeout,
        }

    @classmethod
    def rpcs_result_dir(cls, source_model_name: str) -> Path:
        return cls.RPCS_RESULT_ROOT / f"{source_model_name}_result"


config = Config()
