import json
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from OCR_MetaReasoning.bench_construction.meta_taxonomy import (
    validate_meta_reasoning_type,
    validate_reasoning_taxonomy,
)


MetaReasoningType = Literal["meta_deductive", "meta_inductive", "meta_abductive"]
ReasoningTaxonomy = Literal[
    "transaction_analysis_reasoning",
    "data_interpretation_reasoning",
    "field_dependency_reasoning",
    "document_logic_reasoning",
    "layout_semantics_reasoning",
]
AnswerType = Literal["string", "integer", "float", "json"]
MetricType = Literal["exact_match", "anls", "numeric", "json_f1"]


class MetaReasoningSample(BaseModel):
    """
    OCR-MetaReasoning sample schema defined by OCR_MetaReasoning_Design.md.
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(description="Unique sample id")
    image: str = Field(description="Relative image path")
    question: str = Field(description="Question text")
    meta_reasoning_type: MetaReasoningType = Field(description="Meta reasoning type")
    reasoning_taxonomy: ReasoningTaxonomy = Field(description="Fine-grained reasoning taxonomy")
    answer_type: AnswerType = Field(description="Answer value type")
    answer: str = Field(description="Normalized answer string")
    reasoning_steps: List[str] = Field(description="Minimal sufficient reasoning chain")
    metric: MetricType = Field(description="Automatic evaluation metric")

    @field_validator("sample_id", "image", "question")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("field must be non-empty")
        return value

    @field_validator("meta_reasoning_type")
    @classmethod
    def validate_meta_type_membership(cls, value: str) -> str:
        return validate_meta_reasoning_type(value)

    @field_validator("reasoning_taxonomy")
    @classmethod
    def validate_taxonomy_membership(cls, value: str) -> str:
        return validate_reasoning_taxonomy(value)

    @field_validator("answer", mode="before")
    @classmethod
    def normalize_answer(cls, value) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return str(value).strip()

    @field_validator("reasoning_steps")
    @classmethod
    def validate_reasoning_steps(cls, value: List[str]) -> List[str]:
        cleaned_steps = [str(step).strip() for step in value if str(step).strip()]
        if not cleaned_steps:
            raise ValueError("reasoning_steps must contain at least one non-empty step")
        for idx, step in enumerate(cleaned_steps, start=1):
            if not re.match(rf"^Step{idx}:\s+\S", step):
                raise ValueError(f"reasoning_steps[{idx - 1}] must start with 'Step{idx}: '")
        return cleaned_steps

    @model_validator(mode="after")
    def validate_answer_and_metric(self):
        if self.answer_type == "json":
            json.loads(self.answer)
            if self.metric != "json_f1":
                raise ValueError("answer_type=json requires metric=json_f1")
        elif self.answer_type in {"integer", "float"}:
            float(self.answer)
            if self.answer_type == "integer" and not float(self.answer).is_integer():
                raise ValueError("integer answer_type requires an integer-like answer")
            if self.metric != "numeric":
                raise ValueError("numeric answer_type requires metric=numeric")
        elif self.answer_type == "string" and self.metric not in {"exact_match", "anls"}:
            raise ValueError("string answer_type requires exact_match or anls")
        return self


class MetaSynthesisOutput(BaseModel):
    """
    LLM synthesis output wrapper. The published benchmark only uses `sample`.
    """

    model_config = ConfigDict(extra="forbid")

    sample: Optional[MetaReasoningSample] = Field(default=None)
    evidence_points: List[str] = Field(default_factory=list)
    construction_notes: Optional[str] = Field(default=None)
    direct_ocr_risk: Optional[str] = Field(default=None)
    give_up: bool = Field(default=False)
    give_up_reason: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_success_payload(self):
        if not self.give_up and self.sample is None:
            raise ValueError("sample is required when give_up is false")
        return self


class LLMSynCheckOutput(BaseModel):
    """
    LLM quality-check output.
    """

    model_config = ConfigDict(extra="forbid")

    judge: Optional[bool] = Field(
        description="True means the synthesized sample is qualified; False means unqualified",
        default=None,
    )
    judge_reason: Optional[str] = Field(description="Short explanation for the decision", default=None)
