import time
from typing import Any, Dict, List, Optional

import openai


class ChatCompletionClient:
    """OpenAI-compatible chat completion client for third-party gateways."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_retries: int = 3,
        timeout: float = 600.0,
        delay_seconds: float = 1.0,
    ) -> None:
        if not api_key:
            raise ValueError(
                "API key is required. Set MODEL_API_KEY or OPENAI_API_KEY."
            )
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.delay_seconds = delay_seconds
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def create(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        thinking_enabled: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        request_args: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "timeout": self.timeout,
        }
        if top_p is not None:
            request_args["top_p"] = top_p
        if max_tokens is not None and max_tokens > 0:
            request_args["max_tokens"] = max_tokens
        if presence_penalty is not None:
            request_args["presence_penalty"] = presence_penalty

        extra_body: Dict[str, Any] = {}
        if top_k is not None and top_k > 0:
            extra_body["top_k"] = top_k
        if repetition_penalty is not None and repetition_penalty != 1:
            extra_body["repetition_penalty"] = repetition_penalty
        if thinking_enabled:
            # Some OpenAI-compatible gateways expose model-native thinking through
            # provider-specific passthrough fields. Unsupported models usually ignore it.
            extra_body["google"] = {"thinking_config": {"include_thoughts": True}}
            if reasoning_effort:
                request_args["reasoning_effort"] = reasoning_effort
        if extra_body:
            request_args["extra_body"] = extra_body

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(**request_args)
                if hasattr(completion, "model_dump"):
                    return completion.model_dump()
                return dict(completion)
            except Exception as exc:
                last_error = exc
                print(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {exc}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay_seconds)

        raise RuntimeError(f"API call failed after {self.max_retries} attempts: {last_error}")


def extract_message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                chunks.append(str(item.get("text", "")))
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunk for chunk in chunks if chunk)
    return "" if content is None else str(content)


def extract_reasoning_content(response: Dict[str, Any]) -> str:
    if not response:
        return ""

    choices = response.get("choices") or []
    if not choices:
        return ""

    choice = choices[0]
    message = choice.get("message") or {}

    for container in (message, choice):
        for key in ("reasoning_content", "reasoning", "thinking"):
            value = container.get(key)
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, dict):
                content = value.get("content") or value.get("text")
                if content:
                    return str(content).strip()
            if isinstance(value, list):
                parts = []
                for item in value:
                    if isinstance(item, dict):
                        parts.append(str(item.get("content") or item.get("text") or ""))
                    else:
                        parts.append(str(item))
                return "\n".join(part for part in parts if part).strip()

    answer = extract_message_text(message)
    if "<think>" in answer and "</think>" in answer:
        return answer.split("</think>", 1)[0].replace("<think>", "").strip()
    return ""
