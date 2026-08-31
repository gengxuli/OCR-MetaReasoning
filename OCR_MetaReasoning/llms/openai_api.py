import time
from typing import Any, Dict, List, Optional

import openai


class OpenAIModel:
    """
    OpenAI-compatible chat completion client.

    The default config in OCR_MetaReasoning.bench_construction.config points to
    https://openrouter.ai/, which follows the OpenAI-style /chat/completions API.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: Optional[str] = None,
        max_retries: int = 2,
        delay_seconds: float = 1.0,
        timeout: float = 3.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model_name
        self.max_retries = max_retries
        self.delay_seconds = delay_seconds
        self.timeout = timeout

    def call_with_messages(
        self,
        model_name: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> Optional[str]:
        if model_name is None:
            model_name = self.model_name
        if not model_name:
            raise ValueError("model_name is required")
        if self.client is None:
            raise ValueError(
                "API key is empty. Set OPENROUTER_API_KEY, SYNTHESIZE_API_KEY, or OPENAI_API_KEY."
            )

        if messages is None:
            messages = []

        format_args = {}
        if json_schema:
            format_args["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    timeout=self.timeout,
                    **format_args,
                )
                return completion.choices[0].message.content
            except openai.APIError as e:
                print(f"OpenAI API Error (Attempt {attempt + 1}): {e}")
            except Exception as e:
                print(f"Unexpected LLM call error (Attempt {attempt + 1}): {e}")

            if attempt < self.max_retries - 1:
                time.sleep(self.delay_seconds)

        return None

    def call_with_prompt_and_image_urls(
        self,
        model_name: Optional[str] = None,
        prompt: str = "",
        image_urls: Optional[List[str]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> Optional[str]:
        if image_urls:
            user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
            for image_url in image_urls:
                user_content.append({"type": "image_url", "image_url": {"url": image_url}})
        else:
            user_content = prompt

        messages = [{"role": "user", "content": user_content}]
        return self.call_with_messages(
            model_name=model_name,
            messages=messages,
            json_schema=json_schema,
            temperature=temperature,
        )
