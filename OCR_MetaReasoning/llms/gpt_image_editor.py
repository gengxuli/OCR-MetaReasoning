import base64
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests


class GPTImageEditor:
    """
    Minimal OpenAI-compatible image edit client for Openrouter's /v1/images/edits.

    Openrouter's gpt-image-2 edits endpoint accepts multipart/form-data with
    model, prompt, image, size, and quality parameters and returns base64 image
    data under data[0].b64_json.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/",
        model_name: str = "gpt-image-2",
        size: str = "auto",
        quality: str = "medium",
        max_retries: int = 2,
        delay_seconds: float = 2.0,
        timeout: float = 300.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/"
        self.model_name = model_name
        self.size = size
        self.quality = quality
        self.max_retries = max_retries
        self.delay_seconds = delay_seconds
        self.timeout = timeout

    def _endpoint(self) -> str:
        return urljoin(self.base_url, "images/edits")

    def edit_image(self, prompt: str, input_image_path: str, output_image_path: str) -> bool:
        if not self.api_key:
            raise ValueError("Image edit API key is empty.")
        if not prompt or not prompt.strip():
            raise ValueError("Image edit prompt is empty.")

        input_path = Path(input_image_path)
        output_path = Path(output_image_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                image_bytes = self._request_edit(prompt=prompt, input_path=input_path)
                output_path.write_bytes(image_bytes)
                return True
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay_seconds)

        if last_error is not None:
            raise RuntimeError(f"Image edit failed after {self.max_retries} attempts: {last_error}")
        return False

    def _request_edit(self, prompt: str, input_path: Path) -> bytes:
        data: Dict[str, str] = {
            "model": self.model_name,
            "prompt": prompt,
        }
        if self.size:
            data["size"] = self.size
        if self.quality:
            data["quality"] = self.quality

        headers = {"Authorization": f"Bearer {self.api_key}"}
        with input_path.open("rb") as image_file:
            files = {"image": (input_path.name, image_file, self._mime_type(input_path))}
            response = requests.post(
                self._endpoint(),
                headers=headers,
                data=data,
                files=files,
                timeout=self.timeout,
            )

        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")

        payload = response.json()
        b64_json = self._extract_b64_json(payload)
        if b64_json:
            return base64.b64decode(b64_json)

        image_url = self._extract_url(payload)
        if image_url:
            image_response = requests.get(image_url, timeout=self.timeout)
            if image_response.status_code >= 400:
                raise RuntimeError(f"Image URL fetch failed: HTTP {image_response.status_code}")
            return image_response.content

        raise RuntimeError(f"No edited image found in response keys: {list(payload.keys())}")

    def _extract_b64_json(self, payload: Dict[str, Any]) -> Optional[str]:
        data = payload.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                value = first.get("b64_json") or first.get("b64")
                if isinstance(value, str) and value.strip():
                    return value
        return None

    def _extract_url(self, payload: Dict[str, Any]) -> Optional[str]:
        data = payload.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                value = first.get("url")
                if isinstance(value, str) and value.strip():
                    return value
        return None

    def _mime_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".webp":
            return "image/webp"
        return "image/png"
