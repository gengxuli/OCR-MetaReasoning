import base64
import io
from typing import Literal

from PIL import Image


def pillow_to_base64_data_url(
    image: Image.Image,
    image_format: Literal["PNG", "JPEG"] = "PNG",
) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if image_format == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"

