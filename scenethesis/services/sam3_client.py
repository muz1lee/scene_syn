from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, List, Optional, Sequence, Tuple

import requests
from PIL import Image


@dataclass
class Sam3Detection:
    prompt: str
    score: float
    bbox: Tuple[int, int, int, int]
    mask_image: Image.Image | None
    raw: dict

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "score": self.score,
            "bbox": list(self.bbox),
            "has_mask": self.mask_image is not None,
        }


class Sam3Client:
    """
    轻量封装 SAM3 推理 HTTP 接口，兼容文本/框/组合提示。
    """

    def __init__(
        self,
        endpoint: str,
        *,
        default_text_prompt: str | None = None,
        timeout: int = 60,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.default_text_prompt = default_text_prompt
        self.timeout = timeout
        self.session = session or requests.Session()

    def segment(
        self,
        image_bytes: bytes,
        *,
        text_prompt: str | None = None,
        box_prompt: Sequence[float] | None = None,
    ) -> List[Sam3Detection]:
        payload = self._build_payload(image_bytes, text_prompt, box_prompt)
        response = self.session.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            return []
        if data.get("success") is False:
            raise RuntimeError(f"SAM3 推理失败: {data}")
        detections = data.get("detections") or []
        return [self._build_detection(det) for det in detections]

    def _build_payload(
        self,
        image_bytes: bytes,
        text_prompt: str | None,
        box_prompt: Sequence[float] | None,
    ) -> dict:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload: dict[str, Any] = {"image": image_b64}
        resolved_text = text_prompt or self.default_text_prompt
        if resolved_text:
            payload["text_prompt"] = resolved_text
        if box_prompt:
            payload["box_prompt"] = list(box_prompt)
        return payload

    def _build_detection(self, det: dict[str, Any]) -> Sam3Detection:
        bbox = det.get("bbox") or [0, 0, 0, 0]
        bbox_tuple: Tuple[int, int, int, int] = tuple(int(x) for x in bbox[:4])
        mask_b64 = det.get("mask")
        mask_image = (
            self._decode_mask(mask_b64) if isinstance(mask_b64, str) and mask_b64 else None
        )
        return Sam3Detection(
            prompt=str(det.get("prompt") or ""),
            score=float(det.get("score") or 0.0),
            bbox=bbox_tuple,
            mask_image=mask_image,
            raw=det,
        )

    @staticmethod
    def _decode_mask(mask_b64: str) -> Image.Image:
        mask_bytes = base64.b64decode(mask_b64)
        return Image.open(BytesIO(mask_bytes)).convert("L")
