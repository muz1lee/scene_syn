from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image


@dataclass
class DepthEstimation:
    depth_map: np.ndarray | None
    min_depth: float | None
    max_depth: float | None
    median_depth: float | None
    raw: dict


class DepthProClient:
    """
    Depth Pro REST 客户端，用于根据图像裁剪估计 Metric Depth。
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        timeout: int = 60,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = session or requests.Session()

    def infer(self, crop_bytes: bytes) -> DepthEstimation:
        payload = self._build_payload(crop_bytes)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self.session.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if data.get("success") is False:
            raise RuntimeError(f"Depth Pro 推理失败: {data}")
        depth_map = self._extract_depth_map(data)
        stats = data.get("stats") or {}
        if depth_map is not None and depth_map.size > 0:
            min_depth = float(depth_map.min())
            max_depth = float(depth_map.max())
            median_depth = float(np.median(depth_map))
        else:
            min_depth = stats.get("min_depth")
            max_depth = stats.get("max_depth")
            median_depth = stats.get("median_depth")
            if min_depth is not None:
                min_depth = float(min_depth)
            if max_depth is not None:
                max_depth = float(max_depth)
            if median_depth is not None:
                median_depth = float(median_depth)
        return DepthEstimation(
            depth_map=depth_map,
            min_depth=min_depth,
            max_depth=max_depth,
            median_depth=median_depth,
            raw=data,
        )

    def _build_payload(self, image_bytes: bytes) -> dict[str, Any]:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return {"image": image_b64}

    def _extract_depth_map(self, data: dict[str, Any]) -> np.ndarray | None:
        if "depth_map_b64" in data:
            return self._decode_depth_image(data["depth_map_b64"])
        if "depth_map" in data:
            array = np.array(data["depth_map"], dtype=np.float32)
            return array
        return None

    @staticmethod
    def _decode_depth_image(image_b64: str) -> np.ndarray:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("F")
        return np.array(image, dtype=np.float32)

