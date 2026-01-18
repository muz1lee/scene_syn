from __future__ import annotations

<<<<<<< HEAD
=======
import base64
>>>>>>> 3837743f33baf1cb5645a8ce728f2e90d31c73ac
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import numpy as np
<<<<<<< HEAD
=======
import requests
>>>>>>> 3837743f33baf1cb5645a8ce728f2e90d31c73ac
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
<<<<<<< HEAD
    Depth Pro 本地客户端，使用 Apple ml-depth-pro 库进行深度估计。
=======
    Depth Pro REST 客户端，用于根据图像裁剪估计 Metric Depth。
>>>>>>> 3837743f33baf1cb5645a8ce728f2e90d31c73ac
    """

    def __init__(
        self,
<<<<<<< HEAD
        device: str = "cuda",
        model_path: Optional[str] = None,
    ) -> None:
        """
        初始化 Depth Pro 客户端。

        Args:
            device: 设备类型 ('cuda' 或 'cpu')
            model_path: 可选的模型路径，如果为 None 则使用默认模型
        """
        self.device = device
        self.model_path = model_path
        self._model = None
        self._transform = None

    def _load_model(self):
        """延迟加载模型，避免初始化时的开销"""
        if self._model is None:
            try:
                import torch
                import depth_pro

                # 使用正确的 torch dtype 而不是字符串
                precision = torch.float16 if self.device == "cuda" else torch.float32

                self._model, self._transform = depth_pro.create_model_and_transforms(
                    device=self.device,
                    precision=precision
                )
                self._model.eval()
            except ImportError:
                raise RuntimeError(
                    "depth_pro 未安装。请运行: pip install git+https://github.com/apple/ml-depth-pro.git"
                )

    def infer(self, crop_bytes: bytes) -> DepthEstimation:
        """
        对图像进行深度估计。

        Args:
            crop_bytes: 图像的字节数据

        Returns:
            DepthEstimation: 包含深度图和统计信息的结果
        """
        self._load_model()

        # 加载图像
        image = Image.open(BytesIO(crop_bytes)).convert("RGB")

        # 转换图像
        import torch
        image_tensor = self._transform(image).to(self.device)

        # 推理
        with torch.no_grad():
            prediction = self._model.infer(image_tensor)

        # 提取深度图
        depth_map = prediction["depth"].cpu().numpy().squeeze()

        # 计算统计信息
        min_depth = float(depth_map.min())
        max_depth = float(depth_map.max())
        median_depth = float(np.median(depth_map))

=======
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
>>>>>>> 3837743f33baf1cb5645a8ce728f2e90d31c73ac
        return DepthEstimation(
            depth_map=depth_map,
            min_depth=min_depth,
            max_depth=max_depth,
            median_depth=median_depth,
<<<<<<< HEAD
            raw={
                "depth_shape": depth_map.shape,
                "device": self.device,
            },
        )

=======
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

>>>>>>> 3837743f33baf1cb5645a8ce728f2e90d31c73ac
