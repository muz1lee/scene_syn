from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import numpy as np
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
    Depth Pro 本地客户端，使用 Apple ml-depth-pro 库进行深度估计。
    """

    def __init__(
        self,
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

        return DepthEstimation(
            depth_map=depth_map,
            min_depth=min_depth,
            max_depth=max_depth,
            median_depth=median_depth,
            raw={
                "depth_shape": depth_map.shape,
                "device": self.device,
            },
        )

