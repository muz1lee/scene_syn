from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class DepthEstimation:
    depth_map: np.ndarray | None
    min_depth: float | None
    max_depth: float | None
    median_depth: float | None
    raw: dict


class DepthProLocal:
    """
    Depth Pro 本地推理实现，直接调用模型而非 HTTP API。
    适用于 GPU 服务器部署场景。
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            device: 'cuda', 'cpu', 或 'mps' (Apple Silicon)
            model_path: 自定义模型路径（可选，默认自动下载）
        """
        self.device = device
        self.model = None
        self.transform = None
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]) -> None:
        """延迟加载模型，避免导入时就占用 GPU"""
        try:
            import depth_pro
            import torch

            print(f"🔧 [DepthProLocal] 加载 Depth Pro 模型到 {self.device}...")

            # 加载模型和预处理器
            if model_path:
                self.model, self.transform = depth_pro.create_model_and_transforms(
                    checkpoint_path=model_path,
                    device=self.device,
                )
            else:
                # 自动下载预训练模型
                self.model, self.transform = depth_pro.create_model_and_transforms(
                    device=self.device,
                )

            self.model.eval()
            print("✅ [DepthProLocal] 模型加载完成")

        except ImportError as e:
            raise ImportError(
                "请安装 Depth Pro: pip install git+https://github.com/apple/ml-depth-pro.git"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Depth Pro 模型加载失败: {e}") from e

    def infer(self, crop_bytes: bytes) -> DepthEstimation:
        """
        对图像裁剪进行深度估计

        Args:
            crop_bytes: PNG/JPEG 格式的图像字节流

        Returns:
            DepthEstimation 包含深度图和统计信息
        """
        import torch

        # 加载图像
        image = Image.open(BytesIO(crop_bytes)).convert("RGB")

        # 预处理
        image_tensor = self.transform(image).to(self.device)

        # 推理
        with torch.no_grad():
            prediction = self.model.infer(image_tensor)

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
                "shape": depth_map.shape,
                "dtype": str(depth_map.dtype),
            },
        )

    def __del__(self):
        """清理 GPU 资源"""
        if self.model is not None:
            del self.model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
