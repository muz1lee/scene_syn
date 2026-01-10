from __future__ import annotations

from typing import Any, Dict, List


class VisualRefinementModule:
    """
    Phase 2 占位实现：当前仅返回固定结构，后续将接入图像生成与几何解析。
    """

    def __init__(self, asset_database: List[str], env_map_database: List[str]) -> None:
        self.asset_db = asset_database
        self.env_db = env_map_database

    def process_layout(self, coarse_plan: Dict[str, Any]) -> Dict[str, Any]:
        print("🖼️ [细化] 占位实现，仅返回 mock 布局。")
        return {
            "image_guidance": "mock_image_guidance",
            "scene_layout": [
                {
                    "label": obj,
                    "initial_pose": {"pos": [0, 0, 0], "bbox": [1, 1, 1]},
                    "asset_id": f"{obj}_placeholder.glb",
                }
                for obj in coarse_plan.get("objects", [])
            ],
            "environment_map": self.env_db[0] if self.env_db else "default_env.hdr",
        }

