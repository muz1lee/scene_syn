from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scenethesis.services.providers import ImageGenerationProvider, LLMProvider
from scenethesis.services.scene_graph import SceneGraphBuilder


class VisualRefinementModule:
    """
    Phase 2：根据粗级规划生成 Guidance 图像与初始布局。
    """

    def __init__(
        self,
        asset_database: List[str],
        env_map_database: List[str],
        image_provider: ImageGenerationProvider,
        output_dir: str | Path,
        guidance_size: Tuple[int, int] = (640, 640),
        image_format: str = "png",
        scene_graph_builder: SceneGraphBuilder | None = None,
        llm_provider: Optional[LLMProvider] = None,
    ) -> None:
        self.asset_db = asset_database
        self.env_db = env_map_database
        self.image_provider = image_provider
        self.guidance_size = guidance_size
        self.image_format = image_format
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scene_graph_builder = scene_graph_builder
        self.llm_provider = llm_provider

    def process_layout(self, coarse_plan: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_guidance_prompt(coarse_plan)
        image_bytes = self.image_provider.generate_image(prompt, self.guidance_size)
        guidance_path = self._save_image(image_bytes)
        scene_layout = self._build_scene_graph(coarse_plan, image_bytes)
        env_map = self._select_environment_map(coarse_plan)
        width, height = self.guidance_size
        return {
            "image_guidance": {
                "path": str(guidance_path),
                "width": width,
                "height": height,
                "prompt": prompt,
            },
            "scene_layout": scene_layout,
            "environment_map": env_map,
        }

    def _build_scene_graph(self, plan: Dict[str, Any], image_bytes: bytes) -> List[Dict[str, Any]]:
        if self.scene_graph_builder:
            try:
                return self.scene_graph_builder.build_scene_layout(plan, image_bytes)
            except Exception as exc:
                print(f"⚠️ [Refiner] Scene graph builder 失败，回退到占位实现: {exc}")
        return self._create_layout_entries(plan)

    def _build_guidance_prompt(self, plan: Dict[str, Any]) -> str:
        anchor = plan.get("anchor", "unknown anchor")
        objects = plan.get("objects", [])
        description = plan.get("detailed_description") or "A room scene."
        object_line = ", ".join(objects) if objects else "no specific objects"
        return (
            f"{description}\n"
            f"Anchor object: {anchor}.\n"
            f"Include the following objects in frame: {object_line}.\n"
            "Render a photorealistic reference at 640x640 for layout guidance."
        )

    def _save_image(self, image_bytes: bytes) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"guidance_{timestamp}.{self.image_format}"
        path = self.output_dir / filename
        path.write_bytes(image_bytes)
        return path

    def _create_layout_entries(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        layout = []
        for idx, obj in enumerate(plan.get("objects", [])):
            asset_id = obj if obj in self.asset_db else f"{obj}_placeholder"
            layout.append(
                {
                    "label": obj,
                    "initial_pose": {
                        "translation": [idx * 0.5, 0.0, 0.0],
                        "bbox": [1.0, 1.0, 1.0],
                    },
                    "asset_id": asset_id,
                }
            )
        return layout

    def _select_environment_map(self, coarse_plan: Dict[str, Any]) -> str:
        """
        使用 LLM 根据场景描述智能选择最合适的环境贴图。

        Args:
            coarse_plan: 粗级规划结果，包含场景描述

        Returns:
            选中的环境贴图文件名
        """
        if not self.env_db:
            return "default_env.hdr"

        # 如果只有一个环境贴图，直接返回
        if len(self.env_db) == 1:
            return self.env_db[0]

        # 如果没有 LLM provider，返回第一个
        if not self.llm_provider:
            print("⚠️ [Refiner] 未配置 LLM provider，使用默认环境贴图")
            return self.env_db[0]

        try:
            description = coarse_plan.get("detailed_description", "")
            if not description:
                return self.env_db[0]

            # 构建选择提示
            env_list = "\n".join([f"- {env}" for env in self.env_db])
            prompt = f"""根据以下场景描述，从可用的环境贴图中选择最合适的一个：

场景描述：
{description}

可用的环境贴图：
{env_list}

请只返回最合适的环境贴图文件名（不要包含任何解释）。"""

            response = self.llm_provider.chat(prompt)
            selected = response.strip()

            # 验证选择是否在列表中
            if selected in self.env_db:
                print(f"✓ [Refiner] LLM 选择环境贴图: {selected}")
                return selected
            else:
                # 尝试模糊匹配
                for env in self.env_db:
                    if env in selected or selected in env:
                        print(f"✓ [Refiner] LLM 选择环境贴图（模糊匹配）: {env}")
                        return env

                print(f"⚠️ [Refiner] LLM 返回无效选择 '{selected}'，使用默认")
                return self.env_db[0]

        except Exception as exc:
            print(f"⚠️ [Refiner] 环境贴图选择失败: {exc}")
            return self.env_db[0]
