from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scenethesis.services.providers import ImageGenerationProvider
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
    ) -> None:
        self.asset_db = asset_database
        self.env_db = env_map_database
        self.image_provider = image_provider
        self.guidance_size = guidance_size
        self.image_format = image_format
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scene_graph_builder = scene_graph_builder

    def process_layout(self, coarse_plan: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_guidance_prompt(coarse_plan)
        image_bytes = self.image_provider.generate_image(prompt, self.guidance_size)
        guidance_path = self._save_image(image_bytes)
        scene_layout = self._build_scene_graph(coarse_plan, image_bytes)
        env_map = self._select_environment_map()
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

    def _select_environment_map(self) -> str:
        return self.env_db[0] if self.env_db else "default_env.hdr"
