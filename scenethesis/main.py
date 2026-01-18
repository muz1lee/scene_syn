from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from scenethesis.modules.planner import CoarseScenePlanner
from scenethesis.modules.refiner import VisualRefinementModule
from scenethesis.services.providers import (
    ImageProviderConfig,
    LLMConfig,
    GeminiProvider,
    create_image_provider,
)
from scenethesis.services.scene_graph import LogicalHierarchyPlanner, SceneGraphBuilder
from scenethesis.services.sam3_client import Sam3Client
from scenethesis.services.depth_pro_client import DepthProClient

CONFIG_PATH = Path("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件 {path} 不存在，请创建 config.yaml")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_plan(plan_json: str, output_dir: str | Path | None) -> Path:
    base_dir = Path(__file__).resolve().parent
    target_dir = Path(output_dir) if output_dir else base_dir / "output"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = "planner_output.json"
    file_path = target_dir / filename
    file_path.write_text(plan_json, encoding="utf-8")
    return file_path


def run_scenethesis_system(config: Dict[str, Any]) -> None:
    prompt = config.get("prompt", "A simple room")
    assets = config.get("db_assets", [])
    model_name = config.get("model_name", "gemini-3-flash")
    output_dir = config.get("output_dir")
    env_maps = config.get("env_maps", [])
    vertex_cfg = config.get("vertex", {})
    vertex_enabled = bool(vertex_cfg.get("enabled", False))
    vertex_project = vertex_cfg.get("project_id")
    vertex_location = vertex_cfg.get("location")
    phase2_cfg = config.get("phase2", {})
    phase2_output_dir = phase2_cfg.get("output_dir") or output_dir
    size_cfg = phase2_cfg.get("guidance_size", [640, 640])
    if not isinstance(size_cfg, (list, tuple)) or len(size_cfg) < 2:
        size_cfg = [640, 640]
    guidance_size = (int(size_cfg[0]), int(size_cfg[1]))
    image_provider_name = phase2_cfg.get("image_provider", "gemini")
    image_model_name = phase2_cfg.get("image_model", "gemini-3-pro-image-preview")
    fallback_models = tuple(phase2_cfg.get("image_model_fallbacks", []) or [])
    image_format = phase2_cfg.get("image_format", "png")
    image_guidance_scale = phase2_cfg.get("guidance_scale")
    negative_prompt = phase2_cfg.get("negative_prompt")
    api_version = phase2_cfg.get("api_version")

    if not assets:
        raise ValueError("配置文件中缺少 db_assets 列表")

    print("🚀 [主循环] 启动 Scenethesis Planner 单元测试...")
    llm_config = LLMConfig(
        model=model_name,
        use_vertex_ai=vertex_enabled,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )
    llm_provider = GeminiProvider(config=llm_config)
    planner = CoarseScenePlanner(assets, llm_provider)
    image_provider_config = ImageProviderConfig(
        model=image_model_name,
        image_size=guidance_size,
        image_format=image_format,
        guidance_scale=image_guidance_scale,
        negative_prompt=negative_prompt,
        fallback_models=fallback_models,
        api_version=api_version,
        use_vertex_ai=vertex_enabled,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )
    image_provider = create_image_provider(image_provider_name, config=image_provider_config)
    scene_graph_cfg = phase2_cfg.get("scene_graph", {}) or {}
    logic_model_name = scene_graph_cfg.get("logic_model")
    if logic_model_name and logic_model_name != model_name:
        logic_llm_config = LLMConfig(
            model=logic_model_name,
            use_vertex_ai=vertex_enabled,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
        )
        logic_llm_provider = GeminiProvider(config=logic_llm_config)
    else:
        logic_llm_provider = llm_provider
    logical_planner = LogicalHierarchyPlanner(logic_llm_provider)
    sam3_client = None
    sam3_cfg = scene_graph_cfg.get("sam3") or {}
    endpoint = sam3_cfg.get("endpoint")
    if endpoint:
        sam3_client = Sam3Client(
            endpoint=endpoint,
            default_text_prompt=sam3_cfg.get("text_prompt"),
        )
    # Depth Pro 本地部署，不再需要 endpoint 配置
    depth_client = DepthProClient(device="cuda")

    # 确定输出目录
    refiner_output_dir = phase2_output_dir or output_dir
    if not refiner_output_dir:
        refiner_output_dir = Path(__file__).resolve().parent / "output"

    scene_graph_builder = SceneGraphBuilder(
        logical_planner=logical_planner,
        sam3_client=sam3_client,
        depth_client=depth_client,
        guidance_size=guidance_size,
        output_dir=refiner_output_dir,
    )
    refiner = VisualRefinementModule(
        asset_database=assets,
        env_map_database=env_maps,
        image_provider=image_provider,
        output_dir=refiner_output_dir,
        guidance_size=guidance_size,
        image_format=image_format,
        scene_graph_builder=scene_graph_builder,
        llm_provider=llm_provider,
    )

    plan = planner.run_pipeline(prompt)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    print("✅ [结果] 粗级规划输出：")
    print(plan_json)

    output_path = _save_plan(plan_json, output_dir)
    print(f"💾 [保存] 规划结果已写入: {output_path}")

    print("🖼️ [细化] 开始 Guidance 图生成...")
    refined = refiner.process_layout(plan)
    refined_json = json.dumps(refined, ensure_ascii=False, indent=2)
    print("✅ [结果] Phase 2 细化输出：")
    print(refined_json)
    refiner_path = _save_refinement_output(refined, refiner_output_dir)
    print(f"💾 [保存] 细化结果与图像位于: {refiner_path}")


if __name__ == "__main__":
    cfg = load_config()
    run_scenethesis_system(cfg)


def _save_refinement_output(refined: Dict[str, Any], output_dir: str | Path | None) -> Path:
    target_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent / "output"
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / "refiner_output.json"
    file_path.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path
