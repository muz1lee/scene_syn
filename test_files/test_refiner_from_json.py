#!/usr/bin/env python3
"""
测试：读取 planner 输出 JSON，调用 Phase 2 Refiner 生成 Guidance 图像。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenethesis.main import load_config
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 planner JSON 生成 Guidance 图像")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径（默认：config.yaml）",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="scenethesis/output/planner_output.json",
        help="planner 输出 JSON 路径（默认：scenethesis/output/planner_output.json）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    plan_path = Path(args.plan)

    config = load_config(config_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assets = config.get("db_assets", [])
    if not assets:
        raise ValueError("配置文件缺少 db_assets，无法构建 Refiner。")

    env_maps = config.get("env_maps", [])
    vertex_cfg = config.get("vertex", {})
    vertex_enabled = bool(vertex_cfg.get("enabled", False))
    vertex_project = vertex_cfg.get("project_id")
    vertex_location = vertex_cfg.get("location")
    phase2_cfg = config.get("phase2", {})
    guidance_size = phase2_cfg.get("guidance_size", [640, 640])
    if not isinstance(guidance_size, (list, tuple)) or len(guidance_size) < 2:
        guidance_size = [640, 640]
    guidance_tuple = (int(guidance_size[0]), int(guidance_size[1]))

    image_provider_cfg = ImageProviderConfig(
        model=phase2_cfg.get("image_model", "gemini-3.0-pro-image"),
        image_size=guidance_tuple,
        image_format=phase2_cfg.get("image_format", "png"),
        guidance_scale=phase2_cfg.get("guidance_scale"),
        negative_prompt=phase2_cfg.get("negative_prompt"),
        fallback_models=tuple(phase2_cfg.get("image_model_fallbacks", []) or []),
        api_version=phase2_cfg.get("api_version"),
        use_vertex_ai=vertex_enabled,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )
    image_provider = create_image_provider(
        phase2_cfg.get("image_provider", "gemini"),
        config=image_provider_cfg,
    )

    scene_graph_cfg = phase2_cfg.get("scene_graph", {}) or {}
    logic_model = scene_graph_cfg.get("logic_model")
    logical_planner = LogicalHierarchyPlanner()
    if logic_model:
        try:
            logic_llm_cfg = LLMConfig(
                model=logic_model,
                use_vertex_ai=vertex_enabled,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
            )
            logic_llm_provider = GeminiProvider(config=logic_llm_cfg)
            logical_planner = LogicalHierarchyPlanner(logic_llm_provider)
        except Exception as exc:
            print(f"⚠️ [Test] 无法初始化逻辑轨 LLM ({logic_model}): {exc}")

    sam3_client = None
    sam3_cfg = scene_graph_cfg.get("sam3") or {}
    sam3_endpoint = sam3_cfg.get("endpoint")
    if sam3_endpoint:
        sam3_client = Sam3Client(
            endpoint=sam3_endpoint,
            default_text_prompt=sam3_cfg.get("text_prompt"),
        )

    # 使用本地 Depth Pro（不需要 endpoint）
    depth_client = None
    depth_cfg = scene_graph_cfg.get("depth_pro") or {}
    # 默认启用本地 Depth Pro
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        depth_client = DepthProClient(device=device)
        print(f"✓ [Test] 初始化本地 Depth Pro (device={device})")
    except Exception as exc:
        print(f"⚠️ [Test] 无法初始化 Depth Pro: {exc}")

    scene_graph_builder = SceneGraphBuilder(
        logical_planner=logical_planner,
        sam3_client=sam3_client,
        depth_client=depth_client,
        guidance_size=guidance_tuple,
    )

    output_dir = phase2_cfg.get("output_dir") or config.get("output_dir") or "scenethesis/output"
    refiner = VisualRefinementModule(
        asset_database=assets,
        env_map_database=env_maps,
        image_provider=image_provider,
        output_dir=output_dir,
        guidance_size=guidance_tuple,
        image_format=image_provider_cfg.image_format,
        scene_graph_builder=scene_graph_builder,
    )

    print("🧪 [Test] 读取 Planner 输出并生成 Guidance 图...")
    refined = refiner.process_layout(plan)
    image_info = refined["image_guidance"]
    print("✅ 细化完成！")
    print(f"Prompt: {image_info['prompt']}")
    print(f"Image Path: {image_info['path']}")
    scene_layout = refined.get("scene_layout", [])
    print(f"Scene Layout Objects: {len(scene_layout)}")
    for node in scene_layout[:5]:
        pose = node.get("initial_pose", {})
        print(f"  - {node.get('label')}: translation={pose.get('translation')} bbox={pose.get('bbox')}")


if __name__ == "__main__":
    main()
