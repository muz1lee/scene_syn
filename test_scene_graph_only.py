#!/usr/bin/env python3
"""
直接使用现有的 planner 输出和引导图像，测试场景图构建功能。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenethesis.services.providers import LLMConfig, GeminiProvider
from scenethesis.services.scene_graph import LogicalHierarchyPlanner, SceneGraphBuilder
from scenethesis.services.sam3_client import Sam3Client
from scenethesis.services.depth_pro_client import DepthProClient


def main() -> None:
    # 读取现有的 planner 输出
    plan_path = Path("scenethesis/output/planner_output.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    # 读取现有的引导图像
    image_path = Path("scenethesis/output/generated_img.png")
    image_bytes = image_path.read_bytes()

    print("=" * 60)
    print("场景图构建测试")
    print("=" * 60)
    print(f"\n📄 Planner 输出: {plan_path}")
    print(f"🖼️  引导图像: {image_path}")
    print(f"🎯 锚点对象: {plan.get('anchor')}")
    print(f"📦 物体列表: {', '.join(plan.get('objects', []))}")

    # 1. 初始化逻辑层级规划器（使用 Gemini LLM）
    print("\n" + "=" * 60)
    print("步骤 1: 构建逻辑层级场景图 (Ground/Parent/Child)")
    print("=" * 60)

    try:
        logic_llm_cfg = LLMConfig(
            model="gemini-2-flash",
            use_vertex_ai=True,
            vertex_project="dp-dev-465308",
            vertex_location="uscentral1",
        )
        logic_llm_provider = GeminiProvider(config=logic_llm_cfg)
        logical_planner = LogicalHierarchyPlanner(logic_llm_provider)
        print("✓ 初始化 Gemini LLM provider (gemini-2-flash)")
    except Exception as exc:
        print(f"⚠️ 无法初始化 LLM，使用启发式方法: {exc}")
        logical_planner = LogicalHierarchyPlanner()

    # 构建逻辑层级
    hierarchy = logical_planner.plan_hierarchy(plan)
    print(f"\n✓ 逻辑层级构建完成")
    print(f"  锚点: {hierarchy.anchor}")
    print(f"  角色分配:")
    for obj, role in hierarchy.roles.items():
        parent = hierarchy.parents.get(obj, "unknown")
        print(f"    - {obj}: {role} (parent: {parent})")

    # 2. 初始化 SAM3 客户端
    print("\n" + "=" * 60)
    print("步骤 2: SAM3 分割")
    print("=" * 60)

    sam3_client = Sam3Client(
        endpoint="http://101.132.143.105:5081/segment",
        default_text_prompt="",
    )
    print("✓ 初始化 SAM3 客户端")
    print(f"  Endpoint: http://101.132.143.105:5081/segment")

    # 3. 初始化 Depth Pro 客户端
    print("\n" + "=" * 60)
    print("步骤 3: Depth Pro 深度估计")
    print("=" * 60)

    depth_client = None
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        depth_client = DepthProClient(device=device)
        print(f"✓ 初始化本地 Depth Pro")
        print(f"  Device: {device}")
    except Exception as exc:
        print(f"⚠️ 无法初始化 Depth Pro: {exc}")
        print("  将继续运行，但不会有深度信息")

    # 4. 构建完整场景图
    print("\n" + "=" * 60)
    print("步骤 4: 构建完整场景图")
    print("=" * 60)

    scene_graph_builder = SceneGraphBuilder(
        logical_planner=logical_planner,
        sam3_client=sam3_client,
        depth_client=depth_client,
        guidance_size=(640, 640),
        output_dir=Path("scenethesis/output"),
    )

    print("\n开始处理物体...")
    scene_layout = scene_graph_builder.build_scene_layout(plan, image_bytes)

    # 5. 输出结果
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)

    print(f"\n✅ 场景图构建完成！共 {len(scene_layout)} 个物体节点\n")

    for idx, node in enumerate(scene_layout, 1):
        print(f"{idx}. {node.get('label')}")
        print(f"   角色: {node.get('role')}")
        print(f"   父节点: {node.get('parent')}")
        print(f"   置信度: {node.get('confidence', 0.0):.3f}")

        bbox = node.get('bbox_pixel', [0, 0, 0, 0])
        print(f"   边界框: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")

        pose = node.get('initial_pose', {})
        translation = pose.get('translation', [0, 0, 0])
        bbox_3d = pose.get('bbox', [0, 0, 0])
        print(f"   位置: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
        print(f"   尺寸: [{bbox_3d[0]:.3f}, {bbox_3d[1]:.3f}, {bbox_3d[2]:.3f}]")

        if 'depth_stats' in pose:
            depth_stats = pose['depth_stats']
            print(f"   深度: min={depth_stats.get('min', 0):.3f}, "
                  f"max={depth_stats.get('max', 0):.3f}, "
                  f"median={depth_stats.get('median', 0):.3f}")

        mask_path = node.get('mask_path')
        crop_path = node.get('crop_path')
        if mask_path:
            print(f"   Mask: {mask_path}")
        if crop_path:
            print(f"   Crop: {crop_path}")
        print()

    # 保存结果
    output_path = Path("scenethesis/output/scene_graph_output.json")
    output_data = {
        "anchor": hierarchy.anchor,
        "scene_layout": scene_layout,
    }
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"💾 场景图已保存到: {output_path}")


if __name__ == "__main__":
    main()
