#!/usr/bin/env python3
"""
测试场景图构建 - 不使用 Depth Pro（避免模型下载）
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


def main() -> None:
    # 读取现有的 planner 输出
    plan_path = Path("scenethesis/output/planner_output.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    # 读取现有的引导图像
    image_path = Path("scenethesis/output/generated_img.png")
    image_bytes = image_path.read_bytes()

    print("=" * 60)
    print("场景图构建测试（不含 Depth Pro）")
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
            model="gemini-2.5-pro",
            use_vertex_ai=True,
            vertex_project="dp-dev-465308",
            vertex_location="uscentral1",
        )
        logic_llm_provider = GeminiProvider(config=logic_llm_cfg)
        logical_planner = LogicalHierarchyPlanner(logic_llm_provider)
        print("✓ 初始化 Gemini LLM provider (gemini-2.5-pro)")
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

    # 3. 构建完整场景图（不使用 Depth Pro）
    print("\n" + "=" * 60)
    print("步骤 3: 构建完整场景图（不含深度信息）")
    print("=" * 60)

    scene_graph_builder = SceneGraphBuilder(
        logical_planner=logical_planner,
        sam3_client=sam3_client,
        depth_client=None,  # 不使用 Depth Pro
        guidance_size=(640, 640),
        output_dir=Path("scenethesis/output"),
    )

    print("\n开始处理物体...")
    scene_layout = scene_graph_builder.build_scene_layout(plan, image_bytes)

    # 4. 输出结果
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

    # 检查生成的 mask 和 crop 文件
    mask_dir = Path("scenethesis/output/masks")
    crop_dir = Path("scenethesis/output/crops")
    if mask_dir.exists():
        masks = list(mask_dir.glob("*.png"))
        print(f"\n📁 生成的 mask 文件: {len(masks)} 个")
        for mask in masks[:5]:
            print(f"   - {mask.name}")
        if len(masks) > 5:
            print(f"   ... 还有 {len(masks) - 5} 个")

    if crop_dir.exists():
        crops = list(crop_dir.glob("*.png"))
        print(f"\n📁 生成的 crop 文件: {len(crops)} 个")
        for crop in crops[:5]:
            print(f"   - {crop.name}")
        if len(crops) > 5:
            print(f"   ... 还有 {len(crops) - 5} 个")


if __name__ == "__main__":
    main()
