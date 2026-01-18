#!/usr/bin/env python3
"""
生成3D场景布局

从场景图（包含资产信息）生成3D场景布局，并导出为JSON和GLTF格式。
"""

import json
import sys
from pathlib import Path

from scenethesis.services.scene_layout import SceneLayoutGenerator


def load_scene_graph(scene_graph_path: str):
    """加载场景图JSON文件。"""
    with open(scene_graph_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # 配置路径
    scene_graph_path = "scenethesis/output/scene_graph_with_assets.json"
    output_json_path = "scenethesis/output/scene_layout_3d.json"
    output_gltf_path = "scenethesis/output/scene_layout_3d.gltf"

    print(f"{'='*60}")
    print("3D场景布局生成")
    print(f"{'='*60}\n")

    # 加载场景图
    print(f"✓ 读取场景图: {scene_graph_path}")
    scene_graph = load_scene_graph(scene_graph_path)
    total_objects = len(scene_graph.get("scene_layout", []))
    print(f"  总物体数: {total_objects}\n")

    # 初始化布局生成器
    print(f"{'='*60}")
    print("初始化布局生成器")
    print(f"{'='*60}")

    # 房间尺寸 (宽度, 高度, 深度) in meters
    room_size = (10.0, 3.0, 10.0)
    generator = SceneLayoutGenerator(room_size=room_size)
    print(f"✓ 房间尺寸: {room_size[0]}m × {room_size[1]}m × {room_size[2]}m\n")

    # 生成3D布局
    print(f"{'='*60}")
    print("生成3D布局")
    print(f"{'='*60}")

    try:
        scene_3d = generator.generate_layout(
            scene_graph=scene_graph,
            use_depth=True,
        )
        print(f"✓ 成功生成3D布局")
        print(f"  放置物体数: {len(scene_3d.objects)}")
        print(f"  跳过物体数: {scene_3d.metadata.get('skipped_objects', 0)}\n")

        # 显示物体列表
        print("物体列表:")
        for i, obj in enumerate(scene_3d.objects, 1):
            print(f"  [{i}] {obj.label}")
            print(f"      资产ID: {obj.asset_id[:16]}...")
            print(f"      位置: ({obj.position[0]:.2f}, {obj.position[1]:.2f}, {obj.position[2]:.2f})")
            print(f"      旋转: ({obj.rotation[0]:.1f}°, {obj.rotation[1]:.1f}°, {obj.rotation[2]:.1f}°)")
            print(f"      缩放: ({obj.scale[0]:.2f}, {obj.scale[1]:.2f}, {obj.scale[2]:.2f})")
            print()

    except Exception as e:
        print(f"❌ 布局生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 导出为JSON
    print(f"{'='*60}")
    print("导出场景")
    print(f"{'='*60}")

    try:
        generator.export_to_json(scene_3d, output_json_path)
        print(f"✓ JSON导出成功: {output_json_path}")
    except Exception as e:
        print(f"❌ JSON导出失败: {e}")

    # 导出为GLTF
    try:
        generator.export_to_gltf(scene_3d, output_gltf_path)
        print(f"✓ GLTF导出成功: {output_gltf_path}")
    except Exception as e:
        print(f"⚠️  GLTF导出失败: {e}")
        print(f"   提示: 需要安装trimesh库 (pip install trimesh)")

    print(f"\n{'='*60}")
    print("完成")
    print(f"{'='*60}")
    print(f"3D布局已生成并保存到:")
    print(f"  - {output_json_path}")
    print(f"  - {output_gltf_path}")
    print()


if __name__ == "__main__":
    main()
