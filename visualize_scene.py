#!/usr/bin/env python3
"""
可视化3D场景布局
"""

import json
import trimesh
import numpy as np
from pathlib import Path


def visualize_gltf(gltf_path: str):
    """使用trimesh可视化GLTF文件"""
    print(f"加载GLTF文件: {gltf_path}")

    # 加载场景
    scene = trimesh.load(gltf_path)

    print(f"场景类型: {type(scene)}")

    # 显示场景信息
    if isinstance(scene, trimesh.Scene):
        print(f"场景包含 {len(scene.geometry)} 个几何体")
        for name, geom in scene.geometry.items():
            print(f"  - {name}: {type(geom)}")

    # 可视化
    print("\n打开3D查看器...")
    scene.show()


def visualize_json(json_path: str):
    """从JSON文件读取并打印场景信息"""
    print(f"\n加载JSON文件: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n场景信息:")
    print(f"  房间尺寸: {data.get('room_size', 'N/A')}")
    print(f"  物体数量: {len(data.get('objects', []))}")

    print(f"\n物体列表:")
    for i, obj in enumerate(data.get('objects', []), 1):
        print(f"  [{i}] {obj['label']}")
        print(f"      位置: {obj['position']}")
        print(f"      旋转: {obj['rotation']}")
        print(f"      缩放: {obj['scale']}")


def main():
    # 文件路径
    output_dir = Path("scenethesis/output_0118")
    gltf_path = output_dir / "scene_layout_3d.gltf"
    json_path = output_dir / "scene_layout_3d.json"

    print("="*60)
    print("3D场景可视化")
    print("="*60)

    # 显示JSON信息
    if json_path.exists():
        visualize_json(str(json_path))
    else:
        print(f"⚠️  JSON文件不存在: {json_path}")

    # 可视化GLTF
    print("\n" + "="*60)
    if gltf_path.exists():
        visualize_gltf(str(gltf_path))
    else:
        print(f"❌ GLTF文件不存在: {gltf_path}")


if __name__ == "__main__":
    main()
