#!/usr/bin/env python3
"""
使用 Depth Pro 为场景图添加深度信息
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenethesis.services.depth_pro_client import DepthProClient
from PIL import Image
import numpy as np


def main() -> None:
    print("=" * 60)
    print("Depth Pro 深度估计")
    print("=" * 60)

    # 读取场景图
    scene_graph_path = Path("scenethesis/output/scene_graph_output.json")
    scene_graph = json.loads(scene_graph_path.read_text(encoding="utf-8"))
    print(f"\n✓ 读取场景图: {scene_graph_path}")
    print(f"  物体数量: {len(scene_graph['scene_layout'])}")

    # 读取引导图像
    image_path = Path("scenethesis/output/generated_img.png")
    image = Image.open(image_path).convert("RGB")
    image_bytes = image_path.read_bytes()
    width, height = image.size
    print(f"\n✓ 读取引导图像: {image_path}")
    print(f"  图像尺寸: {width} x {height}")

    # 初始化 Depth Pro
    print("\n" + "=" * 60)
    print("初始化 Depth Pro")
    print("=" * 60)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  使用设备: {device}")

        depth_client = DepthProClient(device=device)
        print("✓ Depth Pro 初始化成功")
    except Exception as exc:
        print(f"❌ 无法初始化 Depth Pro: {exc}")
        return

    # 对整个图像进行深度估计
    print("\n" + "=" * 60)
    print("深度估计")
    print("=" * 60)
    print("正在处理图像...")

    try:
        depth_estimation = depth_client.infer(image_bytes)
        print("✓ 深度估计完成")
        print(f"  深度图尺寸: {depth_estimation.depth_map.shape}")
        print(f"  深度范围: [{depth_estimation.min_depth:.3f}, {depth_estimation.max_depth:.3f}]")
        print(f"  中位深度: {depth_estimation.median_depth:.3f}")
    except Exception as exc:
        print(f"❌ 深度估计失败: {exc}")
        import traceback
        traceback.print_exc()
        return

    # 保存深度图
    depth_map_path = Path("scenethesis/output/depth_map.npy")
    np.save(depth_map_path, depth_estimation.depth_map)
    print(f"\n💾 深度图已保存: {depth_map_path}")

    # 保存深度图可视化
    depth_vis_path = Path("scenethesis/output/depth_map_vis.png")
    depth_normalized = (depth_estimation.depth_map - depth_estimation.min_depth) / (
        depth_estimation.max_depth - depth_estimation.min_depth
    )
    depth_vis = (depth_normalized * 255).astype(np.uint8)
    Image.fromarray(depth_vis).save(depth_vis_path)
    print(f"💾 深度图可视化已保存: {depth_vis_path}")

    # 更新场景图中的深度信息
    print("\n" + "=" * 60)
    print("更新场景图深度信息")
    print("=" * 60)

    depth_h, depth_w = depth_estimation.depth_map.shape

    for node in scene_graph['scene_layout']:
        label = node['label']
        pose = node['initial_pose']

        # 从归一化坐标获取图像坐标
        norm_x = pose['translation'][0]
        norm_y = pose['translation'][2]  # translation[2] 是 y 坐标

        # 转换为深度图坐标
        depth_x = int(norm_x * depth_w)
        depth_y = int(norm_y * depth_h)

        # 确保坐标在范围内
        depth_x = max(0, min(depth_w - 1, depth_x))
        depth_y = max(0, min(depth_h - 1, depth_y))

        # 提取该位置的深度值
        point_depth = float(depth_estimation.depth_map[depth_y, depth_x])

        # 更新 pose 中的深度信息
        pose['translation'][1] = round(point_depth, 4)  # 更新 y 坐标为深度值

        # 添加深度统计信息
        pose['depth_stats'] = {
            'min': float(depth_estimation.min_depth),
            'max': float(depth_estimation.max_depth),
            'median': float(depth_estimation.median_depth),
            'point_depth': point_depth,
        }

        print(f"  {label}: depth={point_depth:.3f} at ({norm_x:.3f}, {norm_y:.3f})")

    # 保存更新后的场景图
    output_path = Path("scenethesis/output/scene_graph_with_depth.json")
    output_path.write_text(json.dumps(scene_graph, indent=2, ensure_ascii=False))
    print(f"\n💾 更新后的场景图已保存: {output_path}")

    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - {depth_map_path} (深度图数据)")
    print(f"  - {depth_vis_path} (深度图可视化)")
    print(f"  - {output_path} (带深度信息的场景图)")


if __name__ == "__main__":
    main()
