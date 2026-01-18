#!/usr/bin/env python3
"""
将 AI2-THOR .pkl.gz 资产转换为 .glb 格式
"""

import gzip
import pickle
import numpy as np
from pathlib import Path
import sys


def convert_pkl_to_glb(pkl_path: Path, output_path: Path):
    """
    将 .pkl.gz 文件转换为 .glb 格式

    Args:
        pkl_path: 输入的 .pkl.gz 文件路径
        output_path: 输出的 .glb 文件路径
    """
    try:
        import trimesh
        from PIL import Image

        # 加载 pkl.gz 文件
        with gzip.open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # 提取mesh数据 - vertices是字典列表 {'x': ..., 'y': ..., 'z': ...}
        vertices_list = data['vertices']
        vertices = np.array([[v['x'], v['y'], v['z']] for v in vertices_list])

        # triangles是平面列表,需要重塑为 (n, 3)
        triangles_flat = data['triangles']
        faces = np.array(triangles_flat).reshape(-1, 3)

        # normals也是字典列表
        normals_list = data.get('normals', [])
        if normals_list:
            normals = np.array([[n['x'], n['y'], n['z']] for n in normals_list])
        else:
            normals = None

        # uvs是字典列表 {'x': u, 'y': v}
        uvs_list = data.get('uvs', [])
        if uvs_list:
            uvs = np.array([[uv['x'], uv['y']] for uv in uvs_list])
        else:
            uvs = None

        # 创建trimesh对象
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
        )

        # 加载纹理 - albedo.jpg在同一目录
        texture_dir = pkl_path.parent
        albedo_path = texture_dir / 'albedo.jpg'

        if albedo_path.exists() and uvs is not None:
            try:
                # 加载纹理图像
                texture_image = Image.open(albedo_path)
                # 创建带纹理的visual
                material = trimesh.visual.material.SimpleMaterial(
                    image=texture_image,
                    diffuse=[255, 255, 255, 255]
                )
                mesh.visual = trimesh.visual.TextureVisuals(
                    uv=uvs,
                    material=material
                )
            except Exception as e:
                print(f"  ⚠️  加载纹理失败: {e}, 使用默认颜色")
                # 使用默认灰色
                mesh.visual.vertex_colors = [200, 200, 200, 255]
        else:
            # 没有纹理,使用默认颜色
            mesh.visual.vertex_colors = [200, 200, 200, 255]

        # 导出为GLB
        mesh.export(output_path)

        return True

    except Exception as e:
        print(f"  ❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_scene_assets(scene_graph_path: str):
    """
    转换场景图中所有资产的 .pkl.gz 文件为 .glb 格式

    Args:
        scene_graph_path: 场景图JSON文件路径
    """
    import json

    # 加载场景图
    with open(scene_graph_path, 'r') as f:
        scene_graph = json.load(f)

    # 收集所有需要的资产ID
    asset_ids = set()
    for obj in scene_graph.get('scene_layout', []):
        for asset in obj.get('retrieved_assets', []):
            asset_id = asset.get('asset_id')
            asset_path = asset.get('asset_path')
            if asset_id and asset_path and asset_path.endswith('.pkl.gz'):
                asset_ids.add(asset_id)

    print(f"找到 {len(asset_ids)} 个需要转换的资产")
    print()

    assets_base = Path.home() / ".objathor-assets" / "2023_09_23" / "assets"
    success_count = 0
    fail_count = 0

    for i, asset_id in enumerate(asset_ids, 1):
        pkl_path = assets_base / asset_id / f"{asset_id}.pkl.gz"
        glb_path = assets_base / asset_id / f"{asset_id}.glb"

        print(f"[{i}/{len(asset_ids)}] {asset_id}")

        # 如果GLB已存在,跳过
        if glb_path.exists():
            print(f"  ✓ GLB已存在,跳过")
            success_count += 1
            continue

        if not pkl_path.exists():
            print(f"  ⚠️  PKL文件不存在")
            fail_count += 1
            continue

        # 转换
        if convert_pkl_to_glb(pkl_path, glb_path):
            print(f"  ✓ 转换成功 ({glb_path.stat().st_size / 1024:.1f} KB)")
            success_count += 1
        else:
            fail_count += 1

    print()
    print(f"{'='*60}")
    print(f"转换完成: {success_count} 成功, {fail_count} 失败")
    print(f"{'='*60}")


def main():
    import sys

    if len(sys.argv) > 1:
        # 批量转换场景资产
        scene_graph_path = sys.argv[1]
        print(f"{'='*60}")
        print("批量转换场景资产")
        print(f"{'='*60}")
        print(f"场景图: {scene_graph_path}")
        print()
        convert_scene_assets(scene_graph_path)
    else:
        # 测试单个文件
        test_asset_id = "b3eeaeb090e2450eaf6428f5476d57a5"
        assets_base = Path.home() / ".objathor-assets" / "2023_09_23" / "assets"
        pkl_path = assets_base / test_asset_id / f"{test_asset_id}.pkl.gz"
        glb_path = assets_base / test_asset_id / f"{test_asset_id}.glb"

        print(f"测试转换: {test_asset_id}")
        print(f"输入: {pkl_path}")
        print(f"输出: {glb_path}")
        print()

        if pkl_path.exists():
            print("开始转换...")
            if convert_pkl_to_glb(pkl_path, glb_path):
                print(f"✓ 转换成功!")
                print(f"  GLB文件: {glb_path}")
                print(f"  文件大小: {glb_path.stat().st_size / 1024:.1f} KB")
            else:
                print("❌ 转换失败")
        else:
            print(f"❌ 文件不存在: {pkl_path}")


if __name__ == "__main__":
    main()
