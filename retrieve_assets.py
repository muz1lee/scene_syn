#!/usr/bin/env python3
"""
为场景图中的每个物体检索匹配的3D资产。

使用CLIP进行基于图像的语义检索，从Objaverse数据库中找到最匹配的3D模型。
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

from scenethesis.services.clip_retrieval import CLIPRetrieval, AssetMatch


def load_scene_graph(scene_graph_path: str) -> Dict[str, Any]:
    """加载场景图JSON文件。"""
    with open(scene_graph_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_scene_graph(scene_graph: Dict[str, Any], output_path: str):
    """保存更新后的场景图。"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scene_graph, f, indent=2, ensure_ascii=False)


def retrieve_assets_for_scene(
    scene_graph: Dict[str, Any],
    clip_retrieval: CLIPRetrieval,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    为场景图中的每个物体检索3D资产。

    Args:
        scene_graph: 场景图数据
        clip_retrieval: CLIP检索客户端
        top_k: 为每个物体返回的候选资产数量

    Returns:
        更新后的场景图，包含资产检索结果
    """
    scene_layout = scene_graph.get("scene_layout", [])

    print(f"\n{'='*60}")
    print("开始资产检索")
    print(f"{'='*60}\n")
    print(f"物体数量: {len(scene_layout)}")
    print(f"每个物体检索 top-{top_k} 候选资产\n")

    for idx, obj in enumerate(scene_layout):
        label = obj.get("label", "unknown")
        crop_path = obj.get("crop_path")

        print(f"[{idx + 1}/{len(scene_layout)}] 检索物体: {label}")

        if not crop_path:
            print(f"  ⚠️  未找到crop图像路径，跳过")
            obj["retrieved_assets"] = []
            continue

        crop_path = Path(crop_path)
        if not crop_path.exists():
            print(f"  ⚠️  Crop图像不存在: {crop_path}")
            obj["retrieved_assets"] = []
            continue

        try:
            # 使用基于文本标签的检索（不需要CLIP模型）
            matches = clip_retrieval.retrieve_by_text_label(
                label=label,
                top_k=top_k,
            )

            # 保存检索结果
            retrieved_assets = []
            for rank, match in enumerate(matches, 1):
                asset_info = {
                    "rank": rank,
                    "asset_id": match.asset_id,
                    "similarity": match.similarity,
                    "asset_path": match.asset_path,
                    "metadata": match.metadata,
                }
                retrieved_assets.append(asset_info)

                print(f"  ✓ Rank {rank}: {match.asset_id} (相似度: {match.similarity:.4f})")

            obj["retrieved_assets"] = retrieved_assets

        except Exception as e:
            print(f"  ❌ 检索失败: {e}")
            obj["retrieved_assets"] = []

        print()

    return scene_graph


def main():
    # 配置路径
    scene_graph_path = "scenethesis/output/scene_graph_output.json"
    output_path = "scenethesis/output/scene_graph_with_assets.json"

    print(f"{'='*60}")
    print("3D资产检索")
    print(f"{'='*60}\n")

    # 加载场景图
    print(f"✓ 读取场景图: {scene_graph_path}")
    scene_graph = load_scene_graph(scene_graph_path)
    print(f"  物体数量: {len(scene_graph.get('scene_layout', []))}\n")

    # 初始化CLIP检索
    print(f"{'='*60}")
    print("初始化CLIP检索系统")
    print(f"{'='*60}")
    try:
        clip_retrieval = CLIPRetrieval(device="cuda")
        print("✓ CLIP检索系统初始化成功\n")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)

    # 为每个物体检索资产
    try:
        scene_graph = retrieve_assets_for_scene(
            scene_graph=scene_graph,
            clip_retrieval=clip_retrieval,
            top_k=3,
        )
    except Exception as e:
        print(f"\n❌ 资产检索失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 保存结果
    print(f"{'='*60}")
    print("保存结果")
    print(f"{'='*60}")
    save_scene_graph(scene_graph, output_path)
    print(f"✓ 场景图已保存: {output_path}\n")

    # 统计信息
    total_objects = len(scene_graph.get("scene_layout", []))
    objects_with_assets = sum(
        1 for obj in scene_graph.get("scene_layout", [])
        if obj.get("retrieved_assets")
    )

    print(f"{'='*60}")
    print("检索完成")
    print(f"{'='*60}")
    print(f"总物体数: {total_objects}")
    print(f"成功检索: {objects_with_assets}")
    print(f"检索率: {objects_with_assets / total_objects * 100:.1f}%\n")


if __name__ == "__main__":
    main()
