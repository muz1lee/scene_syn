"""
3D场景布局生成模块

从场景图生成3D场景布局，包括物体的位置、旋转、缩放等信息。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Object3D:
    """3D场景中的物体"""

    label: str
    asset_id: str
    asset_path: Optional[str]
    position: Tuple[float, float, float]  # (x, y, z)
    rotation: Tuple[float, float, float]  # (rx, ry, rz) in degrees
    scale: Tuple[float, float, float]  # (sx, sy, sz)
    parent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Scene3D:
    """3D场景布局"""

    objects: List[Object3D]
    anchor: str
    room_size: Tuple[float, float, float] = (10.0, 3.0, 10.0)  # (width, height, depth)
    metadata: Optional[Dict[str, Any]] = None


class SceneLayoutGenerator:
    """
    从场景图生成3D场景布局。

    将2D场景图中的物体转换为3D空间中的位置、旋转和缩放。
    """

    def __init__(
        self,
        room_size: Tuple[float, float, float] = (10.0, 3.0, 10.0),
    ) -> None:
        """
        初始化场景布局生成器。

        Args:
            room_size: 房间尺寸 (宽度, 高度, 深度) in meters
        """
        self.room_size = room_size

    def generate_layout(
        self,
        scene_graph: Dict[str, Any],
        use_depth: bool = True,
    ) -> Scene3D:
        """
        从场景图生成3D布局。

        Args:
            scene_graph: 场景图数据（包含资产信息）
            use_depth: 是否使用深度信息

        Returns:
            Scene3D对象
        """
        scene_layout = scene_graph.get("scene_layout", [])
        anchor = scene_graph.get("anchor", "unknown")

        objects_3d = []

        for obj_data in scene_layout:
            # 跳过没有资产的物体
            retrieved_assets = obj_data.get("retrieved_assets", [])
            if not retrieved_assets:
                print(f"  ⚠️  跳过物体 {obj_data.get('label')}: 没有检索到资产")
                continue

            # 使用第一个检索到的资产
            best_asset = retrieved_assets[0]
            asset_id = best_asset.get("asset_id")
            asset_path = best_asset.get("asset_path")
            asset_metadata = best_asset.get("metadata", {})

            # 获取初始位姿
            initial_pose = obj_data.get("initial_pose", {})

            # 计算3D位置
            position = self._compute_position(
                initial_pose=initial_pose,
                use_depth=use_depth,
            )

            # 计算旋转
            rotation = self._compute_rotation(
                asset_metadata=asset_metadata,
                obj_data=obj_data,
            )

            # 计算缩放
            scale = self._compute_scale(
                initial_pose=initial_pose,
                asset_metadata=asset_metadata,
            )

            # 创建3D物体
            obj_3d = Object3D(
                label=obj_data.get("label", "unknown"),
                asset_id=asset_id,
                asset_path=asset_path,
                position=position,
                rotation=rotation,
                scale=scale,
                parent=obj_data.get("parent"),
                metadata={
                    "role": obj_data.get("role"),
                    "confidence": obj_data.get("confidence"),
                    "bbox_pixel": obj_data.get("bbox_pixel"),
                    "asset_metadata": asset_metadata,
                },
            )

            objects_3d.append(obj_3d)

        # 创建场景
        scene = Scene3D(
            objects=objects_3d,
            anchor=anchor,
            room_size=self.room_size,
            metadata={
                "total_objects": len(objects_3d),
                "skipped_objects": len(scene_layout) - len(objects_3d),
            },
        )

        return scene

    def _compute_position(
        self,
        initial_pose: Dict[str, Any],
        use_depth: bool = True,
    ) -> Tuple[float, float, float]:
        """
        计算物体的3D位置。

        Args:
            initial_pose: 初始位姿数据
            use_depth: 是否使用深度信息

        Returns:
            (x, y, z) 位置坐标
        """
        translation = initial_pose.get("translation", [0.5, 1.0, 0.5])

        # 将归一化坐标转换为房间坐标
        # translation[0]: 归一化的x坐标 (0-1)
        # translation[1]: 深度值 (meters)
        # translation[2]: 归一化的z坐标 (0-1)

        x = translation[0] * self.room_size[0]  # 宽度方向
        z = translation[2] * self.room_size[2]  # 深度方向

        if use_depth:
            # 使用深度估计的深度值
            depth_stats = initial_pose.get("depth_stats", {})
            y = depth_stats.get("point_depth", translation[1])
        else:
            # 使用默认深度
            y = translation[1]

        return (x, y, z)

    def _compute_rotation(
        self,
        asset_metadata: Dict[str, Any],
        obj_data: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        """
        计算物体的旋转角度。

        Args:
            asset_metadata: 资产元数据
            obj_data: 物体数据

        Returns:
            (rx, ry, rz) 旋转角度 (degrees)
        """
        # 从资产元数据获取推荐的旋转角度
        pose_z_rot = asset_metadata.get("pose_z_rot_angle", 0.0)

        # 将弧度转换为角度
        rz = np.degrees(pose_z_rot)

        # 默认不旋转x和y轴
        rx = 0.0
        ry = 0.0

        return (rx, ry, rz)

    def _compute_scale(
        self,
        initial_pose: Dict[str, Any],
        asset_metadata: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        """
        计算物体的缩放比例。

        Args:
            initial_pose: 初始位姿数据
            asset_metadata: 资产元数据

        Returns:
            (sx, sy, sz) 缩放比例
        """
        # 从初始位姿获取bbox尺寸
        bbox = initial_pose.get("bbox", [0.2, 0.2, 0.2])

        # 从资产元数据获取推荐的缩放
        asset_scale = asset_metadata.get("scale", 1.0)

        # 计算缩放比例
        # bbox是归一化的尺寸，需要转换为实际尺寸
        sx = bbox[0] * self.room_size[0] * asset_scale
        sy = bbox[1] * self.room_size[1] * asset_scale
        sz = bbox[2] * self.room_size[2] * asset_scale

        return (sx, sy, sz)

    def export_to_json(
        self,
        scene: Scene3D,
        output_path: str,
    ) -> None:
        """
        将场景导出为JSON格式。

        Args:
            scene: Scene3D对象
            output_path: 输出文件路径
        """
        scene_dict = {
            "anchor": scene.anchor,
            "room_size": {
                "width": scene.room_size[0],
                "height": scene.room_size[1],
                "depth": scene.room_size[2],
            },
            "objects": [
                {
                    "label": obj.label,
                    "asset_id": obj.asset_id,
                    "asset_path": obj.asset_path,
                    "position": {
                        "x": obj.position[0],
                        "y": obj.position[1],
                        "z": obj.position[2],
                    },
                    "rotation": {
                        "x": obj.rotation[0],
                        "y": obj.rotation[1],
                        "z": obj.rotation[2],
                    },
                    "scale": {
                        "x": obj.scale[0],
                        "y": obj.scale[1],
                        "z": obj.scale[2],
                    },
                    "parent": obj.parent,
                    "metadata": obj.metadata,
                }
                for obj in scene.objects
            ],
            "metadata": scene.metadata,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scene_dict, f, indent=2, ensure_ascii=False)

    def export_to_gltf(
        self,
        scene: Scene3D,
        output_path: str,
    ) -> None:
        """
        将场景导出为GLTF格式。

        Args:
            scene: Scene3D对象
            output_path: 输出文件路径
        """
        try:
            import trimesh
            from trimesh.scene import Scene as TrimeshScene
        except ImportError:
            raise RuntimeError(
                "trimesh not installed. Please run: pip install trimesh"
            )

        # 创建trimesh场景
        trimesh_scene = TrimeshScene()

        # 添加房间边界框（可选）
        room_box = trimesh.creation.box(extents=self.room_size)
        room_box.visual.face_colors = [200, 200, 200, 50]  # 半透明灰色
        trimesh_scene.add_geometry(room_box, node_name="room")

        # 添加每个物体
        for obj in scene.objects:
            if not obj.asset_path or not Path(obj.asset_path).exists():
                # 如果没有资产文件，创建一个占位符
                placeholder = trimesh.creation.box(extents=obj.scale)
                placeholder.visual.face_colors = [100, 100, 255, 255]

                # 应用变换
                transform = trimesh.transformations.compose_matrix(
                    translate=obj.position,
                    angles=np.radians(obj.rotation),
                )
                placeholder.apply_transform(transform)

                trimesh_scene.add_geometry(
                    placeholder,
                    node_name=f"{obj.label}_{obj.asset_id[:8]}",
                )
            else:
                # 加载实际的3D资产
                try:
                    mesh = trimesh.load(obj.asset_path)

                    # 应用缩放
                    mesh.apply_scale(obj.scale)

                    # 应用旋转和平移
                    transform = trimesh.transformations.compose_matrix(
                        translate=obj.position,
                        angles=np.radians(obj.rotation),
                    )
                    mesh.apply_transform(transform)

                    trimesh_scene.add_geometry(
                        mesh,
                        node_name=f"{obj.label}_{obj.asset_id[:8]}",
                    )
                except Exception as e:
                    print(f"  ⚠️  加载资产失败 {obj.asset_id}: {e}")

        # 导出为GLTF
        trimesh_scene.export(output_path)
