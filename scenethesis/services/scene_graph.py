from __future__ import annotations

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from scenethesis.services.depth_pro_client import DepthEstimation, DepthProClient
from scenethesis.services.providers import LLMProvider
from scenethesis.services.sam3_client import Sam3Client, Sam3Detection


@dataclass
class HierarchyPlan:
    anchor: str
    roles: Dict[str, str]
    parents: Dict[str, str]


class LogicalHierarchyPlanner:
    """
    逻辑轨：利用 Gemini/GPT 输出 Ground/Parent/Child 层级结构。
    """

    SYSTEM_PROMPT = """
你是场景布局的逻辑规划助手。请根据输入的 anchor、objects 与详细描述，输出 JSON：
{
  "anchor": "<anchor name>",
  "nodes": [
     {"label": "<object>", "role": "ground|parent|child", "parent": "<parent_label or ground>"}
  ]
}
要求：
1. anchor 必须存在于 nodes 中，role 默认 parent。
2. 没有父节点的对象 parent 设为 "ground"。
3. 输出必须是合法 JSON，不能包含额外文本或注释。
"""

    def __init__(self, llm_provider: Optional[LLMProvider] = None) -> None:
        self.llm = llm_provider

    def plan_hierarchy(self, coarse_plan: Dict[str, Any]) -> HierarchyPlan:
        if not self.llm:
            return self._heuristic_plan(coarse_plan)
        user_prompt = self._build_user_prompt(coarse_plan)
        try:
            result = self.llm.generate_json(self.SYSTEM_PROMPT, user_prompt)
            return self._parse_llm_result(result, coarse_plan)
        except Exception as exc:
            print(f"⚠️ [LogicalHierarchyPlanner] LLM 规划失败，使用启发式方案: {exc}")
            return self._heuristic_plan(coarse_plan)

    def _heuristic_plan(self, coarse_plan: Dict[str, Any]) -> HierarchyPlan:
        objects = coarse_plan.get("objects") or []
        anchor = coarse_plan.get("anchor") or (objects[0] if objects else "ground")
        roles: Dict[str, str] = {}
        parents: Dict[str, str] = {}
        for obj in objects:
            if obj == anchor:
                roles[obj] = "parent"
                parents[obj] = "ground"
            else:
                roles[obj] = "child"
                parents[obj] = anchor
        if anchor not in roles:
            roles[anchor] = "parent"
            parents[anchor] = "ground"
        return HierarchyPlan(anchor=anchor, roles=roles, parents=parents)

    def _parse_llm_result(self, data: dict, coarse_plan: Dict[str, Any]) -> HierarchyPlan:
        nodes = data.get("nodes") or []
        anchor = str(data.get("anchor") or coarse_plan.get("anchor") or "ground")
        roles: Dict[str, str] = {}
        parents: Dict[str, str] = {}
        for node in nodes:
            label = str(node.get("label") or "").strip()
            if not label:
                continue
            roles[label] = node.get("role") or ("parent" if label == anchor else "child")
            parents[label] = node.get("parent") or ("ground" if label == anchor else anchor)
        if anchor not in roles:
            roles[anchor] = "parent"
            parents[anchor] = "ground"
        return HierarchyPlan(anchor=anchor, roles=roles, parents=parents)

    def _build_user_prompt(self, coarse_plan: Dict[str, Any]) -> str:
        objects = coarse_plan.get("objects") or []
        description = coarse_plan.get("detailed_description") or ""
        prompt_dict = {
            "anchor": coarse_plan.get("anchor"),
            "objects": objects,
            "description": description,
        }
        return json.dumps(prompt_dict, ensure_ascii=False)


class SceneGraphBuilder:
    """
    几何轨：SAM3 + Depth Pro 推理链，结合逻辑轨输出节点 pose/bbox。
    """

    def __init__(
        self,
        logical_planner: LogicalHierarchyPlanner,
        sam3_client: Optional[Sam3Client],
        depth_client: Optional[DepthProClient],
        guidance_size: Tuple[int, int],
    ) -> None:
        self.logical_planner = logical_planner
        self.sam3_client = sam3_client
        self.depth_client = depth_client
        self.guidance_size = guidance_size

    def build_scene_layout(self, coarse_plan: Dict[str, Any], guidance_image: bytes) -> List[Dict[str, Any]]:
        hierarchy = self.logical_planner.plan_hierarchy(coarse_plan)
        objects = coarse_plan.get("objects") or []
        if not self.sam3_client:
            return [self._placeholder_node(obj, idx, hierarchy, len(objects)) for idx, obj in enumerate(objects)]
        base_image = Image.open(BytesIO(guidance_image)).convert("RGB")
        width, height = base_image.size
        nodes: List[Dict[str, Any]] = []
        for idx, obj in enumerate(objects):
            detection = self._detect_object(guidance_image, obj)
            if not detection:
                nodes.append(self._placeholder_node(obj, idx, hierarchy, len(objects)))
                continue
            depth_est = self._estimate_depth(base_image, detection)
            pose = self._lift_pose(detection, depth_est, width, height)
            nodes.append(
                {
                    "label": obj,
                    "role": hierarchy.roles.get(obj, "child"),
                    "parent": hierarchy.parents.get(obj, hierarchy.anchor),
                    "confidence": detection.score,
                    "bbox_pixel": list(detection.bbox),
                    "initial_pose": pose,
                }
            )
        return nodes

    def _detect_object(self, image_bytes: bytes, label: str) -> Sam3Detection | None:
        if not self.sam3_client:
            return None
        try:
            detections = self.sam3_client.segment(image_bytes, text_prompt=label)
        except Exception as exc:
            print(f"⚠️ [SceneGraphBuilder] SAM3 请求失败 ({label}): {exc}")
            return None
        if not detections:
            print(f"⚠️ [SceneGraphBuilder] SAM3 未检测到目标: {label}")
            return None
        return max(detections, key=lambda det: det.score)

    def _estimate_depth(self, base_image: Image.Image, detection: Sam3Detection) -> DepthEstimation | None:
        if not self.depth_client:
            return None
        crop_bytes = self._crop_to_bytes(base_image, detection.bbox)
        try:
            return self.depth_client.infer(crop_bytes)
        except Exception as exc:
            print(f"⚠️ [SceneGraphBuilder] Depth Pro 估计失败: {exc}")
            return None

    @staticmethod
    def _crop_to_bytes(image: Image.Image, bbox: Tuple[int, int, int, int]) -> bytes:
        x1, y1, x2, y2 = bbox
        crop = image.crop((max(0, x1), max(0, y1), max(x1 + 1, x2), max(y1 + 1, y2)))
        buffer = BytesIO()
        crop.save(buffer, format="PNG")
        return buffer.getvalue()

    def _lift_pose(
        self,
        detection: Sam3Detection,
        depth_est: DepthEstimation | None,
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        x1, y1, x2, y2 = detection.bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        norm_cx = round(cx / width, 4)
        norm_cy = round(cy / height, 4)
        norm_w = round(w / width, 4)
        norm_h = round(h / height, 4)
        depth_value = depth_est.median_depth if depth_est and depth_est.median_depth is not None else 1.0
        depth_span = (
            (depth_est.max_depth - depth_est.min_depth)
            if depth_est
            and depth_est.max_depth is not None
            and depth_est.min_depth is not None
            else 0.25
        )
        pose = {
            "translation": [norm_cx, depth_value, norm_cy],
            "bbox": [norm_w, depth_span, norm_h],
        }
        if depth_est:
            pose["depth_stats"] = {
                "min": depth_est.min_depth,
                "max": depth_est.max_depth,
                "median": depth_est.median_depth,
            }
        return pose

    def _placeholder_node(
        self,
        label: str,
        idx: int,
        hierarchy: HierarchyPlan,
        total: int,
    ) -> Dict[str, Any]:
        spacing = 1.0 / max(1, total)
        translation = [round(spacing * (idx + 0.5), 4), 1.0, 0.5]
        return {
            "label": label,
            "role": hierarchy.roles.get(label, "child"),
            "parent": hierarchy.parents.get(label, hierarchy.anchor),
            "confidence": 0.0,
            "bbox_pixel": [0, 0, 0, 0],
            "initial_pose": {"translation": translation, "bbox": [0.2, 0.2, 0.2]},
        }
