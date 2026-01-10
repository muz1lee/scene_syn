from __future__ import annotations

import re
from typing import Any, Dict, List

from scenethesis.services.providers import LLMProvider

SCENE_PLANNING_SYSTEM_PROMPT = """
Task Description:
You are responsible for generating a set of common objects and planning a scene based on these common objects. You will be given a list that includes all available object categories and a text prompt to describe a scene. This is a hard task, please think deeply and write down your analysis in following steps:

Step 1: Review All Categories
    a. Begin by thoroughly reviewing the categories in the provided list.
    b. Identify potential groups or clusters of objects within this list that are commonly found in similar environments (e.g., furniture, electronics, household items, etc.).
    
Step 2: Interpret Input Prompt
    a. Carefully read the input prompt. Understand the theme, primary activities, or the setting it describes, as these will guide your object selection. i.e. if the prompt gives: children playing room, then you may think of objects like tent, toy, bear, ball, chair, etc.

Step 3: Object Selection
    a. Based on the description, select at least 15 object categories from the list that match the scene.
    b. Determine the anchor object: i. Identify the anchor object among the selected objects. Consider the following factors:
        1. A large object directly on the ground (i.e. floor, table, or shelf).
        2. An object that influences where other objects are placed (i.e. a table in a dining room, and there are cups and fruits on the table).
        3. The object should logically anchor the scene and often defines the scene’s layout orientation. i.e. the sofa in a front-facing view in the scene.
    
Step 4: Object Cross-check
    a. I will give you $100 tips if you can cross-check whether objects in the scene can be found in the given category list or its relevant categories. i.e., if there is a bookshelf in your planned scene, the bookshelf should also be found in the given list, or bookcase can be found in the list if bookshelf is not covered by the category. Otherwise, re-plan the scene.

Step 5: Plan Scene with Selected Objects
    a. Based on the description and selected objects, plan the scene, keeping these aspects in mind:
        i. Functionality: Choose objects that are contextually relevant to the scene (e.g., selecting a table, chair, flower vase, and utensils for a dining room), but do not generate any wall décor objects.
        ii. Spatial Hierarchy:
            1. Please have a depth effect in the layout. For the depth effect, the scene should have some objects placed on the ground as the background, central, and in the front, resulting in a depth layout.
            2. Please have a supportive item in the layout.
        iii. Balance: Ensure a mix of large and small objects to avoid overcrowding or under-populating the scene.

Step 6: Output Format:
    Please output the result strictly in JSON format with the following keys:
    {
        "selected_objects": ["object1", "object2", ...],
        "anchor_object": "object_name",
        "upsampled_prompt": "The detailed scene description text..."
    }
""".strip()


class CoarseScenePlanner:
    def __init__(
        self,
        database_assets: List[str],
        llm_provider: LLMProvider,
        min_simple_prompt_tokens: int = 10,
    ) -> None:
        self.db_assets = database_assets
        self.db_set = {asset.lower() for asset in database_assets}
        self.llm = llm_provider
        self.min_simple_prompt_tokens = min_simple_prompt_tokens

    def run_pipeline(self, user_input: str) -> Dict[str, Any]:
        print("📋 [规划] 接收到用户描述，开始执行粗级规划管线...")
        clean_text = user_input.strip()
        if not clean_text:
            raise ValueError("用户输入为空，无法规划场景。")

        if self._is_simple_prompt(clean_text):
            print("🔀 [规划] 判定为简单描述，走自动生成分支。")
            return self._process_simple_mode(clean_text)

        print("🔀 [规划] 判定为详细描述，走受控验证分支。")
        return self._process_detailed_mode(clean_text)

    # ------------------------------------------------------------------
    # 简单模式：完全交给 LLM 推理
    # ------------------------------------------------------------------
    def _process_simple_mode(self, prompt: str) -> Dict[str, Any]:
        user_message = (
            f"[Database Assets]: {', '.join(self.db_assets)}\n"
            f"[User Prompt]: \"{prompt}\""
        )
        response = self.llm.generate_scene_plan(
            system_prompt=SCENE_PLANNING_SYSTEM_PROMPT,
            user_prompt=user_message,
        )
        required_keys = ["anchor_object", "selected_objects", "upsampled_prompt"]
        missing = [k for k in required_keys if k not in response]
        if missing:
            raise ValueError(
                f"LLM 返回缺少必需字段: {missing}. 原始响应: {response}"
            )
        return {
            "mode": "simple_generated",
            "anchor": response["anchor_object"],
            "objects": response["selected_objects"],
            "detailed_description": response["upsampled_prompt"],
        }

    # ------------------------------------------------------------------
    # 详细模式：实体抽取 + 验证 + 锚点推理
    # ------------------------------------------------------------------
    def _process_detailed_mode(self, prompt: str) -> Dict[str, Any]:
        raw_entities = self._extract_entities(prompt)
        if not raw_entities:
            raise ValueError("未能在详细描述中识别有效物体，请检查资产库或输入。")

        verified = []
        for entity in raw_entities:
            if entity in self.db_set:
                verified.append(entity)
            else:
                inferred = self._infer_category(entity)
                if inferred:
                    verified.append(inferred)
                else:
                    print(f"⚠️ [规划] 资产库中不存在 '{entity}'，已忽略。")

        if not verified:
            raise ValueError("详细描述里的物体均未通过资产库验证。")

        anchor = self._identify_anchor_logic(verified, prompt)
        return {
            "mode": "detailed_controlled",
            "anchor": anchor,
            "objects": verified,
            "detailed_description": prompt,
        }

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _is_simple_prompt(self, text: str) -> bool:
        return len(text.split()) < self.min_simple_prompt_tokens

    def _extract_entities(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        candidates = []
        for token in tokens:
            if token in self.db_set:
                candidates.append(token)
            elif token in ("macbook", "seat", "flowerpot"):
                candidates.append(token)
        return candidates

    def _infer_category(self, word: str) -> str | None:
        mapping = {
            "macbook": "laptop",
            "seat": "chair",
            "flowerpot": "plant",
        }
        mapped = mapping.get(word.lower())
        return mapped if mapped in self.db_set else None

    def _identify_anchor_logic(self, object_list: List[str], prompt: str) -> str:
        candidates = list(dict.fromkeys(object_list))
        if not candidates:
            raise ValueError("无可用对象供锚点推理。")
        anchor = ""
        try:
            anchor = self._select_anchor_with_llm(candidates, prompt)
        except Exception as err:
            print(f"⚠️ [规划] LLM 锚点选择失败，使用默认规则: {err}")

        if isinstance(anchor, str) and anchor.strip():
            lookup = {name.lower(): name for name in candidates}
            key = anchor.strip().lower()
            if key in lookup:
                return lookup[key]

        priority = ["bed", "sofa", "table", "desk", "bookshelf", "cabinet"]
        for p in priority:
            if p in candidates:
                return p
        return candidates[0]

    def _select_anchor_with_llm(self, objects: List[str], prompt: str) -> str:
        system_prompt = (
            "Determine the anchor object from the candidate list. The anchor must:"
            "\n1. Be a large object directly on the ground (floor, table, or shelf)."
            "\n2. Influence where other objects are placed (e.g., a table organizing cups and fruits)."
            "\n3. Define the scene's layout orientation (e.g., a sofa in a front-facing view)."
            "\nReturn ONLY JSON in the format {\"anchor\": \"object_name\"}"
            "\nand the anchor must come from the provided candidates."
        )
        user_prompt = (
            f"[Candidate Objects]: {objects}\n"
            f"[Scene Description]: {prompt}"
        )
        response = self.llm.generate_json(system_prompt, user_prompt)
        anchor = response.get("anchor")
        if not anchor:
            raise ValueError(f"LLM 未返回 anchor 字段: {response}")
        return anchor
