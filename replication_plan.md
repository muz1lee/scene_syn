

import torch
import torch.optim as optim

def run_scenethesis_system():
    # -------------------------------------------------
    # 0. 基础设施初始化
    # -------------------------------------------------
    db_assets = ["bed", "sofa", "desk", "chair", "table", "laptop", "plant", "lamp", "bookshelf"]
    env_maps = ["sunny.hdr", "cloudy.hdr"]
    
    # 实例化模块
    planner = CoarseScenePlanner(db_assets)
    refiner = VisualRefinementModule(db_assets, env_maps)
    physics_engine = PhysicsOptimizer(device='cuda')
    judge = SceneJudge(threshold=0.7)
    
    # 初始化可微渲染器 (Phase 3 需要用到)
    # renderer = DifferentiableRenderer(device='cuda') 

    user_prompt = "A messy bedroom"
    max_retries = 3
    current_try = 0

    # -------------------------------------------------
    # Re-planning Loop (Phase 4 闭环控制)
    # -------------------------------------------------
    while current_try < max_retries:
        print(f"\n🚀 === 尝试第 {current_try + 1} 次生成 ===")

        # --- Phase 1: Planning ---
        # 你的代码: 生成物体清单和粗略描述
        plan = planner.run_pipeline(user_prompt)
        
        # --- Phase 2: Visual Refinement ---
        # 你的代码: 生成参考图，初始 3D 布局 (此时物体可能穿模/悬空)
        initial_layout = refiner.process_layout(plan)
     
        # -------------------------------------------------
        # Phase 3: Physics Optimization (核心缺失部分)
        # -------------------------------------------------
        print("\n🔨 [Phase 3] 开始物理与姿态优化...")
        
        # 1. 将 Layout 转换为可优化的 PyTorch 参数 (T, R, s)
        # 这一步需要把 initial_layout 里的字典转成 requires_grad=True 的 Tensor
        scene_graph_params = prepare_optimization_params(initial_layout) 
        
        # 2. 设置优化器 (论文强调使用 SGD)
        optimizer = optim.SGD(scene_graph_params.parameters(), lr=0.01, momentum=0.9)
        
        # 3. 优化循环
        for i in range(200): # 迭代 200 次
            optimizer.zero_grad()
            
            # 计算总 Loss (Pose + Collision + Stability)
            # 注意：这里需要传入 renderer 来计算 Pose Loss
            loss = physics_engine(scene_graph_params, initial_layout['image_guidance'])
            
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0:
                print(f"    Iter {i}: Loss = {loss.item():.4f}")

        # 获取优化后的最终布局
        final_layout = export_layout(scene_graph_params)

        # -------------------------------------------------
        # Phase 4: Judge (裁判)
        # -------------------------------------------------
        # 渲染最终结果给裁判看 (这里需要渲染器生成一张图)
        # final_render_img = renderer.render(final_layout) 
        # 这里暂时用占位符代替
        final_render_img = "final_render_placeholder.jpg" 

        passed, report = judge.evaluate(final_render_img, initial_layout['image_guidance'])
        
        if passed:
            print("\n 成功！生成符合物理规律且视觉对齐的场景。")
            print("FINAL JSON:", json.dumps(final_layout, indent=2))
            break
        else:
            print(f"\n❌ 失败: {report['reasoning']}")
            print("🔄 触发 Re-planning...")
            current_try += 1
            # 可以在这里修改 user_prompt 或增加随机种子来获得不同结果

    if current_try >= max_retries:
        print("达到最大重试次数，生成失败。")

# 辅助函数: 模拟参数转换
class SceneGraphModel(torch.nn.Module):
    def __init__(self, layout_dict):
        super().__init__()
        # 将 T, R, s 变成可学习参数
        self.objects = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(3)) for _ in layout_dict['scene_layout']
        ])
    def forward(self):
        return self.objects

def prepare_optimization_params(layout):
    return SceneGraphModel(layout)

def export_layout(model):
    return {"optimized": True, "data": "..."} 

if __name__ == "__main__":
    run_scenethesis_system()

1. Coarse Scene Planning
import json
from typing import List, Dict, Any

# ==========================================
# 1. 论文 Section 7.1: 核心提示词模板
# ==========================================
SCENE_PLANNING_SYSTEM_PROMPT = Coarse_Scene_Planning_Instruction_Prompts
class CoarseScenePlanner:
    def __init__(self, database_assets: List[str]):
        self.db_assets = database_assets
        self.db_set = set(asset.lower() for asset in database_assets) # 优化查询速度

    def run_pipeline(self, user_input: str) -> Dict[str, Any]:
        """
        主入口函数
        """
        # 1. 分支处理 (Branching)
        if self._is_simple_prompt(user_input):
            print(f"--- 检测到简单 Prompt: '{user_input}' -> 进入生成模式 ---")
            return self._process_simple_mode(user_input)
        else:
            print(f"--- 检测到详细 Prompt -> 进入验证模式 ---")
            return self._process_detailed_mode(user_input)

    # ==========================================
    # 路径 A: Flexible Scene Generation (简单模式)
    # ==========================================
    def _process_simple_mode(self, simple_prompt: str) -> Dict[str, Any]:
        """
        利用 Section 7.1 的 Prompt，让 LLM 自己推理、选品、定锚点
        """
        # 构造输入消息
        user_message = f"""
        [Database Assets]: {", ".join(self.db_assets)}
        [User Prompt]: "{simple_prompt}"
        """

        # 模拟调用 LLM (GPT-4o)
        # 这里的关键是 System Prompt 包含了 Section 7.1 的所有约束
        response_json = self._mock_llm_call(
            system_prompt=SCENE_PLANNING_SYSTEM_PROMPT, 
            user_message=user_message
        )
        
        # 简单模式下，LLM 的输出直接就是最终结果，因为 Prompt 里已经要求它定好 anchor 和 spatial relations 了
        return {
            "mode": "simple_generated",
            "anchor": response_json["anchor_object"],
            "objects": response_json["selected_objects"],
            "detailed_description": response_json["upsampled_prompt"]
        }

    # ==========================================
    # 路径 B: Controllable Scene Generation (专家/详细模式)
    # ==========================================
    def _process_detailed_mode(self, detailed_prompt: str) -> Dict[str, Any]:
        """
        原文逻辑：Checks for presence -> Infers categories -> Skips up-sampling -> Identifies anchor
        """
        # 1. 提取实体 (NER)
        raw_objects = self._extract_entities(detailed_prompt)
        
        # 2. 查库验证 (Review & Check Presence)
        valid_objects = []
        for obj in raw_objects:
            # 尝试直接匹配
            if obj.lower() in self.db_set:
                valid_objects.append(obj.lower())
            else:
                # 尝试推断 (Infer relevant categories)
                # 例如：用户写 "Macbook", 库里有 "Laptop"
                inferred = self._infer_category(obj)
                if inferred:
                    valid_objects.append(inferred)
                else:
                    print(f"Warning: 忽略未知物体 '{obj}'")
        
        if not valid_objects:
            raise ValueError("无法在详细描述中匹配到任何数据库资产")

        # 3. 确定锚点 (Identifies an anchor object)
        # 详细模式下，我们有了物体列表，但需要找出谁是老大
        # 再次调用一个小型的 LLM 任务，遵循 Holodeck 策略
        anchor = self._identify_anchor_logic(valid_objects)

        # 4. 跳过上采样 (Skip up-sampling)，直接使用用户输入作为描述
        # 但我们需要建立层级关系 (Coarse spatial hierarchy)
        return {
            "mode": "detailed_controlled",
            "anchor": anchor,
            "objects": valid_objects,
            "detailed_description": detailed_prompt # 原封不动保留用户的详细描述
        }

    # ==========================================
    # 辅助与模拟方法 (Helpers)
    # ==========================================
    def _is_simple_prompt(self, text: str) -> bool:
        # 简单判定：长度短，或者缺少介词方位词(on, next to, behind)
        return len(text.split()) < 10

    def _extract_entities(self, text: str) -> List[str]:
        # 实际应该调用 NLP 模型，这里模拟提取
        # 假设输入: "A chair next to a table" -> ["chair", "table"]
        # 这里仅作演示
        import re
        words = re.findall(r'\w+', text.lower())
        return [w for w in words if w in self.db_set or w in ["macbook", "sofa"]] 

    def _infer_category(self, obj: str) -> str:
        # 模拟语义映射
        mapping = {"macbook": "laptop", "seat": "chair", "flowerpot": "plant"}
        val = mapping.get(obj.lower())
        return val if val in self.db_set else None

    def _identify_anchor_logic(self, object_list: List[str]) -> str:
        """
        对应原文: "occupying the highest spatial hierarchy apart from the ground"
        """
        # 可以用 LLM，也可以用硬规则。这里模拟 LLM 决策。
        priority = ["bed", "sofa", "table", "desk", "bookshelf", "cabinet"]
        for p in priority:
            if p in object_list:
                return p
        return object_list[0] # Fallback

    def _mock_llm_call(self, system_prompt, user_message):
        """
        模拟 GPT-4o 的 JSON 返回
        """
        # 假设 Simple Prompt 是 "A messy bedroom"
        return {
            "anchor_object": "bed",
            "selected_objects": ["bed", "desk", "chair", "laptop", "books", "clothes", "lamp"],
            "upsampled_prompt": "The bed is the central anchor against the back wall. A desk is placed next to the bed..."
        }

"""
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
        i. Functionality: Choose objects that are contextually relevant to the scene (e.g., selecting a table, chair, flower vase, and utensils for a dining room), but do not generate any wall d ́ecor objects.
        ii. Spatial Hierarchy:
            1. Please have a depth effect in the layout. For the depth effect, the scene should have some objects placed on the ground as the background, central, and in the front, resulting in a depth layout. i.e. the sofa and bookshelf are the background of the table and chair set in the living room.
            2. Please have a supportive item in the layout. i.e. the shoes, bag, and hat are in the display shelf in a clothes store, where the display shelf is a supportive item.
        iii. Balance: Ensure a mix of large and small objects to avoid overcrowding or under-populating the scene. i.e. taking the table as the center, there are flower vases, fruits, and cups on the table, and chairs are on the sides.

Step 6: Output Format:
    a. Save the selected objects as a json file follow the output format:
        Anchor object:
        Other common objects:
    b. Save scene planning as txt file.
"""
2. Layout Visual Refinement (视觉布局细化)

Step 1: Image Guidance (以图引路)
- 输入： 第一部分生成的 detailed_description 
- 动作： 调用文生图模型  GPT-4o (DALL-E 3) 。
- 目的： 利用生成模型在大规模数据集学到的“物体共现”和“空间关系”，生成一张看起来很合理的 2D 参考图。这张图就是后续 3D 布局的蓝图。

Step 2: Scene Graph Generation (双轨场景图构建)
- 逻辑轨 (GPT-4o): 定义场景树结构（谁是地基 Ground，谁是父节点 Parent，谁是子节点 Child）。注意： 这一步不涉及像素，只涉及物体间的层级逻辑。
- 几何轨 (Grounded-SAM + Depth Pro): 处理像素。
  - Grounded-SAM: 分割 Mask + 裁剪图片 (Cropped Images)。
  - Depth Pro: 估算 Metric Depth。
  - Lifting: 2D 像素 -> 3D 点云 -> 3D Bounding Box。
- 合并: 将几何坐标填入逻辑树中。

Step 3: Asset Retrieval (资产检索)
- 动作 A (物体检索): 使用 CLIP (ViT-L/14)。
  - 输入：Grounded-SAM 扣出来的物体切片图。
  - 数据库：Objaverse。
  - 原理：对比“切片图”和“3D资产缩略图”的语义特征，找最像的。
- 动作 B (环境检索 - 新增): 使用 GPT-4o。
  - 为什么需要这个？ 论文明确说 Scenethesis 只生成地面物体 (objects on the ground)。墙壁、窗户、阳光、海滩背景等，不生成 3D 模型，而是直接从数据库里找一张匹配的 HDRI 环境贴图。
  - 输入：Upsampled Prompt (文本)。
  - 输出：Environment Map (e.g., "sunny_beach.hdr")。

import json
import random
from typing import List, Dict, Any

class VisualRefinementModule:
    def __init__(self, asset_database: List[str], env_map_database: List[str]):
        self.asset_db = asset_database
        self.env_db = env_map_database
        print("--- [系统初始化] Loading Models ---")
        print("  |-- Visual Gen: GPT-4o (DALL-E 3 Integration)")
        print("  |-- Segmentation: Grounded-SAM")
        print("  |-- Depth Estimation: Depth Pro")
        print("  |-- Retrieval: CLIP (ViT-L/14) + GPT-4o (EnvMap)")

    def process_layout(self, coarse_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        对应论文 6.2.2 流程
        """
        detailed_prompt = coarse_plan["detailed_description"]
        print(f"\n[Phase 2 Start] Visual Refinement for: '{detailed_prompt[:30]}...'")

        # Step 1: Image Guidance (GPT-4o/DALL-E 3)
        # "GPT-4o generates an image to serve as fine-grained layout guidance"
        generated_image = self._generate_image_gpt4o(detailed_prompt)

        # Step 2: Scene Graph Construction (Hybrid: Logic + Geometry)
        # "GPT-4o ... define ground/parent/child" + "Grounded-SAM ... segments"
        scene_graph = self._construct_scene_graph(
            image=generated_image,
            object_list=coarse_plan["objects"],
            anchor_name=coarse_plan["anchor"]
        )

        # Step 3: Asset Retrieval (CLIP)
        # "CLIP ... retrieve 3D assets that align with image guidance"
        final_objects = self._retrieve_3d_assets_clip(scene_graph)

        # Step 4: Environment Map Selection (GPT-4o)
        # "GPT-4o is further utilized to select the most relevant environment map"
        selected_env_map = self._select_env_map_gpt4o(detailed_prompt)

        return {
            "image_guidance": generated_image,
            "scene_layout": final_objects,
            "environment_map": selected_env_map
        }

    # =========================================================================
    # 核心子函数实现
    # =========================================================================

    def _generate_image_gpt4o(self, prompt: str) -> str:
        # 模拟调用 DALL-E 3
        print(f"  [1. Image Gen] Generating guidance image via GPT-4o...")
        return "fake_image_tensor_640x640"

    def _construct_scene_graph(self, image, object_list, anchor_name):
        print(f"  [2. Scene Graph] Constructing 3D Spatial Graph...")
        
        # A. 逻辑层级 (GPT-4o)
        # 论文提到 GPT-4o 负责定义 Ground -> Parent -> Child 关系
        # 这一步是为了防止几何计算出错（比如把枕头算到地板上）
        hierarchy = self._gpt4o_define_hierarchy(object_list, anchor_name)
        
        # B. 几何感知 (Grounded-SAM + Depth Pro)
        # 论文: "Segments each object... projected into 3D space using Depth Pro"
        nodes = []
        for obj_name in object_list:
            # 1. Segment (SAM) -> 得到 Mask 和 Crop
            mask, cropped_img = self._sam_segment(image, obj_name)
            
            # 2. Depth Project (Depth Pro) -> 得到 3D 坐标
            # Initial positioning within a spatial relationship graph
            pos_3d, bbox_3d = self._depth_pro_lift(mask)
            
            nodes.append({
                "label": obj_name,
                "role": hierarchy.get(obj_name, "child"), # 从 GPT-4o 获取角色
                "parent": hierarchy.get(f"{obj_name}_parent", "ground"),
                "initial_pose": {"pos": pos_3d, "bbox": bbox_3d},
                "visual_crop": cropped_img # 用于 CLIP 检索
            })
            
        return nodes

    def _retrieve_3d_assets_clip(self, scene_graph):
        print(f"  [3. Asset Retrieval] Matching assets using CLIP (ViT-L/14)...")
        for node in scene_graph:
            # 模拟 CLIP 向量搜索
            # Query: node['visual_crop'] (图像特征) + node['label'] (语义特征)
            # Key: Database Assets
            best_match_id = f"{node['label']}_premium_v1.glb"
            node["asset_id"] = best_match_id
            print(f"    - Matched '{node['label']}' -> {best_match_id}")
        return scene_graph

    def _select_env_map_gpt4o(self, prompt):
        print(f"  [4. Env Map] Selecting HDR Environment Map via GPT-4o...")
        # 逻辑：把 prompt 给 GPT-4o，让它从 env_db 里选一个
        # 比如 prompt 是 "sunset beach"，GPT-4o 会选 "beach_sunset_4k.hdr"
        selected = random.choice(self.env_db)
        print(f"    - Selected Context: {selected}")
        return selected

    # --- 模拟底层模型 ---
    def _gpt4o_define_hierarchy(self, objects, anchor):
        # 简单模拟 GPT-4o 返回的层级关系
        # 假设 Anchor 是 Parent，其他都是 Child
        h = {anchor: "parent"}
        for o in objects:
            if o != anchor:
                h[o] = "child"
                h[f"{o}_parent"] = anchor # 记录谁是谁的父节点
        return h

    def _sam_segment(self, image, label):
        return "mask_array", "cropped_image_tensor"

    def _depth_pro_lift(self, mask):
        # 模拟从 Depth Pro 算出的坐标
        return [random.uniform(-2,2), 0, random.uniform(-2,2)], [1, 1, 1]



3. Physics-aware Optimization

在上一阶段（Visual Refinement），系统虽然给出了 3D 布局，但它只是基于 2D 图片反推的，存在严重的**“幻觉”和误差**：
- 遮挡误差： 图片里桌子被椅子挡住了，反推出来的桌子可能只有一半。
- 穿模与悬空： 物体可能会插进墙里，或者浮在半空中。
- 形状差异： 数据库里找出来的模型和图片里的模型长得不完全一样（比如图片是圆角桌，模型是直角桌）。
Physics-aware Optimization 的作用就是：通过数学优化，把这些模型“推”到正确的位置，既要对齐参考图，又要符合物理规律。

---
核心步骤详解
这个过程不是一次性的计算，而是一个迭代优化循环 (Iterative Loop)。系统定义了一个总能量函数（Total Loss），通过不断微调物体的 5-DoF 参数（缩放 $s$、旋转 $R$、平移 $T$），让能量降到最低。
步骤 1：姿态对齐 (Pose Alignment) —— “看着要像”
这步是为了解决“模型摆放位置不对”的问题。
- 技术核心： Dense Semantic Correspondence (稠密语义匹配)，使用 RoMa 模型。
- 为什么不用像素对比？ 因为检索到的 3D 资产纹理和生成图不一样，直接比像素（MSE Loss）会彻底失败。RoMa 比较的是“语义特征”，比如它知道“这是桌腿”，不管桌腿是黑是白。
- 具体操作：
  1. 渲染： 把当前的 3D 场景渲染成 2D 图像 $I$。
  2. 匹配： 拿这张图 $I$ 和 Phase 2 生成的参考图 $\tilde{I}$ 进行特征点匹配。系统会筛选出置信度 $\tau \ge 0.6$ 的 $m=100$ 对关键点。
  3. 优化 ($L_{pose}$): 调整 3D 模型的参数，让这 100 个点在 3D 空间和 2D 投影空间上的距离最小化。

步骤 2：物理合理性 (Physical Plausibility) —— “物理要真”
这步是为了解决“穿模”和“悬空”问题。Scenethesis 抛弃了粗糙的 Bounding Box，改用 SDF (符号距离场) 来实现高精度碰撞检测。
系统会在物体表面均匀采样 $n=400$ 个点来探测环境。
A. 防碰撞：平移推离 ($L_{translation}$)
- 现象： 如果物体表面的点检测到 SDF 值 $d < 0$，说明撞进别人体内了。
- 动作： 计算一个推力向量 $u$（从碰撞点指向物体质心）。系统会把物体沿着 $u$ 方向推开，推开的距离就是穿模深度 $|d|$。
- 直观理解： 就像两个磁铁同极相斥，离得越近（穿模越深），推开的力越大。
B. 防碰撞：挤压缩小 ($L_{scale}$)
- 现象： 如果一个物体被从两个不同的方向同时挤压（比如书被夹在两层书架之间，或者被左右两本书夹住），光靠平移是推不出来的。
- 判断逻辑： 系统检测碰撞点的方向，如果聚类数 $N_{cluster} > 1$（即来自不同方向的碰撞），就判定为“被夹击”。
- 动作： 减小物体的缩放值 $s$，让它变小一点，直到能塞进去。
C. 稳定性：重力吸附 ($L_{stability}$)
- 现象： 物体不能悬浮。
- 动作： 采样物体底部的点 $V^B$。系统强制要求这些点相对于“父节点表面”（比如桌面）的 SDF 值必须为 0。
- 数学实现： 使用 Loss $= \sum (1 - \exp(-d^2))$。如果不贴合，Loss 就大；贴合了，Loss 就是 0。这相当于给物体施加了一个“数字重力”。

---
关键工程细节 (复现必读)
根据 Section 6.3 和 Method Overview，复现时必须遵守以下“军规”：
1. 两阶段优化策略 (Two-stage Optimization):
  - Phase A: 先只跑 Pose Alignment。让物体先飞到大概正确的位置。
  - Phase B: 再加入 Physical Constraints。在视觉对齐的基础上，把物体“按”在地上，并推开重叠部分。
  - 原因： 如果一开始就开物理碰撞，物体可能会被弹飞，导致视觉匹配失败。
2. 优化器选择：SGD (Stochastic Gradient Descent)
  - 不要用 Adam！ 论文明确指出 Adam 的动量机制 (Adaptive Momentum) 会导致优化不稳定（因为物理碰撞的梯度是突变的），SGD 效果更稳。
3. 技术栈：
  - 底层基于 PyTorch3D 实现（因为需要可微渲染和 3D 操作）。
  - 在 A100 GPU 上运行。
总结流程图
1. 输入： 粗糙的 Scene Graph + 2D 参考图。
2. 预处理： 将 3D 资产转换为 SDF 场，采样表面点 ($n=400$)。
3. 循环迭代 (SGD):
  - 计算视觉 Loss ($L_{pose}$): RoMa 找点 -> 算距离 -> 梯度回传。
  - 计算物理 Loss ($L_{phy}$): 查 SDF -> 算穿模深度 -> 施加推力/缩力/重力。
  - 更新参数: 修改 $T$ (位移), $R$ (旋转), $s$ (大小)。
4. 裁判 (Judge): GPT-4o 最终看一眼（Section 3.4），合格则输出。
5. 输出： 一个视觉对齐、无穿模、接触紧实的完美 3D 场景。

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsOptimizer(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # 超参数设置
        self.n_samples = 400         # n surface points
        self.m_correspondence = 100  # m matching points
        self.tau = 0.6               # confidence threshold
        
        # Loss 权重系数 (需要根据实验调整)
        self.lambda_p = 1.0   # Pose
        self.lambda_cT = 10.0 # Collision Translation
        self.lambda_cS = 10.0 # Collision Scale
        self.lambda_s = 5.0   # Stability

    def forward(self, scene_graph, ref_image_data):
        """
        对应论文 Eq (9): L = λp*Lpose + λcT*Ltranslation + λcS*Lscale + λs*Lstability
        """
        # 1. 姿态对齐损失 (Eq 4, 5)
        loss_pose = self._compute_pose_loss(scene_graph, ref_image_data)
        
        # 2. 物理碰撞与稳定性损失
        loss_trans, loss_scale, loss_stab = 0.0, 0.0, 0.0
        
        for obj in scene_graph.objects:
            # 采样物体表面点 n=400
            surface_points = self._sample_surface(obj, self.n_samples)
            # 计算这些点在场景中的 SDF 值 (d)
            # d < 0 表示碰撞 (Inside), d > 0 表示安全 (Outside)
            sdf_values, gradients = self._query_scene_sdf(surface_points, scene_graph)
            
            # --- Eq 6: Translation Collision Loss ---
            loss_trans += self._compute_translation_loss(obj, sdf_values, gradients)
            
            # --- Eq 7: Scale Collision Loss ---
            loss_scale += self._compute_scale_loss(obj, sdf_values, gradients)
            
            # --- Eq 8: Stability Loss ---
            loss_stab += self._compute_stability_loss(obj, scene_graph)

        # 总损失
        total_loss = (self.lambda_p * loss_pose + 
                      self.lambda_cT * loss_trans + 
                      self.lambda_cS * loss_scale + 
                      self.lambda_s * loss_stab)
        
        return total_loss

    # =========================================================
    # 具体的公式实现 (Mapping Formulas to Code)
    # =========================================================

    def _compute_translation_loss(self, obj, sdf, gradients):
        """
        对应 L_translation
        公式: || f(T, |d|, u) - T ||^2
        """
        # 1. 筛选碰撞点 V- (SDF < 0)
        mask_collision = sdf < 0
        if not mask_collision.any():
            return 0.0
            
        d_collided = sdf[mask_collision]    # 负的 SDF 值
        grad_collided = gradients[mask_collision] # 碰撞点的梯度方向
        
        # 2. 定义方向 u (从碰撞点指向物体质心)
        # 简化实现：通常 SDF 的梯度方向就是推开碰撞的最快方向
        u = -grad_collided 
        
        # 3. 计算偏移量 |d| * u
        # f(T,...) - T 其实就是计算需要移动的向量 delta
        # Eq 6: |d| = max(0, -d(vi))，即绝对值
        push_vector = u * torch.abs(d_collided).unsqueeze(1)
        
        # 4. L2 Loss
        return torch.sum(push_vector ** 2)

    def _compute_scale_loss(self, obj, sdf, gradients):
        """
        对应 L_scale
        逻辑: 如果物体被两边夹住 (N_cluster > 1)，就缩小它
        """
        mask_collision = sdf < 0
        if mask_collision.sum() < 2:
            return 0.0

        # 1. 聚类检测 (N_cluster)
        # 简单判别：如果碰撞点的梯度方向差异很大(比如点积 < -0.5)，说明来自不同方向的夹击
        grad_collided = gradients[mask_collision]
        # 这里用简化的余弦相似度模拟聚类检测
        # 实际代码可能需要 K-Means，这里为了可导性用方向方差代替
        grad_variance = torch.var(grad_collided, dim=0).sum()
        
        is_squeezed = grad_variance > 0.5 # 阈值判定 N_cluster > 1
        
        if is_squeezed:
            # 2. 计算目标缩放 (Target Scale)
            # Eq 7: g(|d|, u) - s
            # 这里的直觉是：如果碰撞深，就要大幅减小 scale
            current_scale = obj.scale
            # 简化的惩罚：碰撞越深，Loss 越大，梯度会指向缩小 scale
            d_abs = torch.abs(sdf[mask_collision])
            return torch.sum(d_abs ** 2) 
        return 0.0

    def _compute_stability_loss(self, obj, scene_graph):
        """
        对应 L_stability
        公式: Sum(1 - exp(-d^2))
        """
        # 1. 采样底部点 V^B
        bottom_points = self._sample_bottom_points(obj)
        
        # 2. 查询父节点表面的 SDF (Parent Surface SDF)
        # 例如：杯子底部点相对于桌面的 SDF
        parent_sdf, _ = self._query_parent_sdf(bottom_points, obj, scene_graph)
        
        # 3. 计算 Loss
        # 当 d=0 (紧贴) 时，exp(0)=1，Loss=0
        # 当 d 大 (悬空) 时，exp(-d^2)->0，Loss->1
        return torch.sum(1.0 - torch.exp(-parent_sdf ** 2))

    def _compute_pose_loss(self, scene_graph, ref_img):
        """
        对应 L_pose = λ2d*L2d + λ3d*L3d
        使用 RoMa 进行匹配
        """
        # 渲染当前物体得到 rendered_img
        # 运行 RoMa(rendered_img, ref_img) 得到 matches
        # 筛选 confidence > 0.6 的点
        # 计算 MSE Loss
        pass # 需要可微渲染器支持
        
    # --- Helper Placeholders ---
    def _sample_surface(self, obj, n): return torch.randn(n, 3)
    def _sample_bottom_points(self, obj): return torch.randn(50, 3)
    def _query_scene_sdf(self, points, graph): return torch.randn(len(points)), torch.randn(len(points), 3)
    def _query_parent_sdf(self, points, obj, graph): return torch.randn(len(points)), None


3.1 缺失模块：Scene Judge (场景裁判)
依据： 论文 Section 3.4 明确提到："After iteratively optimizing... a scene judge powered by GPT-4o evaluates the spatial alignment... comparing the generated scene [with] image guidance."
作用： 优化结束后，不能盲目输出。需要 GPT-4o 来看一眼最终结果，判断是否“对味”。如果评分太低，可能需要重新做。
代码实现 (SceneJudge 类)：
Python
class SceneJudge:
    def __init__(self):# 裁判是 GPT-4o
        print("Initializing GPT-4o Scene Judge...")

    def evaluate(self, generated_3d_snapshot, guidance_image):"""
        对应 Section 3.4: "design three metrics"
        1. Object Category Accuracy
        2. Spatial Relationship Consistency
        3. Visual Similarity
        """
        prompt = """
        You are a Scene Judge. Compare the 'Generated 3D Scene View' with the 'Guidance Image'.
        Evaluate on 3 criteria:
        1. Do the objects match? (Accuracy)
        2. Is the layout/positioning consistent? (Spatial Coherence)
        3. Is there physical inter-penetration? (Physical Plausibility)
        
        Output a Score (0-10) and a Decision (Pass/Refine).
        """# 调用 GPT-4o Vision API 发送两张图# return response_json
        print("  [Judge] GPT-4o is reviewing the final layout...")
        
        return {"score": 8.5, "decision": "PASS"}

---
3.2 基础设施：SDF 数据预处理 (The Data Prep)
依据： 你的 PhysicsOptimizer 里使用了 sdf[mask_collision]。但在代码里这只是个占位符。
现实问题： 网上下载的 .obj 或 .glb 模型是网格 (Mesh)，不是 SDF。你无法直接查询 Mesh 的负值。
你需要补充的内容： 在加载资产库时，必须有一个预计算 (Baking) 步骤。
代码逻辑补充：
Python
import pytorch3d.structures
from pytorch3d.ops import sample_points_from_meshes

def preprocess_asset_to_sdf(mesh_path):"""
    "replace 3DBB... with Signed Distance Fields (SDFs)"
    在加载模型时运行一次
    """# 1. 加载 Mesh
    mesh = load_objs_as_meshes([mesh_path])
    
    # 2. 转换为点云并计算 SDF (简化版：使用点云近似)# 真正的 SDF 需要计算空间中每个点到 Mesh 表面的距离# 工业界通常使用 'mesh_to_sdf' 库或 DeepSDF 预训练网络
    print(f"Baking SDF for {mesh_path}...")
    
    # 返回一个可查询的 SDF 函数或 3D Gridreturn sdf_grid
- 建议： 复现时，去 GitHub 找一个轻量级的 mesh-to-sdf 库，把每个家具模型预处理成一个 $64 \times 64 \times 64$ 的 SDF 矩阵存起来。

---
3.3 基础设施：可微渲染器 (Differentiable Renderer)
依据： 论文 Section 3.3.1 和 Section 6.2.3 提到 $$L_{pose}$$ 需要 "backpropagating gradients"。
现实问题： 普通的渲染（如 Blender）不可导，梯度传不回来。必须使用 PyTorch3D 的 MeshRenderer。
你需要补充的内容： 一个渲染器包装类，用于在优化循环中实时生成图像。
代码逻辑补充：

# "optimization implementation is based on pytorch3D"
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, SoftPhongShader, 
    RasterizationSettings, PerspectiveCameras
)

class DifferentiableRenderer:def __init__(self, device):# 初始化相机和光照
        self.cameras = PerspectiveCameras(device=device)
        self.raster_settings = RasterizationSettings(image_size=512)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
            shader=SoftPhongShader(device=device)
        )

    def render_scene(self, scene_graph):"""
        输入: 包含当前 T, R, s 参数的场景图
        输出: 带有梯度的 2D 图像张量 (Tensor)
        """# 将 scene_graph 中的 mesh 组合成一个 Batch# 应用当前的 transforms# images = self.renderer(meshes_world)return images



4.  Pose Alignment Evaluation 

import json

# ==============================================================================
# Section 3.4: 空间一致性裁判的核心 Prompt
# ==============================================================================
SPATIAL_ALIGNMENT_PROMPT = """
This task involves evaluating the pose alignment between two images in a pair. 
One image serves as the image guidance (GT), while the other is a generated image (Render). Your objective is to measure the pose alignment of the generated image relative to the GT image.

Follow these steps for evaluation:

1. Review Objects in the GT Image: Examine locations, sizes, and orientations. Understand spatial relationships (on top of, inside, under).

2. Evaluate pose alignment based on 3 aspects:
   • Location and Size Similarity (0-1): Compare placement. (e.g. 1.0 = perfect center match, 0.1 = misplaced on ground).
   • Orientation Similarity (0-1): Check for tilts or rotations. (e.g. 1.0 = aligned perspective).
   • Overall Layout Similarity (0-1): Assess visual coherence and hierarchical structure.

3. Exclusions: Do not consider style, appearance, object shape, or texture. Focus solely on POSE ALIGNMENT.

4. Output Format: 
   Save the evaluated scores as a JSON file strictly following this structure:
   {
     "location_size_score": 0.0,
     "orientation_score": 0.0,
     "overall_layout_score": 0.0,
     "reasoning": "Brief explanation..."
   }
"""

class SceneJudge:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.system_prompt = SPATIAL_ALIGNMENT_PROMPT

    def evaluate(self, render_image_b64, guidance_image_b64):
        """
        执行裁判逻辑: Render vs Guidance
        """
        print("--- [Scene Judge] Assessing Spatial Alignment (0-1 Scale) ---")
        
        # 1. 调用 GPT-4o Vision (伪代码)
        # response = call_llm(self.system_prompt, [render_image_b64, guidance_image_b64])
        
        # --- 模拟 LLM 返回的结果 ---
        result = {
            "location_size_score": 0.85,
            "orientation_score": 0.9,
            "overall_layout_score": 0.8,
            "reasoning": "The sofa is correctly centered, but the chair is slightly rotated compared to the GT."
        }
        
        # 2. 决策逻辑 (Pass or Replan)
        # 论文 3.4: "If any metric falls below a predefined threshold, trigger re-planning"
        scores = [
            result["location_size_score"], 
            result["orientation_score"], 
            result["overall_layout_score"]
        ]
        
        min_score = min(scores)
        avg_score = sum(scores) / len(scores)
        
        print(f"    Scores: Loc={scores[0]}, Ori={scores[1]}, Layout={scores[2]}")
        
        if min_score < self.threshold:
            print(f"!!! FAIL: Metric below threshold {self.threshold}. Triggering RE-PLANNING.")
            return False, result # 触发重做
        else:
            print(f"*** PASS: Scene Approved. (Avg: {avg_score:.2f}) ***")
            return True, result

This task involves evaluating the pose alignment between two images in a pair. One image serves as the image guidance (GT), while the other is a generated image. Your objective is to measure the pose alignment of the generated image relative to the GT image. Follow these steps for evaluation:
1. Review Objects in the GT Image: Examine all objects in the GT image, focusing on their locations, sizes, and orientations. Understand the spatial relationships among objects, such as on top of, inside, under, etc.
2. Evaluate pose alignment: Assess the similarity between the generated image and the GT image based on the following three aspects:
    • Location and Size Similarity: Compare the location and size of objects in the generated image with those in the GT image. Assign a similarity score between 0 and 1, where 1 indicates the highest similarity. For example:– If an apple in the GT image is placed at the center of a table, and in the generated image it is placed on the left side of the table, the similarity might be moderate (e.g., 0.5).– If the apple is misplaced (e.g., on the ground or missing entirely), the similarity would be very low (e.g., 0.1).
    • Orientation Similarity: Examine the orientation of each object in the generated image compared to the GT image. Pay close attention to details, noting any deviations such as slight tilts (e.g., right/left, up/down) or rotations that create different perspectives. Assign a score from 0 to 1, where 1 indicates perfect alignment and 0 indicates a significant mismatch (e.g., opposite orientation).
    • Overall Layout Similarity: Assess the overall visual coherence of the generated image compared to the GT image, including spatial relationships and hierarchical structure. Assign a similarity score between 0 and 1, where 1 represents a perfect match. For instance:– A perfect match occurs when the generated image maintains the same spatial relationships, relative locations, sizes, and orientations as the GT image (e.g., an apple placed at the center of a table in both images).– Small deviations in placement or orientation are acceptable but should result in a lower score.
3. Exclusions: Do not consider style, appearance, object shape, or texture in your evaluation. Focus solely on pose
alignment.
4. Output Format: Clearly document your similarity scores for each aspect (Location and Size Similarity, Orientation Similarity, and Overall Layout Similarity) following the format: location and size similarity score is {}, orientation similarity score is {}, and overall layout similarity score is {}. Please save the evaluated scores as a json file.