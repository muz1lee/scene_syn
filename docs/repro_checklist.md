# Scenethesis 复现检查清单

> 依据 `replication_plan.md` 梳理复现阶段、依赖与交付物。勾选状态会在实现后逐步更新。

## Phase 0 · 基础设施与依赖
- [x] 统一大模型接口：Gemini Provider 接入真实 API，支持 gemini-3-flash / gemini-2.5-pro 可配置。
- [ ] 资产与环境库：梳理家具/装饰物 `db_assets`，HDR 环境贴图清单，补充版本管理。
- [ ] 预计算 SDF：为资产库生成 `64^3` 体素或函数式 SDF，落地 mesh-to-sdf 管线与缓存格式。
- [ ] 可微渲染器：集成 PyTorch3D，封装用于 Pose Loss 的渲染/相机配置。
- [ ] 评估与日志：统一日志、指标收集（score、loss）、失败样本缓存。

## Phase 1 · Coarse Scene Planning
- [x] System Prompt 模板（Section 7.1）落地到源码，可热更新。
- [x] Prompt 分支：自动识别简单/复杂描述，执行对应推理路线。
- [x] 简单模式：Gemini 3 Flash 直接完成 anchor + object list + upsampled prompt，输出 JSON。
- [x] 详细模式：实体抽取、类目验证、语义映射、锚点推理、层级关系草图。
- [x] 数据契约：`plan` 结构包含 mode/anchor/objects/detailed_description，供 Phase 2 消费。

## Phase 2 · Visual Refinement
- [ ] Guidance 图生成：Gemini 或视觉模型生成 640x640 参考图。
- [ ] 场景图构建：Grounded-SAM + Depth Pro 推理链封装，输出节点 pose/bbox。
- [ ] 资产检索：CLIP + 数据库 embedding 检索，记录命中资产 ID。
- [ ] 环境贴图选择：Gemini 3 Flash 基于描述选择 HDR。
- [ ] 输出格式：`image_guidance`、`scene_layout`、`environment_map`。

## Phase 3 · Physics Optimization
- [ ] Scene Graph 参数化：把 layout dict 映射到可训练的 `nn.Module`。
- [ ] Loss 组合：实现 Pose/Translation/Scale/Stability 四项，可配置权重。
- [ ] SDF 查询/采样：封装 surface sampling 与 parent/scene SDF 查询接口。
- [ ] 优化循环：SGD + 学习率/动量/迭代次数配置。
- [ ] Layout 导出：将优化结果写回 JSON 序列化结构。

## Phase 4 · Scene Judge + Loop
- [ ] Judge 提示：Gemini 3 Flash Vision 输入 guidance + 渲染图，输出评分/通过与否。
- [ ] 阈值策略：score < τ 触发 re-plan，记录失败原因。
- [ ] Re-planning Loop：主循环控制重试次数、随机扰动或 prompt 调整。

## 支撑模块
- [ ] 任务文件管理：`docs/tasks/*.md` 记录 Analysis/Plan/Progress/Review。
- [ ] 配置管理：集中管理数据库路径、API key、设备。
- [x] CLI/入口：`run_scenethesis_system` 主函数，串联四大模块并输出日志。
