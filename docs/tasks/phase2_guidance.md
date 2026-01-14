# 上下文
文件名：phase2_guidance.md
创建于：2024-06-13
创建者：Codex
关联协议：RIPER-5 + Multidimensional + Agent Protocol

# 任务描述
docs/repro_checklist.md 进入 Phase 2，当前聚焦 Guidance 图生成：基于 Phase 1 planner 输出调用 Gemini 3 Pro Image 生成 640x640 参考图并写入 `/Users/knowin-wenqian/scene_gen/scenethesis/output`，同时考虑未来切换 Qwen 文生图的扩展性。

# 项目概述
SceneThesis：LLM 驱动的四阶段 3D 场景生成系统。现阶段 Phase 1 已完成，需要把 Refiner 阶段接入真实图像生成、输出物体初始布局，为物理优化与裁判阶段提供输入。

---

# 分析
- Phase 2 模块 `scenethesis/modules/refiner.py` 仍是占位实现，没有调用任何图像生成接口，也未写入磁盘输出。
- 仅存在文本 LLM Provider（Gemini JSON 输出），缺少图像 Provider 抽象，无法在不同图像模型间切换。
- 主流程 `scenethesis/main.py` 只运行 Planner，需要串联 Refiner 结果与输出存储。
- 配置与文档缺少 Phase 2 相关字段（例如 image provider/模型、输出目录、环境贴图列表），不利于扩展。

# 提议的解决方案
- 在 services 层新增 Image Provider 抽象与 Gemini 3 Pro Image 实现，统一返回二进制图像，便于未来扩展 Qwen。
- 重构 Refiner：根据 planner 输出构造图像 prompt，调用 provider 生成 640x640 PNG，并写入 `scenethesis/output`，同时生成初始 `scene_layout` 与 `environment_map`。
- 主入口在 Planner 后调用 Refiner，并把结果落盘，保证 Phase 3/4 复用。
- 更新配置示例与复现清单，记录 Phase 2 Guidance 进度，确保路径和模型设置可配置。

# 实施计划
实施检查清单：
1. 在 `scenethesis/services/providers.py` 中新增图像生成抽象、`ImageProviderConfig`、`GeminiImageProvider` 实现以及工厂函数，封装 Gemini 3 Pro Image 的请求与响应解析。
2. 重构 `scenethesis/modules/refiner.py`：接入图像 provider、输出目录配置，生成 Guidance 图文件并返回包含路径、分辨率、初始 scene_layout 与环境贴图的结构。
3. 更新 `scenethesis/main.py`：读取 Phase 2 配置，实例化 Refiner，执行 Planner→Refiner 流程，并写入 Phase 2 JSON 输出。
4. 扩充 `config.yaml.example`：加入 `env_maps` 与 `phase2` 配置示例（image provider/model/size/output_dir）。
5. 修改 `docs/repro_checklist.md` Phase 2 “Guidance 图生成”项，标注已实现并说明能力。

# 当前执行步骤
> 正在执行: "全部计划步骤完成，等待复核"

# 任务进度
* 2026-01-10 22:58 CST
    * 步骤：1. providers.py 添加图像 Provider 抽象与 Gemini 实现
    * 修改：`scenethesis/services/providers.py`
    * 更改摘要：新增 ImageProviderConfig、GeminiImageProvider 以及 create_image_provider，提供统一图像生成入口
    * 原因：执行计划步骤 1
    * 阻碍：无
    * 用户确认状态：待确认
* 2026-01-10 22:58 CST
    * 步骤：2. refiner.py 接入图像 provider 并写入 Guidance 输出
    * 修改：`scenethesis/modules/refiner.py`
    * 更改摘要：Refiner 读取 planner 结果生成 prompt，调用 provider 产出图像并写入 output，同时生成初始 layout 与环境贴图
    * 原因：执行计划步骤 2
    * 阻碍：无
    * 用户确认状态：待确认
* 2026-01-10 22:58 CST
    * 步骤：3. main.py 串联 planner→refiner 并输出 Phase 2 结果
    * 修改：`scenethesis/main.py`
    * 更改摘要：主流程根据配置实例化 Refiner，生成并保存 Phase 2 输出 JSON 与 Guidance 图
    * 原因：执行计划步骤 3
    * 阻碍：无
    * 用户确认状态：待确认
* 2026-01-10 22:58 CST
    * 步骤：4. config.yaml.example 添加 Phase 2 配置示例
    * 修改：`config.yaml.example`
    * 更改摘要：新增 env_maps 与 phase2 块，展示图像 provider/模型/尺寸/输出目录配置方式
    * 原因：执行计划步骤 4
    * 阻碍：无
    * 用户确认状态：待确认
* 2026-01-10 22:58 CST
    * 步骤：5. docs/repro_checklist.md 更新 Guidance 进度
    * 修改：`docs/repro_checklist.md`
    * 更改摘要：将 Phase 2 Guidance 生成项标记完成并补充能力说明
    * 原因：执行计划步骤 5
    * 阻碍：无
    * 用户确认状态：待确认
* 2026-01-10 23:05 CST
    * 步骤：附加 - 调整 Gemini 图像 contents 格式
    * 修改：`scenethesis/services/providers.py`
    * 更改摘要：修正 generate_image 调用，改为字符串 prompt 以符合 google.genai SDK 的 contents 要求
    * 原因：确保测试脚本能成功生成图像
    * 阻碍：无
    * 用户确认状态：待确认
* 2026-01-10 23:10 CST
    * 步骤：附加 - 支持 Gemini 图像模型别名与 fallback
    * 修改：`scenethesis/services/providers.py`, `scenethesis/main.py`, `test_files/test_refiner_from_json.py`, `config.yaml`, `config.yaml.example`
    * 更改摘要：ImageProviderConfig 增加 fallback/api_version/negative_prompt，Refiner 测试脚本与主流程读取 phase2 扩展配置，并在实际配置中设置 Gemini 2.5 + fallback
    * 原因：满足用户切换 Gemini 2.5 Flash Image (Nano Banana) 的需求，同时在缺少模型时自动回退
    * 阻碍：无
    * 用户确认状态：待确认
* 2026-01-10 23:15 CST
    * 步骤：附加 - 兼容 Gemini API 不支持 negative_prompt
    * 修改：`scenethesis/services/providers.py`, `config.yaml`, `config.yaml.example`
    * 更改摘要：仅在用户配置非空 negative_prompt 时才传递，示例配置去掉该字段以避免 API 报错
    * 原因：解决运行测试时出现的 `negative_prompt parameter is not supported` 错误
    * 阻碍：无
    * 用户确认状态：待确认
* 2026-01-10 23:20 CST
    * 步骤：附加 - 兼容 Gemini API 不支持 temperature
    * 修改：`scenethesis/services/providers.py`, `scenethesis/main.py`, `test_files/test_refiner_from_json.py`, `config.yaml`, `config.yaml.example`
    * 更改摘要：移除 temperature 参数并清理相关配置，避免 `Extra inputs are not permitted` 校验错误
    * 原因：运行测试时 GenerateImagesConfig 不支持 temperature 字段
    * 阻碍：无
    * 用户确认状态：待确认
* 2026-01-10 23:25 CST
    * 步骤：附加 - API 版本 fallback
    * 修改：`scenethesis/services/providers.py`
    * 更改摘要：当配置了 api_version 时，generate_image 会先尝试指定版本失败后再自动回退到默认版本，避免 404 NOT_FOUND
    * 原因：Gemini 2.5 Flash Image 在 v1beta 不可用，需要回退至默认版本
    * 阻碍：无
    * 用户确认状态：待确认
* 2026-01-10 23:32 CST
    * 步骤：附加 - Vertex AI 调用 Nano 支持
    * 修改：`scenethesis/services/providers.py`, `scenethesis/main.py`, `test_files/test_refiner_from_json.py`, `config.yaml`, `config.yaml.example`
    * 更改摘要：LLM/Image Provider 可根据 vertex.enabled 切换到 Vertex AI 客户端（传入 project/location），测试脚本与配置示例加入 Vertex 段落，满足通过 Vertex AI 调用 Nano Banana 的需求
    * 原因：用户要求使用 Vertex AI 部署方式
    * 阻碍：无
    * 用户确认状态：待确认

# 最终审查
