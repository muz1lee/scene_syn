# SceneThesis - 3D Scene Generation System

基于论文复现的 3D 场景生成系统，使用 LLM 驱动的多阶段管线生成物理合理的 3D 场景。

## 项目状态

- ✅ **Phase 1: Coarse Scene Planning** - 已完成
- 🔄 Phase 2: Visual Refinement - 开发中
- 🔄 Phase 3: Physics Optimization - 待开发
- 🔄 Phase 4: Scene Judge - 待开发

## 功能特性

### Phase 1 - 粗级场景规划
- ✅ 自动识别简单/详细描述，执行对应推理路线
- ✅ 基于 Gemini 2.0 的智能物体选择和锚点推理
- ✅ 符合论文标准的锚点选择算法
- ✅ 资产库验证和语义映射
- ✅ 描述增强功能（统一输出格式）
- ✅ 完整的输出验证和错误处理

## 快速开始

### 环境要求
- Python 3.10+
- Google Gemini API Key

### 安装

```bash
# 克隆仓库
git clone https://github.com/muz1lee/scene_syn.git
cd scene_syn

# 安装依赖
pip install google-genai pyyaml

# 配置 API Key
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 配置

创建 `config.yaml`：

```yaml
prompt: "A messy bedroom with a study nook"
model_name: "gemini-2.0-flash-exp"
db_assets:
  - bed
  - sofa
  - desk
  - chair
  - table
  - laptop
  - plant
  - lamp
  - bookshelf
output_dir: "scenethesis/output"
```

### 运行

```bash
# 测试 API 连接
python test_api.py

# 运行 Phase 1 规划
python -m scenethesis.main
```

### 输出示例

```json
{
  "mode": "simple_generated",
  "anchor": "bed",
  "objects": ["bed", "desk", "chair", "table", "bookshelf", "lamp", "laptop", "plant"],
  "detailed_description": "A messy bedroom scene featuring a study nook. In the background, a large unmade bed is positioned on the left side of the room..."
}
```

## 项目结构

```
scene_syn/
├── scenethesis/
│   ├── modules/
│   │   ├── planner.py      # Phase 1: 粗级规划
│   │   ├── refiner.py      # Phase 2: 视觉细化（待实现）
│   │   ├── physics.py      # Phase 3: 物理优化（待实现）
│   │   └── judge.py        # Phase 4: 场景裁判（待实现）
│   ├── services/
│   │   └── providers.py    # LLM Provider (Gemini)
│   └── main.py             # 主入口
├── docs/
│   ├── repro_checklist.md  # 复现检查清单
│   └── sdk_migration.md    # SDK 迁移文档
├── config.yaml             # 配置文件
├── test_api.py             # API 测试脚本
└── README.md
```

## 技术栈

- **LLM**: Google Gemini 2.0 Flash
- **SDK**: google-genai (新版)
- **语言**: Python 3.12

## 核心设计

### Phase 1: Coarse Scene Planning

#### 简单模式
- 输入：简短描述（如 "A cozy bedroom"）
- 处理：LLM 自动推理物体、锚点和空间关系
- 输出：完整的场景规划 JSON

#### 详细模式
- 输入：详细描述（如 "A desk next to a bed with a laptop on it"）
- 处理：实体抽取 → 资产验证 → 锚点推理 → 描述增强
- 输出：验证后的场景规划 JSON

### 锚点选择标准（符合论文）
1. 大型物体直接接地（如床、桌子、书架）
2. 影响其他物体摆放的物体
3. 定义场景布局方向的物体
4. 占据最高空间层级（除地面外）

## 开发文档

- [复现检查清单](docs/repro_checklist.md) - 各阶段开发进度
- [SDK 迁移文档](docs/sdk_migration.md) - Gemini SDK 迁移指南
- [复现计划](replication_plan.md) - 完整的论文复现计划

## API 配置

### 获取 Gemini API Key
1. 访问 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 创建 API Key
3. 添加到 `.env` 文件

### 模型选择
- 推荐：`gemini-2.0-flash-exp`（速度快，成本低）
- 备用：`gemini-1.5-flash`, `gemini-1.5-pro`

## 测试

```bash
# 测试 API 连接
python test_api.py

# 测试 Phase 1 规划
python -m scenethesis.main
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 致谢

本项目基于 SceneThesis 论文复现。
