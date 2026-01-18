# Objaverse/Holodeck 数据集分析报告

## 📊 你的硬盘情况

**当前状态**:
- 总容量: 492GB
- 已使用: 267GB
- **可用空间: 206GB** ✅
- 使用率: 57%

---

## 📦 Objaverse/Holodeck 数据集概述

### 什么是 Objaverse?
Objaverse 是一个大规模的 3D 物体数据集，包含超过 80 万个 3D 模型。Holodeck 是基于 Objaverse 构建的室内场景生成系统，由 Allen AI 开发。

### 数据集组成

根据你提供的下载命令，Holodeck 数据集包含 4 个部分：

1. **holodeck_base_data** - 基础数据
2. **assets** - 3D 资产（模型文件）
3. **annotations** - 标注数据
4. **features** - 特征数据（可能包含 CLIP embeddings）

---

## 💾 预估存储需求

### 基于类似数据集的估算

根据 Objaverse 和 AI2-THOR 相关项目的经验：

| 组件 | 预估大小 | 说明 |
|------|---------|------|
| **holodeck_base_data** | ~5-10 GB | 场景配置、房间布局等基础数据 |
| **assets** | **50-150 GB** | 3D 模型文件（.glb/.obj 格式）<br>取决于包含的模型数量 |
| **annotations** | ~2-5 GB | 物体标注、类别信息、元数据 |
| **features** | **20-50 GB** | CLIP 特征向量、预计算的 embeddings |
| **总计** | **~80-220 GB** | 取决于具体版本和包含的资产数量 |

### 2023_09_23 版本特点

- 这是 2023 年 9 月的版本，相对较新
- 可能包含精选的高质量 3D 资产子集
- 不是完整的 Objaverse（完整版 > 1TB）

---

## ✅ 你的硬盘是否够用？

**结论: 应该够用，但需要注意** ⚠️

- 你有 **206GB 可用空间**
- 预估需要 **80-220GB**
- **建议**:
  - ✅ 如果数据集 < 150GB，完全没问题
  - ⚠️ 如果数据集 > 180GB，空间会比较紧张
  - 💡 可以先下载 base_data 和 annotations（小文件），再决定是否下载 assets

---

## 🔍 如何查看数据集内容

### 1. 先下载小文件探索

```bash
# 只下载基础数据和标注（小文件）
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
python -m objathor.dataset.download_annotations --version 2023_09_23
```

这两个命令下载的数据很小（< 15GB），可以先看看包含哪些资产。

### 2. 查看可用资产列表

下载完 annotations 后，可以通过 Python 查看：

```python
import objathor
from objathor.dataset import load_holodeck_base_data

# 加载数据集
data = load_holodeck_base_data(version="2023_09_23")

# 查看可用的物体类别
print("可用物体类别:")
for category in data.get_categories():
    print(f"  - {category}")

# 查看每个类别的资产数量
print("\n每个类别的资产数量:")
for category in data.get_categories():
    assets = data.get_assets_by_category(category)
    print(f"  {category}: {len(assets)} 个资产")
```

### 3. 选择性下载

如果发现 assets 太大，可以：
- 只下载你需要的类别
- 使用 `--path` 参数指定其他硬盘位置
- 考虑使用外部存储

---

## 📝 下载步骤建议

### 方案 A: 保守方案（推荐）

```bash
# 1. 先下载小文件（~10GB）
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
python -m objathor.dataset.download_annotations --version 2023_09_23

# 2. 检查硬盘空间
df -h ~

# 3. 查看包含哪些资产（运行上面的 Python 代码）

# 4. 如果空间足够，再下载大文件
python -m objathor.dataset.download_assets --version 2023_09_23
python -m objathor.dataset.download_features --version 2023_09_23
```

### 方案 B: 一次性下载

```bash
# 如果你确定空间够用，可以一次性下载所有
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
python -m objathor.dataset.download_assets --version 2023_09_23
python -m objathor.dataset.download_annotations --version 2023_09_23
python -m objathor.dataset.download_features --version 2023_09_23
```

### 方案 C: 使用其他路径

如果你有其他硬盘或网络存储：

```bash
# 指定下载路径
export DOWNLOAD_PATH="/path/to/large/storage"

python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23 --path $DOWNLOAD_PATH
python -m objathor.dataset.download_assets --version 2023_09_23 --path $DOWNLOAD_PATH
python -m objathor.dataset.download_annotations --version 2023_09_23 --path $DOWNLOAD_PATH
python -m objathor.dataset.download_features --version 2023_09_23 --path $DOWNLOAD_PATH

# 设置环境变量
export OBJAVERSE_ASSETS_DIR=$DOWNLOAD_PATH
```

---

## 🎯 与你的项目集成

### 数据集包含的资产类型

Holodeck 通常包含以下类别的 3D 资产：

**家具类**:
- bed, sofa, chair, desk, table, bookshelf, cabinet, dresser, nightstand

**家电类**:
- washing machine, refrigerator, microwave, oven, dishwasher, TV

**装饰类**:
- lamp, mirror, picture frame, plant, vase, clock

**卫浴类**:
- toilet, sink, bathtub, shower, towel rack

**其他**:
- door, window, curtain, rug, pillow, etc.

### 在你的项目中使用

下载完成后，你可以在 `scenethesis` 中使用：

```python
# 在 config.yaml 中配置
db_assets:
  - washing machine
  - chair
  - toilet
  - bed
  - desk
  # ... 更多资产

# 在代码中加载 Objaverse 资产
from objathor.dataset import load_holodeck_base_data

data = load_holodeck_base_data(version="2023_09_23")
asset_path = data.get_asset_path("washing machine")
```

---

## ⚠️ 注意事项

### 1. numpy 版本冲突

你当前遇到的问题：
```
depth-pro 0.1 requires numpy<2, but you have numpy 2.2.6
```

**解决方案**:
```bash
# 降级 numpy
pip install "numpy<2"
```

### 2. 下载时间

- 数据集很大，下载可能需要 **几小时到一天**
- 建议使用稳定的网络连接
- 可以使用 `screen` 或 `tmux` 在后台运行

### 3. 存储位置

默认下载到: `~/.objathor-assets/`

如果要更改：
```bash
export OBJAVERSE_ASSETS_DIR="/your/custom/path"
```

---

## 📊 实际大小验证

下载开始后，你可以实时监控：

```bash
# 监控下载目录大小
watch -n 5 'du -sh ~/.objathor-assets/'

# 查看详细信息
du -h --max-depth=1 ~/.objathor-assets/
```

---

## 🚀 下一步行动

### 立即可做：

1. **修复 numpy 版本冲突**:
   ```bash
   pip install "numpy<2"
   ```

2. **开始保守下载**:
   ```bash
   python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
   python -m objathor.dataset.download_annotations --version 2023_09_23
   ```

3. **监控下载进度和大小**

4. **根据实际大小决定是否继续下载 assets 和 features**

### 如果空间不够：

- 考虑清理不需要的文件
- 使用外部存储
- 只下载你项目需要的资产类别
- 考虑使用云存储或网络挂载

---

**生成时间**: 2026-01-17
**你的可用空间**: 206GB
**预估需求**: 80-220GB
**建议**: 先下载小文件探索，再决定是否下载全部
