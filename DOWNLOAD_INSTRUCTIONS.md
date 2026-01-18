# Objaverse 数据集下载指令

## 📋 环境信息

**Conda 环境**: syn3d
**Python 版本**: 3.10.17
**当前已安装**:
- numpy: 2.2.6 ⚠️ (需要降级)
- torch: 2.2.0+cu121
- objathor: 0.0.8 ✅
- torchvision: 0.17.0+cu121

**硬盘空间**: 206GB 可用

---

## 🚀 完整下载流程

### 步骤 1: 激活环境并修复依赖

```bash
# 激活 syn3d 环境
conda activate syn3d

# 修复 numpy 版本冲突（depth-pro 需要 numpy<2）
pip install "numpy<2" --force-reinstall

# 安装缺失的依赖
pip install attrs ai2thor

# 验证安装
python -c "import objathor; import numpy; print(f'objathor: OK, numpy: {numpy.__version__}')"
```

---

### 步骤 2: 保守下载方案（推荐）⭐

**先下载小文件，查看内容后再决定是否下载大文件**

```bash
# 2.1 下载基础数据（~5-10 GB）
echo "开始下载 holodeck_base_data..."
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23

# 2.2 下载标注数据（~2-5 GB）
echo "开始下载 annotations..."
python -m objathor.dataset.download_annotations --version 2023_09_23

# 2.3 检查已下载的大小
echo "当前下载大小:"
du -sh ~/.objathor-assets/

# 2.4 查看包含哪些资产（可选）
python << 'EOF'
import objathor
from pathlib import Path
import json

assets_dir = Path.home() / ".objathor-assets"
print(f"\n数据集位置: {assets_dir}")
print(f"已下载内容:")
for item in assets_dir.rglob("*"):
    if item.is_file():
        size_mb = item.stat().st_size / (1024*1024)
        if size_mb > 10:  # 只显示大于10MB的文件
            print(f"  {item.name}: {size_mb:.1f} MB")
EOF

# 2.5 如果空间足够，继续下载大文件
echo "准备下载 assets（50-150 GB）..."
read -p "是否继续下载 assets? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -m objathor.dataset.download_assets --version 2023_09_23
fi

# 2.6 下载 CLIP features（20-50 GB）
echo "准备下载 features（20-50 GB）..."
read -p "是否继续下载 features? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -m objathor.dataset.download_features --version 2023_09_23
fi
```

---

### 步骤 3: 一次性下载方案（如果确定空间够用）

```bash
# 激活环境
conda activate syn3d

# 一次性下载所有数据（预计 80-220 GB）
echo "开始下载所有 Objaverse 数据..."
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23 &
PID1=$!

python -m objathor.dataset.download_assets --version 2023_09_23 &
PID2=$!

python -m objathor.dataset.download_annotations --version 2023_09_23 &
PID3=$!

python -m objathor.dataset.download_features --version 2023_09_23 &
PID4=$!

# 等待所有下载完成
wait $PID1 $PID2 $PID3 $PID4
echo "所有下载完成！"
```

---

### 步骤 4: 使用自定义路径（如果默认路径空间不够）

```bash
# 激活环境
conda activate syn3d

# 设置自定义下载路径（替换为你的路径）
export CUSTOM_PATH="/path/to/large/storage"
mkdir -p $CUSTOM_PATH

# 下载到自定义路径
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23 --path $CUSTOM_PATH
python -m objathor.dataset.download_assets --version 2023_09_23 --path $CUSTOM_PATH
python -m objathor.dataset.download_annotations --version 2023_09_23 --path $CUSTOM_PATH
python -m objathor.dataset.download_features --version 2023_09_23 --path $CUSTOM_PATH

# 设置环境变量（添加到 ~/.bashrc 以永久生效）
export OBJAVERSE_ASSETS_DIR=$CUSTOM_PATH
echo "export OBJAVERSE_ASSETS_DIR=$CUSTOM_PATH" >> ~/.bashrc
```

---

## 📊 监控下载进度

### 在另一个终端窗口运行：

```bash
# 实时监控下载大小
watch -n 5 'du -sh ~/.objathor-assets/'

# 或者更详细的监控
watch -n 5 'du -h --max-depth=1 ~/.objathor-assets/ | sort -h'

# 查看硬盘剩余空间
watch -n 10 'df -h /home/knowin-wenqian'
```

---

## 🔧 后台下载（推荐用于长时间下载）

### 使用 screen 或 tmux：

```bash
# 方案 A: 使用 screen
screen -S objaverse_download
conda activate syn3d

# 运行下载命令...
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
# ... 其他下载命令

# 按 Ctrl+A 然后按 D 来 detach
# 重新连接: screen -r objaverse_download

# 方案 B: 使用 nohup
conda activate syn3d
nohup python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23 > download_base.log 2>&1 &
nohup python -m objathor.dataset.download_assets --version 2023_09_23 > download_assets.log 2>&1 &
nohup python -m objathor.dataset.download_annotations --version 2023_09_23 > download_annotations.log 2>&1 &
nohup python -m objathor.dataset.download_features --version 2023_09_23 > download_features.log 2>&1 &

# 查看进度
tail -f download_*.log
```

---

## ✅ 验证下载完成

```bash
conda activate syn3d

# 检查下载的文件
ls -lh ~/.objathor-assets/

# 测试加载数据集
python << 'EOF'
import objathor
from objathor.dataset import load_holodeck_base_data

try:
    print("正在加载 Holodeck 数据集...")
    data = load_holodeck_base_data(version="2023_09_23")
    print("✅ 数据集加载成功！")

    # 显示一些统计信息
    print(f"\n数据集信息:")
    print(f"  版本: 2023_09_23")
    print(f"  位置: ~/.objathor-assets/")

except Exception as e:
    print(f"❌ 加载失败: {e}")
EOF
```

---

## 🎯 推荐执行顺序

### 最保险的方式（分步执行）：

```bash
# 1. 环境准备
conda activate syn3d
pip install "numpy<2" --force-reinstall
pip install attrs ai2thor

# 2. 下载小文件（~15 GB，快速）
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
python -m objathor.dataset.download_annotations --version 2023_09_23

# 3. 检查空间
du -sh ~/.objathor-assets/
df -h /home/knowin-wenqian

# 4. 如果空间足够，下载大文件（可以在 screen 中运行）
screen -S objaverse
conda activate syn3d
python -m objathor.dataset.download_assets --version 2023_09_23
# Ctrl+A, D 来 detach

# 5. 最后下载 features
python -m objathor.dataset.download_features --version 2023_09_23
```

---

## ⚠️ 注意事项

1. **下载时间**:
   - base_data + annotations: ~30分钟 - 1小时
   - assets: **几小时到半天**（取决于网络）
   - features: **1-3小时**

2. **网络稳定性**:
   - 建议使用 screen/tmux 防止断线
   - 如果下载中断，重新运行命令会自动续传

3. **硬盘监控**:
   - 定期检查 `df -h` 确保不会满盘
   - 如果空间不够，立即 Ctrl+C 停止下载

4. **版本一致性**:
   - 所有下载命令都使用 `--version 2023_09_23`
   - 确保版本一致，否则可能不兼容

---

## 📝 快速复制命令（推荐方案）

```bash
# === 完整下载脚本 ===
# 复制以下所有内容到终端执行

# 激活环境
conda activate syn3d

# 修复依赖
pip install "numpy<2" --force-reinstall
pip install attrs ai2thor

# 创建下载脚本
cat > ~/download_objaverse.sh << 'SCRIPT'
#!/bin/bash
set -e

echo "=== Objaverse 数据集下载 ==="
echo "开始时间: $(date)"
echo "可用空间: $(df -h /home/knowin-wenqian | tail -1 | awk '{print $4}')"
echo ""

# 激活环境
source /home/muz1lee1022/miniconda3/etc/profile.d/conda.sh
conda activate syn3d

# 下载基础数据
echo "[1/4] 下载 holodeck_base_data..."
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
echo "✓ 完成"

# 下载标注
echo "[2/4] 下载 annotations..."
python -m objathor.dataset.download_annotations --version 2023_09_23
echo "✓ 完成"

# 检查空间
echo ""
echo "当前已下载: $(du -sh ~/.objathor-assets/ | cut -f1)"
echo "剩余空间: $(df -h /home/knowin-wenqian | tail -1 | awk '{print $4}')"
echo ""

# 下载资产
echo "[3/4] 下载 assets（这可能需要几小时）..."
python -m objathor.dataset.download_assets --version 2023_09_23
echo "✓ 完成"

# 下载特征
echo "[4/4] 下载 features..."
python -m objathor.dataset.download_features --version 2023_09_23
echo "✓ 完成"

echo ""
echo "=== 下载完成 ==="
echo "结束时间: $(date)"
echo "总大小: $(du -sh ~/.objathor-assets/ | cut -f1)"
echo "剩余空间: $(df -h /home/knowin-wenqian | tail -1 | awk '{print $4}')"
SCRIPT

# 赋予执行权限
chmod +x ~/download_objaverse.sh

# 在 screen 中运行
screen -S objaverse -dm bash -c "~/download_objaverse.sh 2>&1 | tee ~/objaverse_download.log"

echo "✅ 下载已在后台启动！"
echo ""
echo "查看进度:"
echo "  screen -r objaverse    # 连接到下载会话"
echo "  tail -f ~/objaverse_download.log    # 查看日志"
echo ""
echo "监控空间:"
echo "  watch -n 5 'du -sh ~/.objathor-assets/'"
```

---

**生成时间**: 2026-01-17
**环境**: syn3d (Python 3.10.17)
**预计下载时间**: 4-12 小时
**预计占用空间**: 80-220 GB
