#!/usr/bin/env python3
"""
测试 Phase 2 的外部服务集成：SAM3 和 Depth Pro
"""

import sys
from pathlib import Path
from io import BytesIO

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
import yaml

from scenethesis.services.sam3_client import Sam3Client
from scenethesis.services.depth_pro_client import DepthProClient


def load_config():
    """加载配置文件"""
    config_path = project_root / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_sam3_client(config):
    """测试 SAM3 客户端"""
    print("\n" + "=" * 60)
    print("测试 SAM3 客户端")
    print("=" * 60)

    sam3_cfg = config.get("phase2", {}).get("scene_graph", {}).get("sam3", {})
    endpoint = sam3_cfg.get("endpoint")

    if not endpoint:
        print("❌ SAM3 endpoint 未配置")
        return False

    print(f"Endpoint: {endpoint}")

    # 创建一个简单的测试图像（640x640 白色背景，中间一个蓝色矩形）
    test_image = Image.new("RGB", (640, 640), color="white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    draw.rectangle([200, 200, 440, 440], fill="blue", outline="black", width=3)

    # 转换为字节
    buffer = BytesIO()
    test_image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    print(f"测试图像大小: {len(image_bytes)} bytes")

    try:
        client = Sam3Client(endpoint=endpoint)
        print("✓ SAM3 客户端创建成功")

        # 测试分割
        print("\n尝试分割 'blue rectangle'...")
        detections = client.segment(image_bytes, text_prompt="blue rectangle")

        print(f"✓ 检测到 {len(detections)} 个目标")
        for idx, det in enumerate(detections, 1):
            print(f"  目标 {idx}:")
            print(f"    - prompt: {det.prompt}")
            print(f"    - score: {det.score:.3f}")
            print(f"    - bbox: {det.bbox}")
            print(f"    - has_mask: {det.mask_image is not None}")

        return True

    except Exception as exc:
        print(f"❌ SAM3 测试失败: {exc}")
        import traceback
        traceback.print_exc()
        return False


def test_depth_pro_client():
    """测试 Depth Pro 客户端"""
    print("\n" + "=" * 60)
    print("测试 Depth Pro 客户端（本地部署）")
    print("=" * 60)

    # 创建一个简单的测试图像
    test_image = Image.new("RGB", (640, 640), color="gray")
    buffer = BytesIO()
    test_image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    print(f"测试图像大小: {len(image_bytes)} bytes")

    try:
        # 检查 CUDA 是否可用
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        client = DepthProClient(device=device)
        print("✓ Depth Pro 客户端创建成功")

        # 测试深度估计
        print("\n执行深度估计...")
        result = client.infer(image_bytes)

        print("✓ 深度估计完成")
        print(f"  - depth_map shape: {result.depth_map.shape if result.depth_map is not None else 'None'}")
        print(f"  - min_depth: {result.min_depth}")
        print(f"  - max_depth: {result.max_depth}")
        print(f"  - median_depth: {result.median_depth}")

        return True

    except ImportError as exc:
        print(f"❌ 依赖库未安装: {exc}")
        print("请运行: pip install git+https://github.com/apple/ml-depth-pro.git")
        return False
    except Exception as exc:
        print(f"❌ Depth Pro 测试失败: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Phase 2 服务集成测试")
    print("=" * 60)

    # 加载配置
    try:
        config = load_config()
        print("✓ 配置文件加载成功")
    except Exception as exc:
        print(f"❌ 配置文件加载失败: {exc}")
        return 1

    # 测试 SAM3
    sam3_ok = test_sam3_client(config)

    # 测试 Depth Pro
    depth_ok = test_depth_pro_client()

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"SAM3 客户端: {'✓ 通过' if sam3_ok else '❌ 失败'}")
    print(f"Depth Pro 客户端: {'✓ 通过' if depth_ok else '❌ 失败'}")

    if sam3_ok and depth_ok:
        print("\n✓ 所有测试通过！")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
