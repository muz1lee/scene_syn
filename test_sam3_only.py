#!/usr/bin/env python3
"""
测试 SAM3 分割服务
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenethesis.services.sam3_client import Sam3Client


def main() -> None:
    print("=" * 60)
    print("SAM3 分割服务测试")
    print("=" * 60)

    # 读取引导图像
    image_path = Path("scenethesis/output/generated_img.png")
    if not image_path.exists():
        print(f"❌ 图像文件不存在: {image_path}")
        return

    image_bytes = image_path.read_bytes()
    print(f"\n✓ 读取图像: {image_path}")
    print(f"  图像大小: {len(image_bytes)} bytes")

    # 初始化 SAM3 客户端
    sam3_client = Sam3Client(
        endpoint="http://101.132.143.105:5081/segment",
        default_text_prompt="",
    )
    print(f"\n✓ 初始化 SAM3 客户端")
    print(f"  Endpoint: http://101.132.143.105:5081/segment")

    # 测试物体列表
    test_objects = [
        "washing machine",
        "toilet",
        "sink",
        "mirror",
        "shower",
    ]

    print(f"\n开始测试分割 {len(test_objects)} 个物体...")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for idx, obj in enumerate(test_objects, 1):
        print(f"\n[{idx}/{len(test_objects)}] 测试物体: {obj}")
        try:
            detections = sam3_client.segment(image_bytes, text_prompt=obj)

            if not detections:
                print(f"  ⚠️  未检测到目标")
                fail_count += 1
                continue

            # 找到置信度最高的检测结果
            best_detection = max(detections, key=lambda det: det.score)

            print(f"  ✓ 检测成功！")
            print(f"    检测数量: {len(detections)}")
            print(f"    最佳置信度: {best_detection.score:.3f}")
            print(f"    边界框: {best_detection.bbox}")
            print(f"    Mask 尺寸: {best_detection.mask_image.size if best_detection.mask_image else 'None'}")

            success_count += 1

        except Exception as exc:
            print(f"  ❌ 分割失败: {exc}")
            fail_count += 1

    # 输出统计
    print("\n" + "=" * 60)
    print("测试结果统计")
    print("=" * 60)
    print(f"✓ 成功: {success_count}/{len(test_objects)}")
    print(f"✗ 失败: {fail_count}/{len(test_objects)}")
    print(f"成功率: {success_count/len(test_objects)*100:.1f}%")


if __name__ == "__main__":
    main()
