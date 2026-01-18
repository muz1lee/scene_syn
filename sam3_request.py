#!/usr/bin/env python3
"""
SAM3 客户端 - 支持文本、框、以及文本+框三种模式。
"""

import base64
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

# ========== 配置 ==========
SERVER_URL = "http://101.132.143.105:5081"
IMAGE_PATH = "/Users/knowin-wenqian/knowin/sam_dino/data_clothes/origin_left_20251218_142233.jpg"

# 方式一：纯文本模式
TEXT_PROMPT = "cloth on the washing machine"
BOX_PROMPT = None

# 方式二：纯框模式
# TEXT_PROMPT = None
# BOX_PROMPT = [600, 200, 750, 400]

# 方式三：文本 + 框
# TEXT_PROMPT = "box"
# BOX_PROMPT = [629, 274, 708, 339]

# 输出文件
ANNOTATED_OUTPUT = "annotated.jpg"
SEGMENTED_OUTPUT = "segmented.png"
SEGMENTED_WHITE_OUTPUT = "segmented_white.jpg"
# ==========================


def load_image_base64(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def describe_mode() -> str:
    if TEXT_PROMPT and BOX_PROMPT:
        return "文本+框"
    if TEXT_PROMPT:
        return "纯文本"
    if BOX_PROMPT:
        return "纯框"
    return "无效"


def build_payload(image_b64: str) -> dict:
    payload: dict[str, object] = {"image": image_b64}
    if TEXT_PROMPT:
        payload["text_prompt"] = TEXT_PROMPT
    if BOX_PROMPT:
        payload["box_prompt"] = BOX_PROMPT
    return payload


def segment() -> dict | None:
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        print(f"✗ 图片不存在: {image_path}")
        return None

    mode = describe_mode()
    if mode == "无效":
        print("✗ 必须提供 TEXT_PROMPT 和/或 BOX_PROMPT")
        return None

    print("\n" + "=" * 60)
    print("SAM3 客户端 - 发送分割请求")
    print("=" * 60)
    print(f"服务器: {SERVER_URL}")
    print(f"模式: {mode}")
    if TEXT_PROMPT:
        print(f"文本 prompt: {TEXT_PROMPT}")
    if BOX_PROMPT:
        print(f"框 prompt: {BOX_PROMPT}")

    try:
        image_b64 = load_image_base64(str(image_path))
    except OSError as exc:
        print(f"✗ 打开图片失败: {exc}")
        return None

    payload = build_payload(image_b64)

    try:
        response = requests.post(
            f"{SERVER_URL}/segment",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
    except requests.exceptions.Timeout:
        print("✗ 请求超时")
        return None
    except requests.exceptions.ConnectionError:
        print(f"✗ 连接失败，请检查服务器: {SERVER_URL}")
        return None

    if response.status_code != 200:
        print(f"✗ 请求失败: HTTP {response.status_code}")
        print(response.text)
        return None

    try:
        result = response.json()
    except ValueError:
        print("✗ 返回数据不是合法 JSON")
        return None

    if not result.get("success"):
        print("✗ 分割失败")
        print(result.get("error", "未知错误"))
        if "traceback" in result:
            print("------ 服务器错误栈 ------")
            print(result["traceback"])
        return None

    return result


def save_results(result: dict) -> None:
    detections = result.get("detections") or []
    num = result.get("num_detections", len(detections))
    mode = result.get("mode", describe_mode())
    message = result.get("message")

    print("\n✓ 分割成功")
    print(f"模式: {mode}")
    print(f"检测到 {num} 个目标")
    if message:
        print(f"信息: {message}")
    if not detections:
        print("没有检测到任何目标。")
        return

    for idx, det in enumerate(detections, 1):
        prompt = det.get("prompt", "")
        score = det.get("score", 0.0)
        bbox = det.get("bbox")
        bbox_str = ""
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = map(int, bbox[:4])
            bbox_str = f" bbox=[{x1},{y1},{x2},{y2}]"
        print(f"  目标 {idx}: {prompt} (score={score:.3f}){bbox_str}")

    original = Image.open(IMAGE_PATH).convert("RGB")
    original_np = np.array(original)
    h, w = original_np.shape[:2]

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for det in detections:
        mask_b64 = det.get("mask")
        if not mask_b64:
            continue
        mask_data = base64.b64decode(mask_b64)
        mask_img = Image.open(BytesIO(mask_data))
        mask_np = np.array(mask_img)
        if mask_np.shape != (h, w):
            mask_img = mask_img.resize((w, h), Image.NEAREST)
            mask_np = np.array(mask_img)
        combined_mask = np.maximum(combined_mask, mask_np)

    segmented_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    segmented_rgba[:, :, :3] = original_np
    segmented_rgba[:, :, 3] = combined_mask
    Image.fromarray(segmented_rgba).save(SEGMENTED_OUTPUT)
    print(f"✓ 透明背景: {SEGMENTED_OUTPUT}")

    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    mask_bool = combined_mask > 128
    white_bg[mask_bool] = original_np[mask_bool]
    Image.fromarray(white_bg).save(SEGMENTED_WHITE_OUTPUT)
    print(f"✓ 白色背景: {SEGMENTED_WHITE_OUTPUT}")

    vis_image = create_visualization(original_np, detections)
    Image.fromarray(vis_image).save(ANNOTATED_OUTPUT)
    print(f"✓ 可视化图: {ANNOTATED_OUTPUT}")


def create_visualization(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    result_image = image.copy()
    h, w = result_image.shape[:2]
    rng = np.random.default_rng(42)

    for idx, det in enumerate(detections):
        color = tuple(int(x) for x in rng.integers(50, 255, size=3))
        prompt = det.get("prompt", f"obj_{idx}")
        score = det.get("score", 0.0)
        bbox = det.get("bbox")
        mask_b64 = det.get("mask")
        if not mask_b64:
            continue

        mask_data = base64.b64decode(mask_b64)
        mask_img = Image.open(BytesIO(mask_data))
        mask_np = np.array(mask_img)
        if mask_np.shape != (h, w):
            mask_img = mask_img.resize((w, h), Image.NEAREST)
            mask_np = np.array(mask_img)
        mask_bool = mask_np > 128

        colored_mask = np.zeros_like(result_image)
        colored_mask[mask_bool] = color
        result_image = cv2.addWeighted(result_image, 1.0, colored_mask, 0.4, 0)

        contours, _ = cv2.findContours(
            mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result_image, contours, -1, color, 2)

        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            label = f"{prompt}: {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(
                result_image,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 5, y1),
                color,
                -1,
            )
            cv2.putText(
                result_image,
                label,
                (x1 + 2, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

    return result_image


def main() -> None:
    result = segment()
    if not result:
        sys.exit(1)
    save_results(result)
    print("\n完成！\n")


if __name__ == "__main__":
    main()
