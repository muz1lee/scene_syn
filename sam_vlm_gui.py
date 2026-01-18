#!/usr/bin/env python3
"""
SAM3 + Qwen VLM 结合打点应用
功能：
1. 加载图片
2. SAM3文本输入 → 获得mask
3. VLM文本输入 → 获得bbox (包含坐标系归一化修复)
4. 计算mask和bbox交集
5. 可视化展示所有结果
"""

import base64
import io
import json
import os
import threading
import re  # 添加正则支持
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageTk

# ========== SAM3 配置 ==========

SAM3_SERVER_URL = "http://101.132.143.105:5081"



# ========== Qwen VLM 配置 ==========

QWEN_BASE_URL = (

"http://1054059136692489.cn-beijing.pai-eas.aliyuncs.com/api/predict/qwen3_vl_235b_a22b_instruct_h20"

)

QWEN_CHAT_URL = f"{QWEN_BASE_URL}/v1/chat/completions"

QWEN_AUTH_TOKEN = "N2I4Mjc0MjkxN2M1Y2NmYzUwNzE0YmEzNjMwOTAwNTE0OWE2YWRjNg=="

QWEN_MODEL_ID = "Qwen3-VL-235B-A22B-Instruct"

QWEN_SYSTEM_PROMPT = "你是一个多模态助手。"




def encode_image_to_base64(image: Image.Image) -> str:
    """将PIL图像编码为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class SamVlmApp(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=12)
        self.pack(fill="both", expand=True)

        self.master = master
        self.master.title("SAM3 + Qwen VLM 打点应用 (Fixed)")

        # 状态变量
        self.current_image: Image.Image | None = None
        self.image_path: str | None = None
        self.base_frame_image: Image.Image | None = None
        self.sam3_mask: np.ndarray | None = None
        self.vlm_bbox: tuple[int, int, int, int] | None = None
        self.intersection_mask: np.ndarray | None = None
        self.cap: cv2.VideoCapture | None = None
        self.video_path: str | None = None
        self.total_frames: int = 0
        self.current_frame_index: int = 0
        self.is_image_source: bool = True
        self.rotation_steps: int = 0

        self.frame_slider: ttk.Scale | None = None
        self.frame_info_var = tk.StringVar(value="帧: -/-")
        self._slider_programmatic = False

        # 构建UI
        self._build_ui()

    def _build_ui(self) -> None:
        """构建用户界面"""
        # ========== 顶部：文件加载 ==========
        file_frame = ttk.LabelFrame(self, text="媒体加载")
        file_frame.pack(fill="x", pady=(0, 8))

        load_btn = ttk.Button(file_frame, text="加载图片/视频", command=self.load_media)
        load_btn.pack(side="left", padx=8, pady=8)

        ttk.Button(
            file_frame, text="↺ 逆时针90°", command=lambda: self.rotate_image(-1)
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            file_frame, text="↻ 顺时针90°", command=lambda: self.rotate_image(1)
        ).pack(side="left")

        self.file_label = ttk.Label(file_frame, text="未加载媒体")
        self.file_label.pack(side="left", padx=8)

        slider_frame = ttk.Frame(self)
        slider_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(slider_frame, textvariable=self.frame_info_var, width=12).pack(
            side="left", padx=(8, 4)
        )
        self.frame_slider = ttk.Scale(
            slider_frame, from_=0, to=0, orient="horizontal", command=self._on_frame_slider
        )
        self.frame_slider.pack(fill="x", expand=True, padx=(0, 8))
        self.frame_slider.state(["disabled"])

        # ========== 中部：输入区域 ==========
        input_frame = ttk.LabelFrame(self, text="模型输入")
        input_frame.pack(fill="x", pady=(0, 8))

        # SAM3 输入
        sam3_row = ttk.Frame(input_frame)
        sam3_row.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(sam3_row, text="SAM3 Prompt:", width=15).pack(side="left")
        self.sam3_entry = ttk.Entry(sam3_row)
        self.sam3_entry.pack(side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Button(sam3_row, text="生成Mask", command=self.run_sam3).pack(side="left")

        # VLM 输入
        vlm_row = ttk.Frame(input_frame)
        vlm_row.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Label(vlm_row, text="VLM Prompt:", width=15).pack(side="left")
        self.vlm_entry = ttk.Entry(vlm_row)
        self.vlm_entry.pack(side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Button(vlm_row, text="预测Bbox", command=self.run_vlm).pack(side="left")

        # 处理按钮
        process_row = ttk.Frame(input_frame)
        process_row.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Button(
            process_row, text="🚀 运行全流程", command=self.run_full_pipeline
        ).pack(side="left", padx=(0, 8))
        ttk.Button(process_row, text="计算交集", command=self.compute_intersection).pack(
            side="left"
        )

        # ========== 可视化区域 ==========
        viz_frame = ttk.LabelFrame(self, text="可视化结果")
        viz_frame.pack(fill="both", expand=True, pady=(0, 8))

        # 创建2x2网格显示
        viz_grid = ttk.Frame(viz_frame)
        viz_grid.pack(fill="both", expand=True, padx=8, pady=8)

        # 配置网格权重
        viz_grid.columnconfigure(0, weight=1)
        viz_grid.columnconfigure(1, weight=1)
        viz_grid.rowconfigure(0, weight=1)
        viz_grid.rowconfigure(1, weight=1)

        # 左上：原图
        frame_original = ttk.LabelFrame(viz_grid, text="原图")
        frame_original.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.canvas_original = tk.Canvas(
            frame_original, bg="black", highlightthickness=0
        )
        self.canvas_original.pack(fill="both", expand=True)

        # 右上：SAM3 Mask
        frame_sam = ttk.LabelFrame(viz_grid, text="SAM3 Mask")
        frame_sam.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.canvas_sam = tk.Canvas(frame_sam, bg="black", highlightthickness=0)
        self.canvas_sam.pack(fill="both", expand=True)

        # 左下：VLM Bbox
        frame_vlm = ttk.LabelFrame(viz_grid, text="VLM Bbox")
        frame_vlm.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.canvas_vlm = tk.Canvas(frame_vlm, bg="black", highlightthickness=0)
        self.canvas_vlm.pack(fill="both", expand=True)

        # 右下：交集结果
        frame_intersection = ttk.LabelFrame(viz_grid, text="交集结果")
        frame_intersection.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)
        self.canvas_intersection = tk.Canvas(
            frame_intersection, bg="black", highlightthickness=0
        )
        self.canvas_intersection.pack(fill="both", expand=True)

        # ========== 底部：状态和日志 ==========
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", pady=(0, 8))

        self.status_var = tk.StringVar(value="等待加载图片...")
        ttk.Label(status_frame, textvariable=self.status_var).pack(
            side="left", fill="x", expand=True
        )

        log_frame = ttk.LabelFrame(self, text="日志")
        log_frame.pack(fill="both", expand=True)

        self.log_box = scrolledtext.ScrolledText(log_frame, height=6, wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.log_box.configure(state="disabled")

    def log(self, message: str) -> None:
        """添加日志消息"""
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def load_media(self) -> None:
        """加载图片或视频"""
        filepath = filedialog.askopenfilename(
            title="选择图片或视频",
            filetypes=[
                ("媒体文件", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.mpg *.mpeg *.wmv *.flv *.webm"),
                ("所有文件", "*.*"),
            ],
        )
        if not filepath:
            return

        ext = os.path.splitext(filepath)[1].lower()
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".flv", ".webm"}

        if ext in video_exts:
            self._load_video_file(filepath)
        else:
            self._load_image_file(filepath)

    def _release_video(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

    def _configure_slider(self, to_value: int, enabled: bool) -> None:
        if not self.frame_slider:
            return
        self._slider_programmatic = True
        self.frame_slider.configure(from_=0, to=to_value)
        self.frame_slider.set(0)
        self._slider_programmatic = False
        if enabled:
            self.frame_slider.state(["!disabled"])
        else:
            self.frame_slider.state(["disabled"])

    def _load_image_file(self, filepath: str) -> None:
        try:
            image = Image.open(filepath).convert("RGB")
        except Exception as exc:
            messagebox.showerror("错误", f"加载图片失败: {exc}")
            self.log(f"✗ 加载失败: {exc}")
            return

        self._release_video()
        self.is_image_source = True
        self.video_path = None
        self.image_path = filepath
        self.base_frame_image = image
        self.rotation_steps = 0
        self.total_frames = 1
        self.current_frame_index = 0
        self._configure_slider(0, enabled=False)
        self.frame_info_var.set("帧: 1/1")
        self.file_label.config(text=os.path.basename(filepath))
        self.status_var.set(f"已加载图片: {os.path.basename(filepath)}")
        self.log(f"✓ 加载图片: {filepath}")
        self._clear_results()
        self._apply_rotation_to_current_frame()

    def _load_video_file(self, filepath: str) -> None:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件")
            self.log("✗ 视频打开失败")
            return
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames <= 0:
            cap.release()
            messagebox.showerror("错误", "无法读取视频帧数")
            self.log("✗ 视频帧数无效")
            return

        self._release_video()
        self.cap = cap
        self.is_image_source = False
        self.video_path = filepath
        self.image_path = filepath
        self.rotation_steps = 0
        self.total_frames = frames
        self.current_frame_index = 0
        self._configure_slider(max(frames - 1, 0), enabled=True)
        self.file_label.config(text=f"{os.path.basename(filepath)} (视频)")
        self.status_var.set(f"已加载视频: {os.path.basename(filepath)}")
        self.log(f"✓ 加载视频: {filepath} (帧数 {frames})")
        self._display_frame(0)

    def _display_frame(self, frame_index: int) -> None:
        if self.is_image_source:
            if self.base_frame_image:
                self.current_frame_index = 0
                self._clear_results()
                self._apply_rotation_to_current_frame()
                self.frame_info_var.set("帧: 1/1")
            return

        if not self.cap:
            return

        if frame_index < 0 or frame_index >= self.total_frames:
            frame_index = max(0, min(frame_index, self.total_frames - 1))

        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            self.log(f"✗ 无法定位到帧 {frame_index}")
            return

        success, frame = self.cap.read()
        if not success or frame is None:
            self.log(f"✗ 无法读取帧 {frame_index}")
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        self.base_frame_image = image
        self.current_frame_index = frame_index
        self._clear_results()
        self._apply_rotation_to_current_frame()
        self._update_frame_info()
        if self.frame_slider and not self._slider_programmatic:
            self._slider_programmatic = True
            self.frame_slider.set(frame_index)
            self._slider_programmatic = False

    def _update_frame_info(self) -> None:
        if self.is_image_source or self.total_frames <= 1:
            self.frame_info_var.set("帧: 1/1")
        else:
            self.frame_info_var.set(
                f"帧: {self.current_frame_index + 1}/{self.total_frames}"
            )

    def _on_frame_slider(self, value: str) -> None:
        if self._slider_programmatic or self.is_image_source:
            return
        index = int(float(value))
        if index != self.current_frame_index:
            self._display_frame(index)

    def _display_image(
        self, canvas: tk.Canvas, image: Image.Image, bbox: tuple | None = None
    ) -> None:
        """在画布上显示图片"""
        # 缩放图片以适应画布
        max_size = 400
        display_img = image.copy()
        orig_w, orig_h = image.size
        display_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        disp_w, disp_h = display_img.size

        # 如果有bbox，绘制在图上
        if bbox:
            draw = ImageDraw.Draw(display_img)
            # 计算缩放比例：显示尺寸 / 真实尺寸
            scale_x = disp_w / orig_w
            scale_y = disp_h / orig_h
            
            x1, y1, x2, y2 = bbox
            scaled_bbox = (
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y),
            )
            # 确保bbox在显示范围内
            scaled_bbox = (
                max(0, min(scaled_bbox[0], disp_w - 1)),
                max(0, min(scaled_bbox[1], disp_h - 1)),
                max(0, min(scaled_bbox[2], disp_w - 1)),
                max(0, min(scaled_bbox[3], disp_h - 1)),
            )
            draw.rectangle(scaled_bbox, outline="red", width=3)

        photo = ImageTk.PhotoImage(display_img)
        canvas.delete("all")
        canvas.config(width=display_img.width, height=display_img.height)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo  # 保持引用

    def _clear_results(self) -> None:
        self.sam3_mask = None
        self.vlm_bbox = None
        self.intersection_mask = None
        self.canvas_sam.delete("all")
        self.canvas_vlm.delete("all")
        self.canvas_intersection.delete("all")

    def rotate_image(self, steps: int) -> None:
        if not self.base_frame_image:
            return
        self.rotation_steps = (self.rotation_steps + steps) % 4
        self._clear_results()
        self._apply_rotation_to_current_frame()
        self.log("↻ 图像已旋转")

    def _apply_rotation_to_current_frame(self) -> None:
        if not self.base_frame_image:
            return
        rotated = self._rotate_image(self.base_frame_image, self.rotation_steps)
        self.current_image = rotated
        self._display_image(self.canvas_original, rotated)

    def _rotate_image(self, image: Image.Image, steps: int) -> Image.Image:
        steps = steps % 4
        if steps == 0:
            return image.copy()
        if steps == 1:  # 顺时针90°
            return image.transpose(Image.Transpose.ROTATE_270)
        if steps == 2:
            return image.transpose(Image.Transpose.ROTATE_180)
        if steps == 3:
            return image.transpose(Image.Transpose.ROTATE_90)
        return image.copy()

    def run_sam3(self) -> None:
        """调用SAM3获取mask"""
        if not self.current_image:
            messagebox.showwarning("警告", "请先加载图片或视频帧")
            return

        text_prompt = self.sam3_entry.get().strip()
        if not text_prompt:
            messagebox.showwarning("警告", "请输入SAM3 prompt")
            return

        self.status_var.set("正在调用SAM3...")
        self.log(f"→ SAM3 请求: {text_prompt}")

        def worker():
            try:
                if not self.current_image:
                    raise RuntimeError("未找到已加载的图像数据")

                image_b64 = encode_image_to_base64(self.current_image)
                payload: dict[str, Any] = {"image": image_b64, "text_prompt": text_prompt}
                response = requests.post(
                    f"{SAM3_SERVER_URL}/segment",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60,
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"SAM3 请求失败: {response.status_code} {response.text}"
                    )

                result = response.json()
                if not result.get("success"):
                    raise RuntimeError(result.get("error", "未知错误"))

                num_detections = result.get("num_detections", 0)
                self.log(f"✓ SAM3检测到 {num_detections} 个目标")

                # 合并所有mask
                if num_detections > 0 and "detections" in result:
                    original_np = np.array(self.current_image)
                    h, w = original_np.shape[:2]
                    combined_mask = np.zeros((h, w), dtype=np.uint8)

                    for detection in result["detections"]:
                        mask_data = base64.b64decode(detection["mask"])
                        mask_img = Image.open(io.BytesIO(mask_data))
                        mask_np = np.array(mask_img)

                        if mask_np.shape != (h, w):
                            mask_img = mask_img.resize((w, h), Image.NEAREST)
                            mask_np = np.array(mask_img)

                        combined_mask = np.maximum(combined_mask, mask_np)

                    self.sam3_mask = combined_mask

                    # 可视化mask
                    self.master.after(0, lambda: self._visualize_sam3_mask())
                    self.master.after(
                        0, lambda: self.status_var.set("SAM3处理完成")
                    )
                else:
                    self.master.after(0, lambda: self.log("⚠ SAM3未检测到目标"))
                    self.master.after(
                        0, lambda: self.status_var.set("SAM3未检测到目标")
                    )

            except Exception as exc:
                self.master.after(0, lambda e=exc: self.log(f"✗ SAM3错误: {e}"))
                self.master.after(0, lambda e=exc: messagebox.showerror("错误", str(e)))
                self.master.after(0, lambda e=exc: self.status_var.set("SAM3处理失败"))

        threading.Thread(target=worker, daemon=True).start()

    def _visualize_sam3_mask(self) -> None:
        """可视化SAM3 mask结果"""
        if self.sam3_mask is None or self.current_image is None:
            return

        # 创建彩色mask叠加图
        original_np = np.array(self.current_image)
        mask_colored = np.zeros_like(original_np)
        mask_colored[self.sam3_mask > 128] = [0, 255, 0]  # 绿色mask

        # 半透明叠加
        overlay = cv2.addWeighted(original_np, 0.6, mask_colored, 0.4, 0)
        overlay_img = Image.fromarray(overlay)

        self._display_image(self.canvas_sam, overlay_img)
        self.log("✓ SAM3 mask可视化完成")

    def run_vlm(self) -> None:
        """调用Qwen VLM获取bbox"""
        if not self.current_image:
            messagebox.showwarning("警告", "请先加载图片")
            return

        text_prompt = self.vlm_entry.get().strip()
        if not text_prompt:
            messagebox.showwarning("警告", "请输入VLM prompt")
            return

        self.status_var.set("正在调用VLM...")
        self.log(f"→ VLM 请求: {text_prompt}")

        def worker():
            try:
                # 编码图片
                frame_b64 = encode_image_to_base64(self.current_image)

                # 修改Prompt，强制要求JSON格式输出
                final_prompt = text_prompt + "\n请输出JSON格式，必须包含键 'bbox_2d'。"

                # 调用Qwen API
                headers = {
                    "Authorization": f"Bearer {QWEN_AUTH_TOKEN}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": QWEN_MODEL_ID,
                    "messages": [
                        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": final_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpg;base64,{frame_b64}"
                                    },
                                },
                            ],
                        },
                    ],
                    "temperature": 0.0,
                }

                response = requests.post(
                    QWEN_CHAT_URL, headers=headers, json=payload, timeout=60
                )

                if response.status_code >= 400:
                    raise RuntimeError(
                        f"VLM请求失败 {response.status_code}: {response.text}"
                    )

                data = response.json()
                choices = data.get("choices")
                if not choices:
                    raise RuntimeError("VLM返回空结果")

                message = choices[0].get("message", {})
                content = message.get("content", "")

                self.log(f"✓ VLM响应: {content}")

                # 解析bbox (原始 0-1000 坐标系)
                bbox_raw = self._parse_bbox(content)
                
                if bbox_raw:
                    # [关键修复]：将 0-1000 的归一化坐标转换为真实像素坐标
                    w, h = self.current_image.size
                    
                    # 转换公式: 真实坐标 = (归一化坐标 / 1000) * 真实尺寸
                    x1 = int(bbox_raw[0] / 1000 * w)
                    y1 = int(bbox_raw[1] / 1000 * h)
                    x2 = int(bbox_raw[2] / 1000 * w)
                    y2 = int(bbox_raw[3] / 1000 * h)
                    
                    self.vlm_bbox = (x1, y1, x2, y2)
                    self.log(f"✓ 坐标转换: {bbox_raw} (1000系) -> {self.vlm_bbox} (像素系)")

                    self.master.after(0, lambda: self._visualize_vlm_bbox())
                    self.master.after(0, lambda: self.status_var.set("VLM处理完成"))
                else:
                    self.master.after(0, lambda: self.log("⚠ 未能从VLM响应中解析bbox"))
                    self.master.after(
                        0, lambda: self.status_var.set("VLM未返回有效bbox")
                    )

            except Exception as exc:
                self.master.after(0, lambda: self.log(f"✗ VLM错误: {exc}"))
                self.master.after(0, lambda: messagebox.showerror("错误", str(exc)))
                self.master.after(0, lambda: self.status_var.set("VLM处理失败"))

        threading.Thread(target=worker, daemon=True).start()

    def _parse_bbox(self, text: str) -> tuple[int, int, int, int] | None:
        """
        解析bbox，支持 JSON 格式提取和正则兜底
        目标格式: [x1, y1, x2, y2]
        """
        # --- 策略 1: 尝试解析 JSON (针对 Qwen 的标准输出格式) ---
        try:
            clean_text = text.strip()
            # 清理 Markdown 代码块标记
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()
            
            # 尝试加载 JSON
            data = json.loads(clean_text)

            # 处理列表包裹的情况 [{"bbox_2d": [...]}]
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                if isinstance(item, dict) and "bbox_2d" in item:
                    bbox = item["bbox_2d"]
                    if len(bbox) == 4:
                        return tuple(map(int, bbox))
            
            # 处理直接是字典的情况 {"bbox_2d": [...]}
            elif isinstance(data, dict) and "bbox_2d" in data:
                bbox = data["bbox_2d"]
                if len(bbox) == 4:
                    return tuple(map(int, bbox))

        except Exception as e:
            print(f"DEBUG: JSON解析尝试失败: {e}")

        # --- 策略 2: 正则表达式兜底 (针对非标准纯文本回复) ---
        matches = re.findall(r"\[([\d\s,]+)\]", text)
        for match in matches:
            parts = [p.strip() for p in match.split(",")]
            parts = [p for p in parts if p]
            if len(parts) == 4:
                try:
                    coords = [float(p) for p in parts]
                    return (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
                except ValueError:
                    continue
        
        return None

    def _visualize_vlm_bbox(self) -> None:
        """可视化VLM bbox结果"""
        if self.vlm_bbox is None or self.current_image is None:
            return

        self._display_image(self.canvas_vlm, self.current_image, self.vlm_bbox)
        self.log(f"✓ VLM bbox可视化完成")

    def compute_intersection(self) -> None:
        """计算mask和bbox的交集"""
        if self.sam3_mask is None:
            messagebox.showwarning("警告", "请先运行SAM3获取mask")
            return

        if self.vlm_bbox is None:
            messagebox.showwarning("警告", "请先运行VLM获取bbox")
            return

        self.status_var.set("正在计算交集...")
        self.log("→ 计算mask和bbox交集...")

        try:
            # 创建bbox mask
            h, w = self.sam3_mask.shape
            bbox_mask = np.zeros((h, w), dtype=np.uint8)
            x1, y1, x2, y2 = self.vlm_bbox

            # 确保bbox在图像范围内
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            bbox_mask[y1:y2, x1:x2] = 255

            # 计算交集
            self.intersection_mask = np.logical_and(
                self.sam3_mask > 128, bbox_mask > 128
            ).astype(np.uint8) * 255

            # 可视化交集
            self._visualize_intersection()
            self.status_var.set("交集计算完成")
            self.log("✓ 交集计算完成")

        except Exception as exc:
            self.log(f"✗ 交集计算错误: {exc}")
            messagebox.showerror("错误", f"交集计算失败: {exc}")
            self.status_var.set("交集计算失败")

    def _visualize_intersection(self) -> None:
        """可视化交集结果"""
        if self.intersection_mask is None or self.current_image is None:
            return

        # 创建彩色mask叠加图 + bbox
        original_np = np.array(self.current_image)
        result_img = original_np.copy()

        # 叠加交集mask（黄色）
        mask_colored = np.zeros_like(original_np)
        mask_colored[self.intersection_mask > 0] = [255, 255, 0]  # 黄色
        result_img = cv2.addWeighted(result_img, 0.6, mask_colored, 0.4, 0)

        result_pil = Image.fromarray(result_img)

        # 绘制bbox
        self._display_image(self.canvas_intersection, result_pil, self.vlm_bbox)

        # 统计交集像素数
        intersection_pixels = np.sum(self.intersection_mask > 0)
        sam_pixels = np.sum(self.sam3_mask > 128)
        self.log(
            f"✓ 交集像素: {intersection_pixels} / SAM总像素: {sam_pixels} "
            f"({100*intersection_pixels/max(sam_pixels, 1):.1f}%)"
        )

    def run_full_pipeline(self) -> None:
        """运行完整流程：SAM3 → VLM → 交集"""
        if not self.current_image:
            messagebox.showwarning("警告", "请先加载图片")
            return

        sam_prompt = self.sam3_entry.get().strip()
        vlm_prompt = self.vlm_entry.get().strip()

        if not sam_prompt or not vlm_prompt:
            messagebox.showwarning("警告", "请输入SAM3和VLM的prompt")
            return

        self.log("=" * 50)
        self.log("🚀 开始运行全流程")

        def worker():
            # 1. 运行SAM3
            self.master.after(0, lambda: self.run_sam3())

            # 等待SAM3完成（简单的轮询检查）
            import time

            max_wait = 60  # 最多等待60秒
            wait_time = 0
            while self.sam3_mask is None and wait_time < max_wait:
                time.sleep(0.5)
                wait_time += 0.5

            if self.sam3_mask is None:
                self.master.after(0, lambda: self.log("✗ SAM3超时"))
                return

            # 2. 运行VLM
            self.master.after(0, lambda: self.run_vlm())

            # 等待VLM完成
            wait_time = 0
            while self.vlm_bbox is None and wait_time < max_wait:
                time.sleep(0.5)
                wait_time += 0.5

            if self.vlm_bbox is None:
                self.master.after(0, lambda: self.log("✗ VLM超时"))
                return

            # 3. 计算交集
            time.sleep(0.5)  # 短暂等待确保UI更新
            self.master.after(0, lambda: self.compute_intersection())
            self.master.after(0, lambda: self.log("🎉 全流程完成"))

        threading.Thread(target=worker, daemon=True).start()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    SamVlmApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
