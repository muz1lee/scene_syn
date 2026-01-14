#!/usr/bin/env python3
"""
快速测试 Gemini API 连接
"""
import os
import sys
from pathlib import Path

# 手动加载 .env
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent

for env_candidate in [SCRIPT_DIR / ".env", ROOT_DIR / ".env"]:
    if env_candidate.exists():
        for line in env_candidate.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value.strip()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("🔑 API Key: NOT FOUND")
    print("❌ 未找到 GEMINI_API_KEY")
    sys.exit(1)

print(f"🔑 API Key: {api_key[:8]}...{api_key[-4:]}")

try:
    import google.genai as genai
    from google.genai import types
    print("✅ google.genai 导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请运行: pip install google-genai")
    sys.exit(1)

# 创建 Client
try:
    client = genai.Client(api_key=api_key)
    print("✅ genai.Client 创建成功")
except Exception as e:
    print(f"❌ Client 创建失败: {e}")
    sys.exit(1)

# 测试简单请求
print("\n🧪 测试简单生成请求...")
test_models = [
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

for model_name in test_models:
    print(f"\n尝试模型: {model_name}")
    try:
        # 使用新 SDK: client.models.generate_content
        response = client.models.generate_content(
            model=model_name,
            contents="Say 'Hello' in JSON format: {\"message\": \"...\"}",
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=100,
                response_mime_type="application/json",
            ),
        )

        if response.text:
            print(f"✅ 成功! 响应: {response.text[:100]}")
            print(f"✅ 推荐使用模型: {model_name}")
            break
        else:
            print(f"⚠️ 响应为空")
    except Exception as e:
        print(f"❌ 失败: {e}")

print("\n" + "="*60)
print("测试完成")
