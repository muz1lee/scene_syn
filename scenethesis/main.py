from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from scenethesis.modules.planner import CoarseScenePlanner
from scenethesis.services.providers import LLMConfig, GeminiProvider

CONFIG_PATH = Path("config.yaml")


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件 {path} 不存在，请创建 config.yaml")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_plan(plan_json: str, output_dir: str | Path | None) -> Path:
    base_dir = Path(__file__).resolve().parent
    target_dir = Path(output_dir) if output_dir else base_dir / "output"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = "planner_output.json"
    file_path = target_dir / filename
    file_path.write_text(plan_json, encoding="utf-8")
    return file_path


def run_scenethesis_system(config: Dict[str, Any]) -> None:
    prompt = config.get("prompt", "A simple room")
    assets = config.get("db_assets", [])
    model_name = config.get("model_name", "gemini-3-flash")
    output_dir = config.get("output_dir")

    if not assets:
        raise ValueError("配置文件中缺少 db_assets 列表")

    print("🚀 [主循环] 启动 Scenethesis Planner 单元测试...")
    llm_config = LLMConfig(model=model_name)
    llm_provider = GeminiProvider(config=llm_config)
    planner = CoarseScenePlanner(assets, llm_provider)

    plan = planner.run_pipeline(prompt)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    print("✅ [结果] 粗级规划输出：")
    print(plan_json)

    output_path = _save_plan(plan_json, output_dir)
    print(f"💾 [保存] 规划结果已写入: {output_path}")


if __name__ == "__main__":
    cfg = load_config()
    run_scenethesis_system(cfg)
