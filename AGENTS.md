# Repository Guidelines

## Project Structure & Module Organization
- `contest_eval_runner.py` 驱动评测：加载数据集、向 OpenAI 兼容端点发送请求、聚合样本并将报告写入 `results/`。
- `datasets/dataset_template.json` 是通用模板；复制后填入题干/答案即可生成新评测集，允许保留 AIME 等原始命名。
- 建议将实验脚本或笔记本放在 `scratch/`，保持仓库根目录干净。

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate` 建立隔离环境。
- `python3 -m pip install -r requirements.txt`（如存在）或 `python3 -m pip install requests` 安装依赖。
- `python3 contest_eval_runner.py --dataset samples/contest_sample.json --samples 1` 进行冒烟测试，可用 `--base_url` / `--model` / `--api_key` 覆盖配置；默认 `max_tokens` 为 8192，可按需调整。

- 若需引入评审模型，可设置 `JUDGE_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL` 并使用 `--enable_judge`（未设置时默认模型为 `deepseek-chat`），运行后摘要会给出评审判定统计。
- 如需多套环境，编写 JSON（如 `configs/staging.json`）并使用 `--config configs/staging.json` 复用。
- JSON 报告包含被测模型名称（`model` 字段）；请在 PR 摘要中注明使用的模型与参数。

## Coding Style & Naming Conventions
- 遵循 PEP 8：4 空格缩进、snake_case 函数变量、CapWords 类名、常量使用 UPPER_SNAKE_CASE。
- 延续现有 typing 风格，使用显式 `Dict`/`List`/`Optional`，如需结构化字段可考虑 `TypedDict`。
- 仅在逻辑复杂处添加注释，避免复述代码表面含义。

## Testing Guidelines
- 单元测试置于 `tests/`（例如 `tests/test_runner_flow.py`），运行 `python3 -m pytest`。
- 复制 `datasets/dataset_template.json` 至 `tests/fixtures/` 填充极端案例验证解析/投票逻辑。
- 回归评估：运行 `python3 contest_eval_runner.py --dataset datasets/aime_2025_I.json --model "openai/gpt-oss-20b" --samples 2`，比对 `results/eval_report_*.json`。

## Commit & Pull Request Guidelines
- 提交消息用祈使句（如 `Add dataset loader hooks`），主题控制在 50 字符左右，必要时写详细描述。
- PR 需说明数据来源、端点/模型/温度配置，并附代表性输出或指标。
- 关联相关 issue / 讨论，标注与既有评测流程的差异。

## Security & Configuration Tips
- 默认端点为 `http://localhost:1234/v1`；可通过命令行或 JSON 配置覆盖。
- 命令行参数与 `--config` JSON 是推荐配置方式；仓库不支持环境变量回退。
- 不要硬编码密钥；放在外部配置文件或本地安全存储，并确保 README 与脚本参数保持一致。
