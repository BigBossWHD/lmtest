# Contest Model Evaluator

本项目提供一套简单的评测脚本，帮助你在本地或远程 OpenAI 兼容端点上批量测试数学/竞赛类题集。通过自定义数据集与配置，你可以快速对模型进行回归测试、比较不同推理策略或跟踪调优效果。

## 快速上手
1. **准备环境**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install requests  # 若尚未安装
   ```
2. **确认推理端点可用**
   - 本地：例如 LM Studio 默认 `http://localhost:1234/v1`
   - 远程：任意兼容 OpenAI Chat Completions 协议的服务，需准备 base URL、模型名称与 API Key
3. **创建数据集**
   ```bash
   cp datasets/dataset_template.json datasets/aime_2025_I.json
   ```
   - 在复制出的文件中，将题干填入 `question`；把题目期望的输出写入 `answer`（以字符串保存，例如 `"512"`、`"[0, 2, 4]"`、`"True"`）
   - 答案比较基于精确字符串匹配（区分大小写与空格），必要时可在数据集中保留前导零或括号等格式，并确保提示能够让模型生成该格式
   - 题目数量不限，只需保证 `problems` 列表中每个对象含有 `question` 与 `answer`
4. **运行评测**

   完整示例（启用评审模型）：
   ```bash
   export JUDGE_BASE_URL="https://api.deepseek.com/v1"
   export JUDGE_API_KEY="<your-key>"
   export JUDGE_MODEL="deepseek-chat"
   export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
   export DEEPSEEK_API_KEY="<your-key>"

   python contest_eval_runner.py --samples 1 --hot_temp 0.7 --enable_judge \
     --base_url "$DEEPSEEK_BASE_URL" --api_key "$DEEPSEEK_API_KEY" \
     --model "deepseek-chat" --dataset datasets/logic_test.json
   ```
   参数说明：
   - `--dataset`：数据集文件路径，需符合模板结构。
   - `--base_url` / `--api_key`：被测模型端点与鉴权，可直接使用环境变量。
   - `--model`：被测模型名称，需与端点支持一致。
   - `--samples`：总生成次数（默认 1 次低温 + 若干高温样本），用于投票或多样性。
   - `--hot_temp`：高温样本温度，配合 `--samples` 调节生成差异。
   - `--enable_judge`：启用评审模型；默认读取 `JUDGE_BASE_URL`、`JUDGE_API_KEY`、`JUDGE_MODEL`，也可用 `--judge_*` 覆盖。
   - `--config`：可选 JSON 配置（字段 `base_url`、`model`、`api_key`），命令行参数优先级更高。
   - `--quiet`：仅输出最终汇总提示（默认逐题打印题干/答案）。
   - `--verbose`：运行结束后打印完整 JSON 摘要。

   更多选项可查看帮助：
   ```bash
   python contest_eval_runner.py -h
   ```

默认情况下终端会逐题展示题干预览、模型答案、参考答案；使用 `--quiet` 可仅保留最终提示。评测完成后会生成 `results/eval_report_<timestamp>.json`，如需在命令行查看完整 JSON，可附加 `--verbose`。若启用评审模型，命令行及报告还会包含评审判定及整体准确率指标。

## 目录结构
- `contest_eval_runner.py`：主脚本，负责构造提示、调用端点、解析答案、汇总投票
- `datasets/dataset_template.json`：数据集模板，可复制后填入任意竞赛题
- `datasets/`：存放各类评测集（例如 `aime_2025_I.json`）
- `results/`：评测输出目录，保存带详细日志的 JSON 报告
- `AGENTS.md`：贡献者指南（编码规范、测试策略、配置约定）
- `samples/`（可选）：轻量级样本数据，用于快速冒烟测试
- `tests/`（可选）：推荐放置解析与投票逻辑的单元测试

## 报告内容
`results/eval_report_<timestamp>.json` 包含：
- `num_questions`：题目数量；目前不提供自动正确率，需人工复核
- `details` 列表（逐题记录）：
  - `question` 原题题干
  - `model_answer` 聚合后的模型答案
  - `reference_answers` 数据集提供的参考答案列表（字符串）
  - `judge_feedback`（可选）评审模型返回的判定/说明
- `judge_total` / `judge_correct` / `judge_accuracy_percent`（仅启用评审时存在）

## 常见用法
- **冒烟测试**（默认逐题打印，如需更安静可加 `--quiet`）：
  ```bash
  python3 contest_eval_runner.py --dataset samples/contest_sample.json --model "openai/gpt-oss-20b" --samples 1
  ```
- **对比不同端点**：准备多份配置文件（如 `configs/local.json`、`configs/cloud.json`），通过 `--config` + 命令行覆盖快速切换
- **扩展题型**：答案可以是任意字符串（包含数字、列表、布尔值等）；如需特殊格式，可扩展 `extract_answer` 与 `normalize_gold`

- **启用评审模型（可选）**：
  ```bash
  export JUDGE_BASE_URL="https://api.deepseek.com/v1"
  export JUDGE_API_KEY="<your-key>"
  export JUDGE_MODEL="deepseek-chat"
  python3 contest_eval_runner.py --dataset datasets/aime_2025_I.json --enable_judge --samples 1
  ```
  默认读取 `JUDGE_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL`，可用 `--judge_base_url`、`--judge_model`、`--judge_api_key` 覆盖。评审仅提供参考，请结合参考答案自行判断。

## 故障排查
- **模型只输出推理无结论**：强化提示，确保末行必须写成 `Answer: ...`；或调整脚本解析逻辑
- **请求失败或无响应**：检查端点、API Key 与网络代理；脚本会自动禁用 `localhost` 的代理环境变量
- **所有题目相同答案**：确认模型已正确加载；可在端点控制台直接发送同样的 system/user prompt 验证

## 贡献指南
- 代码风格：PEP 8，模块常量使用 `UPPER_SNAKE_CASE`
- 测试：新增解析/投票逻辑时，请在 `tests/` 编写单元测试并运行 `python3 -m pytest`
- 提交：使用祈使句提交信息，PR 中注明数据来源、端点配置与代表性评测结果

欢迎提交 Issue/PR 反馈改进需求或新的题库支持。
