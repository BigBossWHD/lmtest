
import argparse, json, time, re, random, os
from typing import List, Dict, Any, Optional
import requests
from urllib.parse import urlparse

# ---------------
# Config defaults
# ---------------
DEFAULT_BASE_URL = "http://localhost:1234/v1"   # Local LM Studio default
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_SEED = 42
TIMEOUT = 180
RESULTS_DIR = "results"
JUDGE_BASE_URL_ENV = "JUDGE_BASE_URL"
JUDGE_API_KEY_ENV = "JUDGE_API_KEY"
JUDGE_DEFAULT_MODEL = os.getenv("JUDGE_MODEL", "deepseek-chat")


SYSTEM_PROMPT = """You are an expert problem solver for programming and math contest questions.
Follow these rules:
- Think briefly if needed, but keep explanations short.
- The final line must be exactly: Answer: <final text>
- The final text should match the required output format (it may be a number, string, list, etc.).
- If unsure, still provide your best attempt in that format.
- The final text must exactly match the required output format (language, casing, spacing) described or implied by the problem.
"""

USER_PROMPT_TEMPLATE = """Problem:
{question}

Please respond with the final output on its own line in the form `Answer: <content>`.
"""

def call_openai_compatible_chat(base_url: str, model: str, messages: List[Dict[str, str]],
                                temperature: float = 0.0, top_p: float = 1.0, seed: Optional[int] = None,
                                max_tokens: Optional[int] = 8192, api_key: Optional[str] = None) -> Dict[str, str]:
    """Calls an OpenAI-compatible /chat/completions endpoint (LM Studio)."""
    url = f"{base_url}/chat/completions"
    auth_token = api_key or "lm-studio"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}"   # LM Studio ignores the token value but expects the header
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if seed is not None:
        payload["seed"] = seed
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    parsed = urlparse(base_url)
    proxies = None
    if parsed.hostname in {"localhost", "127.0.0.1"}:
        proxies = {"http": None, "https": None}

    resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT, proxies=proxies)
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or ""
    text = content.strip() or reasoning.strip()
    return {
        "text": text,
        "content": content,
        "reasoning": reasoning,
        "finish_reason": data["choices"][0].get("finish_reason"),
    }

def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"Answer\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return None

def normalize_gold(answer: Any) -> List[str]:
    """Normalize ground-truth answers into a list of strings."""
    if isinstance(answer, (list, tuple, set)):
        return [str(x).strip() for x in answer]
    if isinstance(answer, (int, float)):
        return [str(answer).strip()]
    if isinstance(answer, str):
        stripped = answer.strip()
        if not stripped:
            return []
        return [stripped]
    return []

def format_reference_answers(answers: List[str]) -> str:
    if not answers:
        return "无参考答案"
    if len(answers) == 1:
        return answers[0]
    return "；".join(answers)


def build_judge_prompt(question: str, references: List[str], model_answer: Optional[str]) -> List[Dict[str, str]]:
    references_text = format_reference_answers(references)
    model_text = model_answer if model_answer is not None else "(模型未给出答案)"
    user_content = (
        "请作为严谨的评审，根据题目与参考答案判断模型回答是否等价。"
        "只根据给定信息判断，不要自行推理新的结论。"
        f"\n题目：{question}"
        f"\n参考答案：{references_text}"
        f"\n模型答案：{model_text}"
        "\n请输出 JSON，格式为 {\\\"verdict\\\": \\\"correct\\\" 或 \\\"incorrect\\\", \\\"explanation\\\": \\\"原因\\\"}，不要包含其它文本或代码块。"
    )
    return [
        {"role": "system", "content": "你是严格的评审员，负责判断模型答案是否与参考答案一致。"},
        {"role": "user", "content": user_content},
    ]


def parse_judge_response(text: str) -> Dict[str, Optional[str]]:
    if not text:
        return {"verdict": None, "explanation": None}
    stripped = text.strip().strip('`')
    try:
        data = json.loads(stripped)
        verdict = data.get("verdict")
        explanation = data.get("explanation")
        if verdict:
            verdict = verdict.strip()
        if explanation:
            explanation = explanation.strip()
        return {"verdict": verdict, "explanation": explanation}
    except (json.JSONDecodeError, AttributeError):
        return {"verdict": None, "explanation": stripped}


def call_judge(question: str, references: List[str], model_answer: Optional[str], config: Dict[str, str]) -> Dict[str, Optional[str]]:
    messages = build_judge_prompt(question, references, model_answer)
    try:
        resp = call_openai_compatible_chat(
            config["base_url"],
            config["model"],
            messages,
            temperature=0.0,
            top_p=1.0,
            seed=None,
            max_tokens=1024,
            api_key=config["api_key"],
        )
    except Exception as exc:
        return {"verdict": None, "explanation": f"评审调用失败: {exc}"}
    return parse_judge_response(resp.get("content") or resp.get("text") or "")

def run_eval(dataset_path: str, base_url: str, model: str, n_samples: int = 3,
             temperature_cold: float = 0.0, temperature_hot: float = 0.7,
             seed: int = DEFAULT_SEED, api_key: Optional[str] = None,
             show_details: bool = True,
             judge_config: Optional[Dict[str, str]] = None,
             max_tokens: Optional[int] = 8192) -> Dict[str, Any]:
    """
    For each problem, generate n_samples (1 cold + hot samples),
    take majority vote among parsed answers; ties prefer the cold sample, otherwise lexicographically smallest.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    problems = data["problems"]
    random.seed(seed)
    results = []
    judge_total = 0
    judge_correct = 0

    for idx, item in enumerate(problems, 1):
        q = item["question"].strip()
        gold_list = normalize_gold(item.get("answer"))
        messages_sys = {"role": "system", "content": SYSTEM_PROMPT}
        messages_user = {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=q)}

        preds = []

        # Cold sample (deterministic)
        pred_cold = None
        try:
            out_cold = call_openai_compatible_chat(base_url, model, [messages_sys, messages_user],
                                                   temperature=temperature_cold, seed=seed, api_key=api_key,
                                                   max_tokens=max_tokens)
        except Exception as e:
            print(f"[{idx:02d}] cold sample error: {e}")
            out_cold = None
        else:
            pred_cold = extract_answer(out_cold.get("text", ""))
            if pred_cold is not None:
                preds.append(pred_cold)
            else:
                snippet = out_cold.get("text", "").replace("\n", " ")
                if len(snippet) > 120:
                    snippet = snippet[:117] + "..."
                print(f"[{idx:02d}] unable to parse cold response -> {snippet}")

        # Hot samples
        for _ in range(max(0, n_samples - 1)):
            pred_hot = None
            try:
                out_hot = call_openai_compatible_chat(base_url, model, [messages_sys, messages_user],
                                                      temperature=temperature_hot, seed=random.randint(0, 10**9),
                                                      api_key=api_key, max_tokens=max_tokens)
            except Exception as e:
                print(f"[{idx:02d}] hot sample error: {e}")
                continue
            pred_hot = extract_answer(out_hot.get("text", ""))
            if pred_hot is not None:
                preds.append(pred_hot)
            else:
                snippet = out_hot.get("text", "").replace("\n", " ")
                if len(snippet) > 120:
                    snippet = snippet[:117] + "..."
                print(f"[{idx:02d}] unable to parse hot response -> {snippet}")

        # Majority vote
        final_pred = None
        if preds:
            # Count votes
            counts = {}
            for p in preds:
                counts[p] = counts.get(p, 0) + 1
            max_vote = max(counts.values())
            cands = [k for k,v in counts.items() if v == max_vote]
            if len(cands) == 1:
                final_pred = cands[0]
            else:
                # tie-breaking: prefer cold if present
                if pred_cold in cands and pred_cold is not None:
                    final_pred = pred_cold
                else:
                    final_pred = min(cands)  # deterministic tie-break

        judge_feedback_entry = None
        if judge_config and final_pred is not None and gold_list:
            judge_feedback_raw = call_judge(q, gold_list, final_pred, judge_config)
            verdict = judge_feedback_raw.get("verdict") if judge_feedback_raw else None
            explanation = judge_feedback_raw.get("explanation") if judge_feedback_raw else None
            judge_feedback_entry = {
                "verdict": verdict,
                "verdict_list": [] if not verdict else [verdict],
                "explanation": explanation,
                "explanation_list": [] if not explanation else [explanation],
            }
            if verdict in {"correct", "incorrect"}:
                judge_total += 1
                if verdict == "correct":
                    judge_correct += 1

        results.append({
            "index": idx,
            "question": q,
            "model_answer": final_pred,
            "reference_answers": gold_list,
            "judge_feedback": judge_feedback_entry,
        })
        if show_details:
            pred_repr = [] if final_pred is None else [final_pred]
            question_preview = q if len(q) <= 120 else q[:117] + "..."
            print(f"[{idx:02d}] 题目：{question_preview}")
            print(f"    参考答案={gold_list}")
            print(f"    模型答案={pred_repr}")
            if judge_feedback_entry:
                verdict_repr = judge_feedback_entry.get("verdict_list", [])
                explanation_repr = judge_feedback_entry.get("explanation_list", [])
                print(f"    评审判定={verdict_repr} 说明={explanation_repr}")

    total = len(results)

    summary = {
        "num_questions": total,
        "model": model,
        "judge_total": judge_total if judge_config else None,
        "judge_correct": judge_correct if judge_config else None,
        "judge_accuracy_percent": round(judge_correct / judge_total * 100, 2) if judge_config and judge_total else None,
        "details": results
    }
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, help="Path to the contest dataset JSON (list of problems).")
    ap.add_argument("--base_url", type=str, default=None, help="OpenAI-compatible base URL (overrides config/default).")
    ap.add_argument("--model", type=str, default=None, help="Model name (overrides config/default).")
    ap.add_argument("--samples", type=int, default=3, help="Total generations per problem (>=1).")
    ap.add_argument("--hot_temp", type=float, default=0.7, help="Temperature for hot samples (>= 0.0).")
    ap.add_argument("--api_key", type=str, default=None, help="Optional bearer token for Authorization header.")
    ap.add_argument("--max_tokens", type=int, default=8192, help="生成上限（默认 8192，可按需调大/调小，设为 0 则不发送该字段）。")
    ap.add_argument("--config", type=str, default=None, help="Optional JSON file providing base_url/model/api_key defaults.")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-question console output.")
    ap.add_argument("--verbose", action="store_true", help="After running, print the full JSON summary to stdout.")
    ap.add_argument("--enable_judge", action="store_true", help="Use a secondary模型评审模型（需配置 DeepSeek 端点和密钥）。")
    ap.add_argument("--judge_base_url", type=str, default=None, help="Overrides JUDGE_BASE_URL for judge model.")
    ap.add_argument("--judge_model", type=str, default=None, help="Overrides默认评审模型名称 (环境变量 JUDGE_MODEL 或 default deepseek-chat)。")
    ap.add_argument("--judge_api_key", type=str, default=None, help="Overrides JUDGE_API_KEY for judge model.")
    args = ap.parse_args()

    assert args.samples >= 1, "--samples must be >= 1"
    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)

    base_url = args.base_url or config.get("base_url") or DEFAULT_BASE_URL
    model = args.model or config.get("model") or DEFAULT_MODEL
    api_key = args.api_key or config.get("api_key")

    show_details = not args.quiet

    judge_config = None
    if args.enable_judge:
        judge_base = args.judge_base_url or os.getenv(JUDGE_BASE_URL_ENV)
        judge_api_key = args.judge_api_key or os.getenv(JUDGE_API_KEY_ENV)
        judge_model = args.judge_model or JUDGE_DEFAULT_MODEL
        if not judge_base or not judge_api_key:
            raise RuntimeError("启用评审需要提供 judge_base_url 和 judge_api_key，或设置环境变量 JUDGE_BASE_URL/JUDGE_API_KEY。")
        judge_config = {
            "base_url": judge_base.rstrip('/'),
            "model": judge_model,
            "api_key": judge_api_key,
        }

    max_tokens = args.max_tokens
    if max_tokens == 0:
        max_tokens = None

    summary = run_eval(args.dataset, base_url, model, n_samples=args.samples,
                       temperature_hot=args.hot_temp, api_key=api_key,
                       show_details=show_details, judge_config=judge_config,
                       max_tokens=max_tokens)

    # Save report
    ts = int(time.time())
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"eval_report_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    total = summary["num_questions"]
    print("\n=== 统计结果 ===")
    print(f"题目总数：{total}")
    print(f"被测模型：{model}")
    if judge_config:
        judge_total = summary.get("judge_total") or 0
        judge_correct = summary.get("judge_correct") or 0
        if judge_total:
            accuracy = summary.get("judge_accuracy_percent") or 0.0
            print(f"评审判定：{judge_correct}/{judge_total}（准确率 {accuracy:.2f}%）")
        else:
            print("评审模型已启用但未返回有效判定，请检查评审响应。")
        print("请结合评审判定与参考答案进行复核。")
    else:
        print("本工具不自动判定对错，请对照“模型答案”和“参考答案”手动复核。")
    print(f"报告路径：{out_path}")
    if args.verbose:
        print("\nDetailed summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
