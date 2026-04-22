import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


DEFAULT_SAVE_INTERVAL = 5


WENYAN_SYSTEM_PROMPT = """
你是一名严谨的中文语义评估助手，专门评估两段文本在“信息一致性与语义相似度”上的匹配程度。

任务：
比较【标准文本】与【模型文本】是否在以下三部分上保持一致或高度相似：
1）通俗白话文翻译
2）文化内涵/价值观解释（核心）
3）现代应用解读（如有）

评估标准（0-2分）：
- 2分：核心信息一致，三部分大体对应，允许表述方式不同但语义基本等价。
- 1分：部分一致，抓住了主要意思，但有明显遗漏、偏差或某一部分不充分。
- 0分：不一致或偏差较大，核心含义错误、错位，或大量内容不对应。

输出要求（必须严格按以下格式）：
只输出一个数字：0 或 1 或 2。不要输出任何解释或其他文字。
""".strip()

BASE_EVAL_RULE = """
你是一名严谨的中文语义评估助手，专门评估两段文本在“信息一致性与语义相似度”上的匹配程度。
请结合当前任务的评估重点进行打分。

评分标准（0-2分）：
- 2分：核心信息一致，关键要点基本覆盖，表达可不同但语义等价。
- 1分：部分一致，抓住主要意思，但有明显遗漏、偏差或不充分。
- 0分：不一致或偏差较大，核心含义错误、错位，或大量内容不对应。

输出要求（必须严格按以下格式）：
只输出一个数字：0 或 1 或 2。不要输出任何解释或其他文字。
""".strip()

FILE_FOCUS = {
    "唐诗宋词元曲.csv": "比较【标准文本】与【模型文本】是否在以下三部分上保持一致或高度相似：\n1）字面意思\n2）核心意象\n3）作者情感",
    "四大名著.csv": "比较【标准文本】与【模型文本】是否在以下三部分上保持一致或高度相似：\n1）情节/对话核心\n2）人物意图\n3）桥段背后的中式文化内涵",
    "古典小说戏曲.csv": "比较【标准文本】与【模型文本】是否在以下三部分上保持一致或高度相似：\n1）情节/对话核心\n2）人物意图\n3）桥段背后的中式文化内涵",
    "民俗文化.csv": "比较【标准文本】与【模型文本】是否在以下三部分上保持一致或高度相似：\n1）起源与核心内涵\n2）背后的中式民间价值观\n3）现代应用场景与文化延续",
    "中式美学哲思.csv": "比较【标准文本】与【模型文本】是否在以下三部分上保持一致或高度相似：\n1）通俗定义\n2）中式价值观/审美追求\n3）具体中式案例",
    "中式现代梗.csv": "比较【标准文本】与【模型文本】是否在以下两部分上保持一致或高度相似：\n1）表层含义与真实意图/内涵\n2）使用场景与表达效果",
}


def display_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    try:
        return os.path.relpath(abs_path, start=os.getcwd())
    except ValueError:
        return os.path.basename(abs_path)


def build_client(api_key: Optional[str], base_url: Optional[str]) -> OpenAI:
    final_key = (api_key or os.getenv("EVAL_API_KEY") or "").strip()
    if not final_key:
        raise ValueError("未提供 API Key。请通过 --api-key 或环境变量 EVAL_API_KEY 设置。")
    if base_url:
        return OpenAI(api_key=final_key, base_url=base_url)
    return OpenAI(api_key=final_key)


def resolve_base_dir(base_dir_arg: Optional[str]) -> str:
    return os.path.abspath(base_dir_arg) if base_dir_arg else os.path.dirname(os.path.abspath(__file__))


def normalize_source_filename(result_filename: str, model_name: str) -> str:
    suffix = f"_{model_name}.csv"
    if result_filename.endswith(suffix):
        return result_filename[: -len(suffix)] + ".csv"
    return result_filename


def build_system_prompt(focus: str) -> str:
    return f"{BASE_EVAL_RULE}\n\n当前任务评估重点：{focus}"


def detect_standard_col(headers: List[str], source_filename: str, model_col: str) -> Optional[str]:
    standard_col = "真实意图" if source_filename.startswith("对话") else "解读内容"
    if standard_col in headers and standard_col != model_col:
        return standard_col
    return None


def parse_eval_score(content: str) -> str:
    score_match = re.search(r"分数[：:]\s*\[?([0-2])\]?", content)
    if score_match:
        return score_match.group(1)

    plain_match = re.search(r"\b([0-2])\b", content.strip())
    if plain_match:
        return plain_match.group(1)

    stripped = content.strip()
    return stripped if stripped in {"0", "1", "2"} else ""


def get_eval_wenyan(
    client: OpenAI,
    eval_model: str,
    standard_text: str,
    model_text: str,
) -> Optional[str]:
    if not standard_text or not model_text:
        return None

    user_prompt = f"""【标准文本】
{standard_text}

【模型文本】
{model_text}

请只输出分数（0/1/2）。"""

    try:
        response = client.chat.completions.create(
            model=eval_model,
            messages=[
                {"role": "system", "content": WENYAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        content = response.choices[0].message.content if hasattr(response.choices[0].message, "content") else str(response.choices[0].message)
        score = parse_eval_score(content)
        return score or None
    except Exception as e:
        print(f"评估调用失败: {e}")
        return None


def get_eval_culture(
    client: OpenAI,
    eval_model: str,
    dialogue: str,
    truth: str,
    prediction: str,
) -> Optional[str]:
    if not dialogue or not truth or not prediction:
        return None

    prompt = f"""
你是一名中国文化沟通与语言学专家。请评估 AI 生成的“意图分析”与“标准真实意图”的一致性。

【输入数据】
1. 对话场景/内容：{dialogue}
2. 标准参考意图（人工标注）：{truth}
3. AI生成的分析：{prediction}

【评估步骤】
1. 首先，从“AI生成的分析”中提取出其对【真实意图】的描述（忽略表面意图部分）。
2. 将 AI 提取的真实意图与“标准参考意图”进行语义对比。
3. 重点考察：
   - 核心动机是否一致（例如：是想拒绝、想要钱、还是在客套）？
   - 情感色彩是否一致（例如：是不满、委婉、还是讽刺）？
   - 是否捕捉到了关键的潜台词？

【打分标准】
- 2分（准确）：精准捕捉到了核心的真实意图与情感倾向，理解无误。
- 1分（一般）：抓住了主要意图，但漏掉了语气细节，或表述不够精准。
- 0分（错误）：完全误解了真实意图，或仅停留在表面意思，未识别出潜台词。

【输出格式】
只输出一个数字：0 或 1 或 2。不要输出任何解释或其他文字。
"""

    try:
        response = client.chat.completions.create(
            model=eval_model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        content = response.choices[0].message.content if hasattr(response.choices[0].message, "content") else str(response.choices[0].message)
        score = parse_eval_score(content)
        return score or None
    except Exception as e:
        print(f"评估调用失败: {e}")
        return None


def get_eval_general(
    client: OpenAI,
    eval_model: str,
    focus: str,
    standard_text: str,
    model_text: str,
) -> Optional[str]:
    if not standard_text or not model_text:
        return None

    system_prompt = build_system_prompt(focus)
    user_prompt = f"""【标准文本】
{standard_text}

【模型文本】
{model_text}

请只输出分数（0/1/2）。"""

    try:
        response = client.chat.completions.create(
            model=eval_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.1,
        )
        content = response.choices[0].message.content if hasattr(response.choices[0].message, "content") else str(response.choices[0].message)
        score = parse_eval_score(content)
        return score or None
    except Exception as e:
        print(f"评估调用失败: {e}")
        return None


def save_csv(path: str, headers: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def process_one_file(
    client: OpenAI,
    abs_path: str,
    model_dir_name: str,
    source_filename: str,
    eval_model: str,
    save_interval: int,
    overwrite: bool,
    max_evals: int = 0,
) -> int:
    with open(abs_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
            rows = list(reader)
        except StopIteration:
            print(f"[跳过] 空文件: {display_path(abs_path)}")
            return 0

    model_col = model_dir_name
    if model_col not in headers:
        print(f"[跳过] 缺少模型列 {model_col}: {display_path(abs_path)}")
        return 0

    standard_col = detect_standard_col(headers, source_filename, model_col)
    if not standard_col:
        print(f"[跳过] 无法识别标准答案列: {display_path(abs_path)}")
        return 0

    eval_score_col = f"eval_score_{model_dir_name}"

    if eval_score_col not in headers:
        headers.append(eval_score_col)

    idx = {h: i for i, h in enumerate(headers)}
    std_idx = idx[standard_col]
    mdl_idx = idx[model_col]
    score_idx = idx[eval_score_col]

    print(f"\n=== 评估文件: {display_path(abs_path)} ===")
    print(f"标准列: {standard_col} | 模型列: {model_col}")

    done = 0
    for i, row in enumerate(rows):
        while len(row) < len(headers):
            row.append("")

        standard_text = row[std_idx].strip() if std_idx < len(row) else ""
        model_text = row[mdl_idx].strip() if mdl_idx < len(row) else ""
        current_score = row[score_idx].strip() if score_idx < len(row) else ""

        if not standard_text or not model_text:
            continue
        if current_score and not overwrite:
            continue

        print(f"[Row {i+2}] 评估中...")

        if source_filename == "文言文_final_version.csv":
            score = get_eval_wenyan(client, eval_model, standard_text, model_text)
        elif source_filename == "对话.csv":
            dialogue_text = row[idx["对话内容"]].strip() if "对话内容" in idx and idx["对话内容"] < len(row) else ""
            score = get_eval_culture(client, eval_model, dialogue_text, standard_text, model_text)
        else:
            focus = FILE_FOCUS.get(source_filename)
            if not focus:
                print(f"[跳过] 未配置评估重点: {source_filename}")
                continue
            score = get_eval_general(client, eval_model, focus, standard_text, model_text)

        if score:
            row[score_idx] = score
            done += 1
            if done % save_interval == 0:
                save_csv(abs_path, headers, rows)
                print(f"  -> 已保存进度: {display_path(abs_path)}")
            if max_evals > 0 and done >= max_evals:
                break

    save_csv(abs_path, headers, rows)
    print(f"完成: {os.path.basename(abs_path)}，新增/更新评估 {done} 条。")
    return done


def iter_target_csv_files(results_dir: str, excluded_model_dirs: List[str]) -> List[Tuple[str, str, str]]:
    targets: List[Tuple[str, str, str]] = []

    for model_dir_name in sorted(os.listdir(results_dir)):
        if model_dir_name in excluded_model_dirs:
            continue

        model_dir_path = os.path.join(results_dir, model_dir_name)
        if not os.path.isdir(model_dir_path):
            continue

        for root, _, files in os.walk(model_dir_path):
            for filename in files:
                if not filename.lower().endswith(".csv"):
                    continue
                abs_path = os.path.join(root, filename)
                source_filename = normalize_source_filename(filename, model_dir_name)
                targets.append((abs_path, model_dir_name, source_filename))

    return targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="统一评估指定 results 目录下的多模型输出。")
    p.add_argument("--base-dir", default=None, help="experiment2 目录，默认当前脚本目录")
    p.add_argument("--api-key", default=None, help="评估模型 API Key")
    p.add_argument("--base-url", default=None, help="OpenAI 兼容接口地址。默认读取环境变量 EVAL_API_BASE_URL")
    p.add_argument("--eval-model", default=None, help="评估模型名。默认读取环境变量 EVAL_MODEL_NAME")
    p.add_argument("--exclude-model-dir", action="append", default=[], help="需要跳过的模型结果目录名，可重复传入")
    p.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL, help="每 N 条保存一次")
    p.add_argument("--overwrite", action="store_true", help="是否覆盖已有 eval_score_<模型目录名>")
    p.add_argument("--max-evals", type=int, default=0, help="最多评估多少条（全局）。0 表示不限制")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    results_dir = os.path.join(base_dir, "results")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"results 目录不存在: {display_path(results_dir)}")

    base_url = args.base_url or os.getenv("EVAL_API_BASE_URL")
    eval_model = args.eval_model or os.getenv("EVAL_MODEL_NAME")
    if not eval_model:
        raise ValueError("未提供评估模型名。请通过 --eval-model 或环境变量 EVAL_MODEL_NAME 设置。")

    client = build_client(args.api_key, base_url)

    targets = iter_target_csv_files(results_dir, args.exclude_model_dir)
    if not targets:
        print("未发现可评估 CSV 文件。")
        return

    print(f"共发现待处理文件: {len(targets)}")
    total_done = 0
    for abs_path, model_dir_name, source_filename in targets:
        remaining = 0
        if args.max_evals > 0:
            remaining = args.max_evals - total_done
            if remaining <= 0:
                print(f"达到评估上限 {args.max_evals}，提前结束。")
                break

        done = process_one_file(
            client=client,
            abs_path=abs_path,
            model_dir_name=model_dir_name,
            source_filename=source_filename,
            eval_model=eval_model,
            save_interval=args.save_interval,
            overwrite=args.overwrite,
            max_evals=remaining,
        )
        total_done += done

    print(f"\n全部评估完成。本次新增/更新评估总数: {total_done}")


if __name__ == "__main__":
    main()
