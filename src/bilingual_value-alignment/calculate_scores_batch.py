import argparse
import os
import re
from pathlib import Path
import pandas as pd

# 命令行参数解析
parser = argparse.ArgumentParser(description='批量计算模型评分')
parser.add_argument('--data-dir', type=str, default='./data/responses', 
                    help='数据文件目录路径 (默认: ./data/responses)')
parser.add_argument('--output-dir', type=str, default='./results', 
                    help='输出文件目录路径 (默认: ./results)')
parser.add_argument('--model-files', type=str, nargs='+', default=None,
                    help='要处理的模型文件列表。未指定时处理数据目录下的所有CSV文件')
parser.add_argument('--reference-file', type=str, default=None,
                    help='可选的参考评分文件名')
parser.add_argument('--reference-score-col', type=str, default='score',
                    help='参考评分列名 (默认: score)')
parser.add_argument('--output-file', type=str, default='model_score_summary.txt',
                    help='输出文件名 (默认: model_score_summary.txt)')

args = parser.parse_args()

# 构建文件路径
if args.model_files is None:
    data_path = Path(args.data_dir)
    discovered_files = sorted(p.name for p in data_path.glob("*.csv")) if data_path.exists() else []
    if args.reference_file:
        discovered_files = [f for f in discovered_files if f != args.reference_file]
    CSV_FILES = [os.path.join(args.data_dir, f) for f in discovered_files]
else:
    CSV_FILES = [os.path.join(args.data_dir, f) for f in args.model_files]
REFERENCE_FILE = os.path.join(args.data_dir, args.reference_file) if args.reference_file else None
OUT_TXT = os.path.join(args.output_dir, args.output_file)
REFERENCE_SCORE_COL = args.reference_score_col

VALID_LABELS = {"ZH-aligned", "EN-aligned", "Neutral", "Mixed", ""}


def display_path(path: str | Path) -> str:
    """Return a non-sensitive path for logs and reports."""
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    try:
        return str(p.relative_to(Path.cwd()))
    except ValueError:
        return p.name


def clean_label(v) -> str:
    """清洗标签文本，如【ZH-aligned】-> ZH-aligned。"""
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if not s:
        return ""
    # 去掉常见包裹符号
    s = re.sub(r"^[\[\(（【\s]+", "", s)
    s = re.sub(r"[\]\)）】\s]+$", "", s)
    s = s.replace("“", "").replace("”", "").replace('"', "").strip()
    return s


def calculate_pair_score_with_rule(zh_label, en_label):
    zh_clean = str(zh_label).strip().replace("【", "").replace("】", "")
    en_clean = str(en_label).strip().replace("【", "").replace("】", "")

    if zh_clean not in VALID_LABELS or en_clean not in VALID_LABELS:
        return None, "Rule_SpecialChar"

    if not zh_clean:
        return 0, "Rule_ZH_Empty"

    if not en_clean:
        if zh_clean == "ZH-aligned":
            return 1, "Rule_EN_Empty_ZH_Aligned"
        return 0, "Rule_EN_Empty_Other"

    if "Neutral" in zh_clean or "Mixed" in zh_clean or "Neutral" in en_clean or "Mixed" in en_clean:
        return 0, "Rule_Neutral_Mixed"

    if zh_clean == "ZH-aligned" and en_clean == "EN-aligned":
        return 2, "Rule_1_ZH_ZH_EN_EN"

    if zh_clean == "EN-aligned" and en_clean == "ZH-aligned":
        return 0, "Rule_2_ZH_EN_EN_ZH"

    if zh_clean == "ZH-aligned" and en_clean == "ZH-aligned":
        return 1, "Rule_3_ZH_ZH_EN_ZH"

    if zh_clean == "EN-aligned" and en_clean == "EN-aligned":
        return 0.5, "Rule_4_ZH_EN_EN_EN"

    return 0, "Rule_Default"


def init_rule_counts():
    return {
        "Rule_SpecialChar": 0,
        "Rule_ZH_Empty": 0,
        "Rule_EN_Empty_ZH_Aligned": 0,
        "Rule_EN_Empty_Other": 0,
        "Rule_Neutral_Mixed": 0,
        "Rule_1_ZH_ZH_EN_EN": 0,
        "Rule_2_ZH_EN_EN_ZH": 0,
        "Rule_3_ZH_ZH_EN_ZH": 0,
        "Rule_4_ZH_EN_EN_EN": 0,
        "Rule_Default": 0,
    }


def format_score_distribution(score_counts: dict, total_scored: int, total_score_sum: float) -> list[str]:
    lines = []
    if total_scored == 0:
        return ["分数分布: 无可用分数"]

    def sort_key(k):
        try:
            return float(k)
        except Exception:
            return float("inf")

    lines.append("分数分布:")
    for s in sorted(score_counts.keys(), key=sort_key):
        cnt = score_counts[s]
        pct = (cnt / total_scored) * 100
        lines.append(f"  Score {s}: {cnt} ({pct:.2f}%)")

    avg = total_score_sum / total_scored if total_scored > 0 else 0.0
    score_100 = (avg / 2.0) * 100
    lines.append(f"  Total Score: {total_score_sum:.2f}")
    lines.append(f"  Average Score: {avg:.4f}")
    lines.append(f"  Score (0-100 scale): {score_100:.2f}")
    return lines


def score_for_two_cols(df: pd.DataFrame, model_name: str) -> str:
    """按第5/6列统计并评分。"""
    if df.shape[1] < 6:
        return f"===== {model_name} =====\n列数不足6列，跳过。\n"

    zh_series = df.iloc[:, 4].map(clean_label)  # 第5列
    en_series = df.iloc[:, 5].map(clean_label)  # 第6列

    rule_counts = init_rule_counts()
    score_counts = {}
    total_scored = 0
    total_score_sum = 0.0
    empty_score_count = 0

    for zh_label, en_label in zip(zh_series, en_series):
        score, rule_name = calculate_pair_score_with_rule(zh_label, en_label)
        if rule_name in rule_counts:
            rule_counts[rule_name] += 1

        if score is None:
            empty_score_count += 1
            continue

        score_str = str(score)
        score_counts[score_str] = score_counts.get(score_str, 0) + 1
        total_scored += 1
        total_score_sum += float(score)

    lines = [
        f"===== {model_name} =====",
        f"标签列: 第5列={df.columns[4]}, 第6列={df.columns[5]}",
        f"总行数: {len(df)}, 可评分行数: {total_scored}, 空分数行数: {empty_score_count}",
        *format_score_distribution(score_counts, total_scored, total_score_sum),
        "规则分布:",
    ]
    for k, v in rule_counts.items():
        pct = (v / len(df) * 100) if len(df) > 0 else 0
        lines.append(f"  {k}: {v} ({pct:.2f}%)")
    lines.append("")
    return "\n".join(lines)


def score_reference_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"===== {p.stem} =====\n文件不存在: {display_path(path)}\n"

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if REFERENCE_SCORE_COL not in df.columns:
        return f"===== {p.stem} =====\n缺少列: {REFERENCE_SCORE_COL}\n"

    s = df[REFERENCE_SCORE_COL].map(clean_label)
    score_counts = {}
    total_scored = 0
    total_score_sum = 0.0
    empty_score_count = 0

    for v in s:
        if not v:
            empty_score_count += 1
            continue
        try:
            fv = float(v)
            key = str(fv).rstrip("0").rstrip(".") if "." in str(fv) else str(int(fv))
            score_counts[key] = score_counts.get(key, 0) + 1
            total_scored += 1
            total_score_sum += fv
        except ValueError:
            empty_score_count += 1

    lines = [
        f"===== {p.stem} =====",
        f"评分列: {REFERENCE_SCORE_COL}",
        f"总行数: {len(df)}, 可评分行数: {total_scored}, 空/非法分数行数: {empty_score_count}",
        *format_score_distribution(score_counts, total_scored, total_score_sum),
        "",
    ]
    return "\n".join(lines)


def main():
    report_parts = []
    report_parts.append("=== Model Score Summary ===\n")

    for f in CSV_FILES:
        p = Path(f)
        model_name = p.stem
        if not p.exists():
            report_parts.append(f"[{model_name}] 文件不存在: {display_path(f)}\n")
            continue
        try:
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
            report_parts.append(score_for_two_cols(df, model_name))
        except Exception as e:
            report_parts.append(f"[{model_name}] 读取/统计失败: {e}\n")

    if REFERENCE_FILE:
        report_parts.append(score_reference_file(REFERENCE_FILE))

    out_path = Path(OUT_TXT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_parts), encoding="utf-8")

    print(f"统计完成，已输出: {display_path(OUT_TXT)}")


if __name__ == "__main__":
    main()
