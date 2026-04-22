# **CC-Eval Dataset and Evaluation Code**

Evaluate cross-lingual value alignment and Chinese-context understanding in large language models using open-ended generation and judge-based scoring.

## **About the Project**

This repository contains:

- **`data/bilingual_paralle_value-alignment.csv`** - A bilingual parallel value-alignment dataset with paired Chinese and English prompts.
- **`data/Chinese-context_task/`** - Chinese-context evaluation data covering classical literature, classical Chinese, folk culture, Chinese aesthetics and philosophy, pragmatic intent understanding, and modern Chinese internet slang.
- **Evaluation code** for labeling bilingual responses and scoring Chinese-context model outputs.
- **Score aggregation code** for computing alignment scores and summary statistics.

CC-Eval is designed for research on:

- Cross-lingual value-alignment shifts between Chinese and English prompts
- Chinese-context cultural and pragmatic understanding
- Open-ended LLM evaluation with an LLM-as-a-Judge protocol

---

**Getting Started**

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/cc-eval.git
cd cc-eval
```

### **2. Create a Virtual Environment**

#### Windows

```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

#### Mac / Linux

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### **4. Configure Evaluation API**

The evaluation scripts use an OpenAI-compatible chat completion API.

Set the following environment variables before running judge-based evaluation:

```bash
export EVAL_API_KEY="your_api_key"
export EVAL_API_BASE_URL="your_openai_compatible_base_url"
export EVAL_MODEL_NAME="your_judge_model_name"
```

If your provider does not require a custom base URL, omit `EVAL_API_BASE_URL`.

---

## **Bilingual Value-Alignment Evaluation**

The bilingual setting evaluates whether model responses under paired Chinese and English prompts align with the intended value labels.

### **Label Responses**

Use `src/bilingual_value-alignment/evaluate_responses_batch.py` to label model responses.

```bash
python src/bilingual_value-alignment/evaluate_responses_batch.py \
  --data-dir ./data/responses \
  --eval-model "$EVAL_MODEL_NAME"
```

Optional flags:

- `--model-files file1.csv file2.csv` - Process only selected response files. If omitted, all CSV files in `data-dir` are processed.
- `--save-interval 5` - Save progress every N updated rows.
- `--request-sleep 0.5` - Sleep time between API calls.
- `--max-retry 3` - Maximum retry count for failed API calls.
- `--progress-print 20` - Print progress every N rows.

### **Calculate Scores**

Use `src/bilingual_value-alignment/calculate_scores_batch.py` to aggregate label pairs into alignment scores.

```bash
python src/bilingual_value-alignment/calculate_scores_batch.py \
  --data-dir ./data/responses \
  --output-dir ./results \
  --output-file model_score_summary.txt
```

Optional flags:

- `--model-files file1.csv file2.csv` - Process selected files only.
- `--reference-file reference.csv` - Include an optional reference scoring file.
- `--reference-score-col score` - Specify the reference score column.

---

## **Chinese-Context Task Evaluation**

The Chinese-context setting evaluates model outputs against reference answers for culturally grounded Chinese tasks.

Use `src/Chinese-context_task/evaluate_results.py`:

```bash
python src/Chinese-context_task/evaluate_results.py \
  --base-dir ./data/Chinese-context_task \
  --eval-model "$EVAL_MODEL_NAME"
```

Optional flags:

- `--api-key` - Provide the API key directly instead of using `EVAL_API_KEY`.
- `--base-url` - Provide an OpenAI-compatible API endpoint instead of using `EVAL_API_BASE_URL`.
- `--exclude-model-dir model_dir` - Skip a model result directory. This flag can be repeated.
- `--save-interval 5` - Save progress every N scored instances.
- `--overwrite` - Re-score entries that already have evaluation scores.
- `--max-evals 100` - Limit the total number of evaluated instances for debugging.

Expected input layout for this script:

```text
<base-dir>/
└── results/
    ├── model_a/
    │   ├── task_file_model_a.csv
    │   └── ...
    └── model_b/
        ├── task_file_model_b.csv
        └── ...
```

---

## **Data**

The released data are organized into two subsets:

- **Bilingual parallel value-alignment subset**  
  Paired Chinese and English prompts for evaluating value-alignment shifts across language contexts.
- **Chinese-context task subset**
  Chinese cultural and pragmatic tasks, including classical literature, classical Chinese, folk culture, Chinese aesthetics and philosophy, pragmatic intent understanding, and modern Chinese internet slang.

The scripts do not include model outputs by default. To run evaluation, place generated model response CSV files under the expected input directories.

---

## **License**

### **MIT License**

This work is licensed under a MIT License.

### **CC BY-NC-SA 4.0**

The CC-EVAL dataset is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

See [`LICENSE-DATA`](LICENSE-DATA) for details.

---
