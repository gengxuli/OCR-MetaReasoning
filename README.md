<p align="center">
  <h2 align="center"><strong>OCR-MetaReasoning Benchmark: Evaluating the Meta-Reasoning Ability of MLLMs in Text-Rich Image Understanding</strong></h2>
</p>

<div align="center">
<h5>
<em>Gengxu Li<sup>1</sup>, Yuan Wu<sup>1*</sup>, Yi Chang<sup>1,2,3</sup></em>
<br><sup>1</sup> School of Artificial Intelligence, Jilin University &emsp; <sup>2</sup> Engineering Research Center of Knowledge-Driven Human-Machine Intelligence, MOE, China</br>
<sup>3</sup> International Center of Future Science, Jilin University
</h5>
</div>

<div align="center">

<p>
  <a href="https://huggingface.co/datasets/GengxuLi123/OCR-MetaReasoning"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-FFD21F?style=flat-square" alt="Hugging Face Dataset"></a>
  <a href="https://arxiv.org/abs/2608.30678"><img src="https://img.shields.io/badge/arXiv-2608.30678-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv 2608.30678"></a>
  <a href="https://github.com/gengxuli/OCR-MetaReasoning"><img src="https://img.shields.io/badge/Code-GitHub-24292f?style=flat-square&logo=github" alt="Source code"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-0b7285?style=flat-square" alt="MIT License"></a>
</p>

<p>
  <a href="#overview">Overview</a> ·
  <a href="#benchmark-at-a-glance">Benchmark</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#results">Results</a> ·
  <a href="#citation">Citation</a>
</p>

</div>


<!--
README layout principles are adapted from the visual-first, navigation-centered
patterns curated in https://github.com/matiassingers/awesome-readme.
-->

> **OCR-MetaReasoning** is a controlled benchmark for testing whether multimodal large language models (MLLMs) can organize OCR-grounded visual evidence according to the required reasoning direction—not merely read text from an image.

<div align="center">
<table>
  <tr>
    <td align="center"><strong>1,500</strong><br><sub>single-image samples</sub></td>
    <td align="center"><strong>3 × 5</strong><br><sub>balanced taxonomy</sub></td>
    <td align="center"><strong>15</strong><br><sub>taxonomy cells</sub></td>
    <td align="center"><strong>2</strong><br><sub>answer / process metrics</sub></td>
  </tr>
</table>
</div>

## News and Updates

- **[`2026/09`]:** Our paper is now accessible at [arXiv](https://arxiv.org/abs/2608.30678).
- **[`2026/08`]:** We are delighted that OCR-MetaReasoning has been accepted to EMNLP 2026 Findings!
- **[`2026/08`]:** Release the [dataset](https://huggingface.co/datasets/GengxuLi123/OCR-MetaReasoning) and evaluation script.

## Contents

- [Overview](#overview)
- [Benchmark at a Glance](#benchmark-at-a-glance)
- [Meta-Reasoning Task](#meta-reasoning-task)
- [Taxonomy](#taxonomy)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visual Assets](#visual-assets)
- [Quick Start](#quick-start)
- [Repository Guide](#repository-guide)
- [Paper](#paper)
- [Citation](#citation)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

Text-rich images encode meaning through words, tables, charts, fields, layout, constraints, legends, and cross-region correspondences. A model can therefore produce a plausible answer while still failing to bind the right visual evidence to the intended reasoning process.

OCR-MetaReasoning evaluates this distinction directly. Every item contains one text-rich image and one question whose dominant bottleneck is explicitly organized as one of three meta-reasoning directions:

- **Meta-deduction:** apply a visible rule or constraint to a candidate case.
- **Meta-induction:** infer a hidden regularity from aligned observations and generalize it.
- **Meta-abduction:** reason backward from an observed result to recover a hidden premise or minimal explanation.

The benchmark reports both final-answer correctness and reasoning-process compliance, making it possible to distinguish “the model answered correctly” from “the model used the intended OCR-grounded reasoning path.”

## Benchmark at a Glance

| Property | OCR-MetaReasoning |
| --- | --- |
| **Primary target** | Meta-reasoning ability of MLLMs in OCR-grounded, text-rich image understanding |
| **Input** | One image \(I\) and one question \(q\) |
| **Output** | Numbered reasoning steps followed by a standalone final answer |
| **Scale** | 1,500 samples; 500 per reasoning direction |
| **Balance** | 3 reasoning types × 5 OCR-object categories; 100 samples per cell |
| **Answer formats** | String, integer, floating-point, and JSON |
| **Scoring** | Normalized exact match, numeric match, and JSON micro-F1 |
| **Primary metric** | MRMS — Meta-Reasoning Macro Score |
| **Process diagnostic** | RPCS — Reasoning Process Compliance Score |
| **Image setting** | Single-image, text-rich visual reasoning |

### Why this benchmark?

OCR-MetaReasoning is designed around three evaluation requirements:

1. **Reasoning direction is explicit.** Deduction, induction, and abduction are evaluated as distinct capabilities rather than being collapsed into a generic “reasoning” score.
2. **OCR is grounded in visual structure.** Relevant evidence can be a word, field, table cell, chart mark, legend, footnote, alignment, or relation between distant regions.
3. **Outcome and process are separated.** MRMS measures whether the answer is correct; RPCS diagnoses whether the visible solution process follows the intended, grounded reasoning path.

## Meta-Reasoning Task

The benchmark uses a hypothesis–rule–observation view of reasoning. Let \(H\) denote a hypothesis, candidate state, or hidden premise; \(R\) a rule, constraint, mapping, or regularity; and \(O\) an observation, result, or consequence grounded in the image.

| Direction | Formal view | What the model must do | Typical failure mode |
| --- | --- | --- | --- |
| **Meta-deductive** | \(H + R \rightarrow O\) | Extract an explicit rule, bind it to image evidence, test candidate conditions, and derive the supported conclusion. | Copies a salient value without checking all clauses, thresholds, units, or exceptions. |
| **Meta-inductive** | \(H + O \rightarrow R\) | Align multiple visible examples, infer an unstated pattern, validate it, and apply it to a target. | Matches a local field or example without identifying the stable rule. |
| **Meta-abductive** | \(O + R \rightarrow H\) | Start from a result, anomaly, or goal; trace constraints backward; compare hypotheses; recover the unique or minimal hidden premise. | Gives a plausible explanation that is weakly grounded or does not cover all observations. |

### Expected model response

The evaluator prompts a model to produce a transparent, machine-readable answer structure:

```text
Step 1: ...
Step 2: ...
...
Final Answer: ...
```

For numeric answers, the final line should contain a number only. For structured answers, it should contain valid JSON. The scorer reads the standalone final-answer line and preserves the preceding steps for RPCS evaluation.

## Taxonomy

The benchmark crosses the three meta-reasoning directions with five OCR-object categories. This balanced design prevents performance on a familiar document type or a single reasoning pattern from hiding localized weaknesses.

| OCR-object category | Scope |
| --- | --- |
| **Transaction analysis** | Receipts and invoices; quantities, prices, totals, discounts, taxes, and units. |
| **Data interpretation** | Tables and charts; trends, comparisons, derived values, and visual encodings. |
| **Field dependency** | Forms and certificates; relations among fields, labels, values, and cross-field constraints. |
| **Document logic** | Notices, policies, and document-like pages; clauses, eligibility, exceptions, and conditions. |
| **Layout semantics** | Posters, webpages, infographics, and spatially organized text; grouping, alignment, legends, and non-adjacent evidence. |

### Balanced distribution

|  | Transaction analysis | Data interpretation | Field dependency | Document logic | Layout semantics |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Meta-deductive** | 100 | 100 | 100 | 100 | 100 |
| **Meta-inductive** | 100 | 100 | 100 | 100 | 100 |
| **Meta-abductive** | 100 | 100 | 100 | 100 | 100 |
| **Total** | **300** | **300** | **300** | **300** | **300** |

## Dataset

### Download

The released dataset is hosted on Hugging Face:

<div align="center">

**[🤗 GengxuLi123/OCR-MetaReasoning](https://huggingface.co/datasets/GengxuLi123/OCR-MetaReasoning)**

</div>

Load the hosted dataset with 🤗 Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("GengxuLi123/OCR-MetaReasoning")
print(dataset)
```

The repository also contains the JSONL benchmark splits and their referenced images:

```text
dataset/
├── meta_deductive/
│   ├── benchmark_meta_deductive.jsonl
│   └── images/
├── meta_inductive/
│   ├── benchmark_meta_inductive.jsonl
│   └── images/
└── meta_abductive/
    ├── benchmark_meta_abductive.jsonl
    └── images/
```

To load the local JSONL files directly:

```python
from datasets import load_dataset

data_files = {
    "meta_deductive": "dataset/meta_deductive/benchmark_meta_deductive.jsonl",
    "meta_inductive": "dataset/meta_inductive/benchmark_meta_inductive.jsonl",
    "meta_abductive": "dataset/meta_abductive/benchmark_meta_abductive.jsonl",
}

dataset = load_dataset("json", data_files=data_files)
```

### Record schema

Each JSONL record follows the same schema:

| Field | Description |
| --- | --- |
| <code>sample_id</code> | Unique sample identifier. |
| <code>image</code> | Image path relative to the corresponding reasoning split. |
| <code>question</code> | Text-rich image reasoning question. |
| <code>meta_reasoning_type</code> | <code>meta_deductive</code>, <code>meta_inductive</code>, or <code>meta_abductive</code>. |
| <code>reasoning_taxonomy</code> | One of the five OCR-object categories. |
| <code>answer_type</code> | <code>string</code>, <code>integer</code>, <code>float</code>, or <code>json</code>. |
| <code>answer</code> | Canonical answer used for scoring. |
| <code>reasoning_steps</code> | Reference reasoning path, with at least two steps per released item. |
| <code>metric</code> | <code>exact_match</code>, <code>numeric</code>, or <code>json_f1</code>. |

### Dataset statistics

| Statistic | Value |
| --- | ---: |
| String answers | 513 |
| Integer answers | 561 |
| Floating-point answers | 146 |
| JSON answers | 280 |
| Average question length | 75.77 tokens |
| Question length range | 21–201 tokens |
| Average reference reasoning steps | 4.22 |
| Reference reasoning step range | 2–8 |
| Images per sample | 1 |

The released benchmark has complete annotations and automatic scoring specifications for all 1,500 samples. Human verification and a blinded re-annotation study reported 96.0% agreement for meta-reasoning type and 92.7% agreement for OCR-object category.

## Evaluation

### MRMS: answer-level correctness

The **Meta-Reasoning Macro Score (MRMS)** is the primary leaderboard metric:

```text
MRMS = (Acc_deductive + Acc_inductive + Acc_abductive) / 3
```

It averages final-answer performance across the three reasoning directions, so a strong result in one direction cannot dominate the overall score.

Final answers are scored per sample using:

| Answer type | Metric |
| --- | --- |
| String | Normalized exact match |
| Integer / float | Normalized numeric match |
| JSON | JSON micro-F1 over flattened key–value items |

### RPCS: process-level compliance

The **Reasoning Process Compliance Score (RPCS)** is a separate diagnostic over the visible reasoning steps:

```text
RPCS = (capability_match + groundedness + step_completeness + non_hallucination) / 4
```

| Criterion | Meaning |
| --- | --- |
| <code>capability_match</code> | The dominant reasoning direction matches the target task. |
| <code>groundedness</code> | Key claims are tied to image evidence or question constraints. |
| <code>step_completeness</code> | Necessary intermediate reasoning nodes are covered. |
| <code>non_hallucination</code> | The process avoids unsupported rules, entities, fields, and values. |

RPCS is a visible-process diagnostic, not a claim about hidden internal reasoning. In the reported evaluation, the judge receives the image, question, target labels, reference steps, and model process with the standalone final-answer line removed; the gold answer and MRMS result are withheld.

## Results

The following snapshot is reported in the manuscript. Scores are percentage points.

| Model | MRMS | RPCS (avg.) | Setting |
| --- | ---: | ---: | --- |
| Gemini-3.1-Pro-Preview | **89.3** | **96.2** | Closed-source |
| GPT-5.4-Medium | 87.7 | 95.2 | Closed-source |
| Doubao-Seed-2.0-Pro | 86.4 | 93.0 | Closed-source |
| Qwen3-VL-235B-A22B-Thinking | 81.5 | 89.6 | Open-source |
| Kimi-K2.5 | 79.6 | 91.1 | Open-source |

Key observations:

- Across the evaluated models, **induction** averages 82.5 and **abduction** 80.1, while **deduction** is lower at 73.6.
- **Layout semantics** is the most difficult OCR-object category, averaging 66.9 across models.
- Closed-source models average 82.9 MRMS versus 73.4 for open-source models, a 9.5-point gap.
- RPCS is higher than MRMS for every reported model: plausible, grounded-looking processes can still lead to incorrect final answers.

## Visual Assets

The following figures and tables are included as SVG assets under <code>assets/</code>. The source locations use the current manuscript's figure/table numbering and pagination.

### Hero example

<p align="center">
  <img src="assets/hero_abduction_example.svg" alt="OCR-grounded abduction example">
</p>

### Benchmark construction

<p align="center">
  <img src="assets/benchmark_pipeline.svg" alt="OCR-MetaReasoning benchmark pipeline">
</p>

### Main results

<p align="center">
  <img src="assets/mrms_heatmap.svg" alt="MRMS heatmap">
</p>

<p align="center">
  <img src="assets/main_results_table.svg" alt="Main results on OCR-MetaReasoning">
</p>

### Process–outcome contrast

<p align="center">
  <img src="assets/rpcs_mrms_table.svg" alt="RPCS and MRMS comparison">
</p>

## Quick Start

### 1. Install

```bash
git clone https://github.com/gengxuli/OCR-MetaReasoning.git
cd OCR-MetaReasoning

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install openai pillow pydantic requests tqdm
```

Install <code>datasets</code> as well if you want to load the Hugging Face or local JSONL data through 🤗 Datasets:

```bash
python -m pip install datasets
```

### 2. Configure an OpenAI-compatible endpoint

The inference and RPCS clients use an OpenAI-compatible chat-completions interface. Keep credentials in environment variables or pass them through the command line; do not hard-code keys in source files.

```bash
export MODEL_API_KEY="YOUR_API_KEY"
export MODEL_BASE_URL="https://openrouter.ai/v1"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

### 3. Run multimodal inference

<code>--sample_num</code> is applied per selected reasoning type. Thus, <code>500</code> with <code>all</code> evaluates up to 500 samples for each of the three splits.

```bash
export MODEL_NAME="your-model-name"

python OCR_MetaReasoning/evaluation/openai_infer.py \
  --model_name "${MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --meta_reasoning_types all \
  --sample_num 500 \
  --batch_size 32 \
  --workers 8 \
  --temperature 0.0 \
  --top_p 1.0 \
  --timeout 600
```

Inference files are written to:

```text
OCR_MetaReasoning/evaluation/result/${MODEL_NAME}_result/
```

The processor resumes from successful existing records by default. Add <code>--retry_all</code> when a complete rerun is intended.

### 4. Summarize MRMS and subscores

```bash
python OCR_MetaReasoning/evaluation/stats.py \
  --model_name "${MODEL_NAME}"
```

This writes <code>score_summary.json</code> in the model's result directory and reports scores by reasoning direction, answer type, OCR-object category, and the full 3 × 5 interaction.

### 5. Evaluate RPCS

RPCS is computed from a completed inference result using a configurable judge model. The default mode includes the original image; use <code>--no_image</code> only for a text-only control.

```bash
export JUDGE_MODEL_NAME="your-judge-model-name"

python OCR_MetaReasoning/evaluation/rpcs_judge.py \
  --source_model_name "${MODEL_NAME}" \
  --judge_model_name "${JUDGE_MODEL_NAME}" \
  --base_url "${MODEL_BASE_URL}" \
  --meta_reasoning_types all \
  --workers 8 \
  --batch_size 32 \
  --temperature 0.0 \
  --timeout 600 \
  --max_retries 3
```

RPCS files are written to:

```text
OCR_MetaReasoning/evaluation/rpcs_result/${MODEL_NAME}_result/
```

## Repository Guide

| Path | Purpose |
| --- | --- |
| <code>dataset/meta_deductive/</code> | Meta-deductive JSONL split and images. |
| <code>dataset/meta_inductive/</code> | Meta-inductive JSONL split and images. |
| <code>dataset/meta_abductive/</code> | Meta-abductive JSONL split and images. |
| <code>OCR_MetaReasoning/evaluation/openai_infer.py</code> | Multimodal inference and answer scoring. |
| <code>OCR_MetaReasoning/evaluation/rpcs_judge.py</code> | Process-level RPCS judging. |
| <code>OCR_MetaReasoning/evaluation/stats.py</code> | Result aggregation and optional rescoring. |
| <code>OCR_MetaReasoning/evaluation/answer_utils.py</code> | Final-answer extraction and exact/numeric/JSON scorers. |
| <code>OCR_MetaReasoning/evaluation/config.py</code> | Dataset, result, model, and endpoint configuration. |
| <code>OCR_MetaReasoning/llms/</code> | OpenAI-compatible text/image client utilities. |
| <code>OCR_MetaReasoning/schemas/</code> | Dataset record schema definitions. |
| <code>OCR_MetaReasoning/utils/</code> | Shared image conversion helpers. |
| <code>assets/</code> | README figures and table screenshots in SVG format. |
| <code>LICENSE</code> | Project license. |

## Paper

**OCR-MetaReasoning Benchmark: Evaluating the Meta-Reasoning Ability of MLLMs in Text-Rich Image Understanding**  
Gengxu Li, Yuan Wu, and Yi Chang

> **arXiv preprint:** [arXiv:2608.30678](https://arxiv.org/abs/2608.30678)

## Citation

If you use OCR-MetaReasoning, please cite:

```bibtex
@misc{li2026ocrmetareasoningbenchmarkevaluatingmetareasoning,
  title         = {OCR-MetaReasoning Benchmark: Evaluating the Meta-Reasoning Ability of MLLMs in Text-Rich Image Understanding},
  author        = {Gengxu Li and Yuan Wu and Yi Chang},
  year          = {2026},
  eprint        = {2608.30678},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2608.30678}
}
```

## Limitations

OCR-MetaReasoning focuses on controlled, single-image, OCR-grounded meta-reasoning. It does not evaluate multi-page evidence chains, cross-image aggregation, retrieval-augmented document reasoning, or interactive clarification. The balanced taxonomy is intended for controlled comparison rather than modeling natural task frequencies. RPCS evaluates visible solution processes and should be interpreted as a diagnostic rather than direct evidence of internal reasoning.

## Acknowledgements

This work is supported by the National Key Research and Development Program of China (No. 2023YFF0905400), the National Natural Science Foundation of China (No. U2341229), and the Reform Commission Foundation of Jilin Province (No. 2024C003).

## License

See [<code>LICENSE</code>](LICENSE) for the project license. Please also consult the Hugging Face dataset card and the licenses of any upstream resources before redistributing derived data.

<div align="center">

<sub>OCR-MetaReasoning · an OCR-grounded benchmark for measuring what models read, how they reason, and whether the two stay connected.</sub>

</div>
