# WattBot 2025 - Team Attention Please 🔋

<div align="center">

<img src="docs/images/wattbot-logo.png" alt="WattBot — Team Attention Please" width="800">

[![Competition](https://img.shields.io/badge/Kaggle-WattBot%202025-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/WattBot2025)
![Team](https://img.shields.io/badge/Team-Attention%20Please-FF6B6B?style=for-the-badge)
![System](https://img.shields.io/badge/System-HERO-9B59B6?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### HERO: Hybrid Evidence Retrieval Optimization

*Evidence-based Energy Estimation for AI Workloads using Retrieval-Augmented Generation*

[📄 Technical Report](docs/WattBot2025_AttentionPlease_Technical_Report.pdf) | [📊 Presentation](docs/WattBot2025_AttentionPlease_Presentation.pdf)

[Baseline Notebook](notebook/v1_gemini_2_5pro.ipynb) · [HERO Notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb) · [繁體中文](#中文版說明)

</div>

## Table of Contents

- [Introduction](#introduction)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
  - [Baseline: BM25 Retrieval](#baseline-bm25-retrieval)
  - [HERO: Visual-Aware Hybrid Retrieval](#hero-visual-aware-hybrid-retrieval)
  - [Architecture Comparison](#architecture-comparison)
- [Project Structure](#project-structure)
- [Dataset and Evaluation](#dataset-and-evaluation)
- [Output Format](#output-format)
- [Experimental Results](#experimental-results)
- [Notebook Setup](#notebook-setup)
- [Team Members](#team-members)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [中文版說明](#中文版說明)

## Introduction

WattBot asks a practical question: **can a system find the evidence behind an AI sustainability claim?** Answers often depend on a single value inside a table or figure, expressed in units that differ from the question.

Our system retrieves the relevant evidence and returns a normalized answer with document citations and supporting text. It also handles questions that cannot be answered from the available documents.

The project develops two approaches:

- **BM25 Baseline** — text and table retrieval with query routing and two-stage answer generation.
- **HERO** — layout-aware parsing and OCR, sparse–dense retrieval, weighted rank fusion, CrossEncoder reranking, and question-specific generation.

The key finding is the **table paradox**: adding raw tables can introduce noise into keyword retrieval, while removing them discards important numerical evidence. HERO addresses this by preserving content types and giving table and OCR evidence explicit weights during retrieval.

## Technology Stack

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">Component</th>
      <th width="1500" align="left">Technology</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Language</td>
      <td>Python</td>
    </tr>
    <tr>
      <td>Environment</td>
      <td>Jupyter Notebook / Google Colab</td>
    </tr>
    <tr>
      <td>Baseline Parsing</td>
      <td>PyMuPDF + pdfplumber</td>
    </tr>
    <tr>
      <td>HERO Parsing</td>
      <td>Docling + EasyOCR + OpenCV</td>
    </tr>
    <tr>
      <td>Sparse Retrieval</td>
      <td>BM25 (rank-bm25)</td>
    </tr>
    <tr>
      <td>Dense Retrieval</td>
      <td>Sentence-Transformers + BGE-large + FAISS</td>
    </tr>
    <tr>
      <td>Reranking</td>
      <td>BGE-reranker-v2-m3</td>
    </tr>
    <tr>
      <td>Answer Generation</td>
      <td>Google Gemini API</td>
    </tr>
    <tr>
      <td>Data Processing</td>
      <td>pandas + NumPy</td>
    </tr>
  </tbody>
</table>

The stack above follows the public notebooks. The technical report describes additional experimental configurations.

## System Architecture

### Baseline: BM25 Retrieval

![Baseline BM25 architecture](docs/images/baseline-architecture.png)

*Figure 2 from the [technical report, page 5](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=5).*

The baseline separates document processing from question answering:

1. **Document extraction.** Parse PDFs and prepare text and table chunks.
2. **Sparse indexing.** Build BM25 indexes for lexical matching.
3. **Query routing.** Identify table-related questions using keywords. The notebook's `hybrid_search` combines **Top-7 text + Top-3 table** results for these questions, applying a **0.8× table-score weight**. Other questions use text retrieval.
4. **Answer generation.** First answer from a small evidence set; if unsuccessful, expand the context and try again.
5. **Structured output.** Return the answer, value, unit, references, supporting evidence, and explanation.

The report diagram presents text-only and hybrid indexes; the notebook routing implementation separates text and table indexes. Here, “hybrid” refers to content types. This baseline routing path does not combine BM25 with dense retrieval.

### HERO: Visual-Aware Hybrid Retrieval

![HERO architecture](docs/images/hero-architecture.png)

*Figure 4 from the [technical report, page 7](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=7).*

HERO extends the baseline in five stages:

1. **Visual-aware ingestion.** Docling extracts document elements; EasyOCR and OpenCV turn text inside figures into searchable evidence. Chunks retain their content type and document ID.
2. **Dual retrieval.** BM25 handles exact terms and numbers, while dense embeddings retrieve semantically related passages.
3. **Weighted reciprocal rank fusion.** Combine sparse and dense rankings, boosting **tables ×2.0**, **OCR ×1.8**, and **author/year matches ×2.0**.
4. **CrossEncoder reranking.** Jointly score each query–chunk pair to prioritize relevant evidence.
5. **Adaptive generation.** Choose prompts by question type, answer from Top-3 context first, and expand the supplied context when a second attempt is needed.

The original report figure labels the embedding model as **BGE-M3**. The public notebook uses **BGE-large-en-v1.5**, normalized embeddings with FAISS `IndexFlatIP`, and **BGE-reranker-v2-m3**. It includes MiniLM fallbacks for model-loading failures.

### Architecture Comparison

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">Component</th>
      <th width="750" align="left">BM25 Baseline</th>
      <th width="750" align="left">HERO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Document parsing</td>
      <td align="left">PyMuPDF / pdfplumber</td>
      <td align="left">Docling + EasyOCR</td>
    </tr>
    <tr>
      <td align="left">Evidence sources</td>
      <td align="left">Text and tables</td>
      <td align="left">Text, tables, and figure OCR</td>
    </tr>
    <tr>
      <td align="left">Retrieval</td>
      <td align="left">Text/table BM25 indexes</td>
      <td align="left">BM25 + dense vector search</td>
    </tr>
    <tr>
      <td align="left">Table handling</td>
      <td align="left">BM25 score weight ×0.8</td>
      <td align="left">RRF weight ×2.0</td>
    </tr>
    <tr>
      <td align="left">Ranking</td>
      <td align="left">Weighted BM25 scores</td>
      <td align="left">Weighted RRF + CrossEncoder</td>
    </tr>
    <tr>
      <td align="left">Query adaptation</td>
      <td align="left">Table-related routing</td>
      <td align="left">Unit expansion and metadata hints</td>
    </tr>
    <tr>
      <td align="left">Generation</td>
      <td align="left">Two-stage fallback</td>
      <td align="left">Type-specific prompts and two-stage fallback</td>
    </tr>
  </tbody>
</table>

## Project Structure

```text
WattBot-RAG-System/
├── README.md
├── notebook/
│   ├── v1_gemini_2_5pro.ipynb                 # BM25 baseline experiments
│   └── Gemini_2_5_pro_新pipeline_0_821.ipynb  # HERO pipeline
├── data/
│   ├── metadata.csv                           # 32 source documents
│   ├── train_QA.csv                           # 41 training questions
│   └── test_Q.csv                             # 282 test questions
└── docs/
    ├── images/                                # Architecture figures
    ├── WattBot2025_AttentionPlease_Technical_Report.pdf
    └── WattBot2025_AttentionPlease_Presentation.pdf
```

## Dataset and Evaluation

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">File</th>
      <th width="750" align="left">Records</th>
      <th width="750" align="left">Content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><code>metadata.csv</code></td>
      <td align="right">32</td>
      <td align="left">Document metadata and source URLs</td>
    </tr>
    <tr>
      <td align="left"><code>train_QA.csv</code></td>
      <td align="right">41</td>
      <td align="left">Questions with answers, citations, and evidence</td>
    </tr>
    <tr>
      <td align="left"><code>test_Q.csv</code></td>
      <td align="right">282</td>
      <td align="left">Test questions and expected units</td>
    </tr>
  </tbody>
</table>

Source PDFs are downloaded separately using the metadata URLs. The PDFs in `docs/` are the project report and presentation.

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">WattBot Score component</th>
      <th width="750" align="left">Weight</th>
      <th width="750" align="left">Criterion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Answer value</td>
      <td align="right">75%</td>
      <td align="left">Numeric tolerance or normalized categorical match</td>
    </tr>
    <tr>
      <td align="left">Reference IDs</td>
      <td align="right">15%</td>
      <td align="left">Jaccard overlap of citation sets</td>
    </tr>
    <tr>
      <td align="left">Unanswerable handling</td>
      <td align="right">10%</td>
      <td align="left">Correct recognition of unavailable evidence</td>
    </tr>
  </tbody>
</table>

The report specifies **±0.1% relative tolerance** for numeric answers.

## Output Format

Generation produces structured JSON, which is mapped to these submission fields:

<table width="100%">
  <thead>
    <tr>
      <th width="1000" align="left">Field</th>
      <th width="1000" align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><code>id</code>, <code>question</code></td>
      <td align="left">Question identifier and text</td>
    </tr>
    <tr>
      <td align="left"><code>answer</code></td>
      <td align="left">Natural-language answer</td>
    </tr>
    <tr>
      <td align="left"><code>answer_value</code>, <code>answer_unit</code></td>
      <td align="left">Normalized value and unit</td>
    </tr>
    <tr>
      <td align="left"><code>ref_id</code>, <code>ref_url</code></td>
      <td align="left">Supporting document IDs and URLs</td>
    </tr>
    <tr>
      <td align="left"><code>supporting_materials</code></td>
      <td align="left">Evidence excerpt</td>
    </tr>
    <tr>
      <td align="left"><code>explanation</code></td>
      <td align="left">Explanation connecting evidence to the answer</td>
    </tr>
  </tbody>
</table>

## Experimental Results

### Final Competition Standings

<table width="100%">
  <thead><tr><th width="800" align="left">Leaderboard</th><th width="600" align="left">Rank</th><th width="600" align="left">Score</th></tr></thead>
  <tbody>
    <tr><td>Public</td><td>2nd</td><td>0.886</td></tr>
    <tr><td>Private — final standings</td><td>4th</td><td>0.840</td></tr>
  </tbody>
</table>

Verified for **Attention Please** on October 14, 2026: [official Public leaderboard](https://www.kaggle.com/competitions/WattBot2025/leaderboard?tab=public) and [official Private leaderboard](https://www.kaggle.com/competitions/WattBot2025/leaderboard?tab=private). The competition has ended; **Private rank 4 is the final competition placement**. The Public score uses approximately 66% of test data; Private uses the other 34%. The report's 0.886 result is the Public score, not the final Private score.

The experiments follow three questions: **does including tables help sparse retrieval, does HERO improve evidence retrieval, and do those improvements carry across language models?** Retrieval is evaluated on internal question sets; the model comparison reproduces report scores from the competition, whose test file contains 282 questions. Official Public and Private scores use different subsets of that test set.

### Experiment Configurations

The development moves from a table-noise experiment to a routed baseline, then to HERO. The configurations below distinguish these stages:

<table width="100%">
  <thead><tr><th width="600" align="left">Configuration</th><th width="1400" align="left">Meaning</th></tr></thead>
  <tbody>
    <tr><td>Hybrid BM25</td><td>Text plus raw Markdown tables; used in the table-paradox experiment.</td></tr>
    <tr><td>Text-Only BM25</td><td>Tables removed; the control configuration in that experiment.</td></tr>
    <tr><td>Routed BM25 baseline</td><td>The architecture described above: question-dependent text/table retrieval with a 0.8× table-score weight.</td></tr>
  </tbody>
</table>

The retrieval tables reproduce report Tables 1 and 16; the model comparison reproduces Figure 5. Evaluation scope and differences from the saved notebook run are collected at the end of this section.

### 1. The Table Paradox: Text-Only vs. Hybrid BM25

The baseline experiment compares a hybrid corpus containing text and raw Markdown tables with a text-only corpus. On the **39-query evaluation**, text-only retrieval matches or exceeds the hybrid version on every reported metric.

<table width="100%">
  <thead><tr>
    <th width="666" align="left">Metric</th>
    <th width="666" align="left">Hybrid: Text + Tables</th>
    <th width="666" align="left">Text-Only</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">Recall@1</td><td align="right">79.49%</td><td align="right">87.18%</td></tr>
    <tr><td align="left">Recall@3</td><td align="right">89.74%</td><td align="right">92.31%</td></tr>
    <tr><td align="left">Recall@5</td><td align="right">92.31%</td><td align="right">92.31%</td></tr>
    <tr><td align="left">Recall@10</td><td align="right">92.31%</td><td align="right">94.87%</td></tr>
    <tr><td align="left">nDCG@1</td><td align="right">0.7949</td><td align="right">0.8718</td></tr>
    <tr><td align="left">nDCG@3</td><td align="right">0.8497</td><td align="right">0.8859</td></tr>
    <tr><td align="left">nDCG@5</td><td align="right">0.8564</td><td align="right">0.8859</td></tr>
    <tr><td align="left">nDCG@10</td><td align="right">0.8617</td><td align="right">0.8950</td></tr>
    <tr><td align="left">MRR</td><td align="right">0.8526</td><td align="right">0.8974</td></tr>
  </tbody>
</table>

Source: [report Table 1, page 4](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=4). Here, “hybrid” means text plus tables, not sparse–dense retrieval.

Removing raw tables raises Recall@1 by **7.69 percentage points** and MRR by **0.0448**. The report attributes this to table-formatting noise: delimiters and disrupted structure dilute meaningful terms in BM25 scoring. Yet removing tables also removes numerical evidence. This trade-off motivates HERO's structured parsing, dense retrieval, and table-aware weighting.

### 2. HERO Retrieval Performance

HERO combines structured ingestion, sparse–dense retrieval, and reranking to improve evidence selection. The following table reproduces the report’s retrieval comparison; the configurations and evaluation samples are discussed in the evaluation notes.

<table width="100%">
  <thead><tr>
    <th width="500" align="left">Metric</th>
    <th width="500" align="left">Baseline (report Table 16)</th>
    <th width="500" align="left">HERO (report Table 16)</th>
    <th width="500" align="left">Difference</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">Recall@1</td><td align="right">79.49%</td><td align="right">80.49%</td><td align="right">+1.00%</td></tr>
    <tr><td align="left">Recall@3</td><td align="right">89.74%</td><td align="right">89.80%</td><td align="right">+0.06%</td></tr>
    <tr><td align="left">Recall@5</td><td align="right">92.31%</td><td align="right">92.68%</td><td align="right">+0.37%</td></tr>
    <tr><td align="left">Recall@10</td><td align="right">92.31%</td><td align="right">95.12%</td><td align="right">+2.81%</td></tr>
    <tr><td align="left">nDCG@1</td><td align="right">0.7949</td><td align="right">0.8049</td><td align="right">+0.0100</td></tr>
    <tr><td align="left">nDCG@3</td><td align="right">0.8497</td><td align="right">0.8502</td><td align="right">+0.0005</td></tr>
    <tr><td align="left">nDCG@5</td><td align="right">0.8564</td><td align="right">0.8584</td><td align="right">+0.0020</td></tr>
    <tr><td align="left">nDCG@10</td><td align="right">0.8617</td><td align="right">0.8665</td><td align="right">+0.0048</td></tr>
    <tr><td align="left">MRR</td><td align="right">0.8526</td><td align="right">0.8560</td><td align="right">+0.0034</td></tr>
  </tbody>
</table>

Source: [report Table 16, page 9](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=9). Recall differences use `%` to show direct subtraction of the displayed percentages (percentage-point changes), not relative percentage growth. Values above reproduce the report; the evaluation caveat below explains differences from the saved notebook run.

- **Coverage:** reported Recall@10 increases from 92.31% to 95.12%, a gain of **2.81 percentage points**.
- **Early evidence:** Recall@1 increases by **1.00 percentage point**, indicating more questions have a relevant reference at the first retrieved position.
- **Ranking:** nDCG@10 increases by **0.0048**, while MRR increases by **0.0034** in the report.

Text-Only BM25 has higher reported Recall@1 and MRR, while HERO has slightly higher Recall@10. These runs are not a verified same-sample comparison. The stronger evidence for HERO’s end-to-end value is the cross-model answer-score improvement shown below.

### 3. Evidence Coverage After Visual-Aware Ingestion

Docling and EasyOCR expand the searchable corpus beyond prose. The report's corpus inventory contains **8,407 chunks**:

<table width="100%">
  <thead><tr>
    <th width="666" align="left">Chunk Type</th>
    <th width="666" align="left">Count</th>
    <th width="666" align="left">Share</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">Text</td><td align="right">7,817</td><td align="right">93.0%</td></tr>
    <tr><td align="left">Table</td><td align="right">153</td><td align="right">1.8%</td></tr>
    <tr><td align="left">Figure OCR</td><td align="right">437</td><td align="right">5.2%</td></tr>
    <tr><td align="left">Total</td><td align="right">8,407</td><td align="right">100%</td></tr>
  </tbody>
</table>

Source: [report Table 8, page 6](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=6). OCR contributes 437 figure-derived chunks, while 153 chunks preserve table content. Content labels allow the retrieval system to weight these sources differently from prose.

Counts follow Table 8; they describe searchable evidence, not a standalone OCR accuracy measurement.

### 4. Pipeline and Model Comparison

<table width="100%">
  <thead><tr>
    <th width="500" align="left">Model</th>
    <th width="500" align="left">BM25 Baseline</th>
    <th width="500" align="left">HERO</th>
    <th width="500" align="left">Score Gain</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">GPT-4o-mini</td><td align="right">0.753</td><td align="right">0.789</td><td align="right">+0.036</td></tr>
    <tr><td align="left">GPT-4o</td><td align="right">0.768</td><td align="right">0.811</td><td align="right">+0.043</td></tr>
    <tr><td align="left">GPT-5.1</td><td align="right">0.778</td><td align="right">0.807</td><td align="right">+0.029</td></tr>
    <tr><td align="left">Gemini 2.5 Pro</td><td align="right">0.807</td><td align="right">0.821</td><td align="right">+0.014</td></tr>
    <tr><td align="left">Gemini 3 Pro</td><td align="right">0.853</td><td align="right">0.886</td><td align="right">+0.033</td></tr>
  </tbody>
</table>

Source: [report Figure 5, page 10](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=10). All five tested models improve with HERO in the report; this table is separate from the final Private standings above.

**GPT-4o has the largest absolute gain (+0.043)**, while **Gemini 3 Pro achieves the highest score in the report (0.886, also confirmed on the Public leaderboard)**. Gemini 3 Pro rises from 0.853 to 0.886, showing that retrieval improvements remain useful with a stronger answer model. Gemini 2.5 Pro rises from 0.807 to 0.821; this is the model initialized in the public HERO notebook.

### 5. Conclusions and Future Work

The experiments connect the table-noise problem to a practical retrieval redesign: preserve text, tables, and OCR as distinct evidence types; combine lexical and semantic search; rerank candidates; and adapt the answer context to the question.

The report proposes **learned fusion weights**, **multilingual corpora**, and **hierarchical indexing for larger collections** as future directions. The current weights and question routing remain heuristic. The two-stage generation design balances an initial narrow context against a broader fallback, but its independent benefit is not quantified in these tables.

<details>
<summary>Evaluation scope and reproducibility details</summary>

The baseline table experiment uses 39 queries. The final saved HERO evaluation uses 41, including unanswerable examples. Its saved values include Recall@3 = 87.80%, nDCG@3 = 0.8384, and MRR = 0.8519, differing from report Table 16. The report comparison should therefore be read as reported results rather than a newly verified same-sample evaluation; the text-only baseline also performs better on some metrics than the HERO figures shown above.

The notebook's Recall@K is an any-reference hit rate. Multiple chunks can come from the same document; chunk ranks and unique-document ranks are not interchangeable. The saved evaluation includes unanswerable entries, which also affects interpretation.

The public HERO pipeline defaults to eight final retrieval chunks; a generation request for ten fallback chunks can only use those supplied. Reranking is currently enabled for every query. Abstention uses heuristics and LLM responses rather than calibrated confidence. These figures are taken from the report and saved outputs, not a new execution of the models or competition evaluator.

</details>

## Notebook Setup

These instructions are based on the checked-in code. **A clean-environment, end-to-end run has not been verified.** The notebooks retain experimental cells and require manual setup.

### HERO Notebook

1. Open the [HERO notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb) in Google Colab. Upload the three CSV files from `data/` to `/content/`.
2. The following consolidated dependency command is derived from the workflow's imports; version compatibility has not been tested. Review the notebook's existing installation cells before running it:

```python
%pip install docling easyocr opencv-python-headless rank-bm25 sentence-transformers faiss-cpu pandas numpy requests pdfminer.six pillow tqdm google-genai
```

3. Add the following imports **before the Docling/OCR processing cell**. The checked-in notebook uses `np` before its later import, and the generation cell uses `time` and `types` without importing them:

```python
import numpy as np
import time
from google.genai import types
```

4. In the Gemini generation cell, replace the existing placeholder key and client initialization. For Colab Secrets, create a secret named `GOOGLE_API_KEY`, allow notebook access, then use:

```python
from google.colab import userdata
from google import genai

client = genai.Client(api_key=userdata.get("GOOGLE_API_KEY"))
```

5. Execute the workflow in this order: CSV loading → PDF download → EasyOCR initialization → Docling/OCR processing → BM25/dense indexes → retrieval and reranking helpers → Gemini generator and submission. Evaluation and cache-cleanup cells are optional.

The code uses these locations:

```text
/content/metadata.csv
/content/train_QA.csv
/content/test_Q.csv
/content/papers/
/content/wattbot_working/chunks/structured_chunks_with_ocr.jsonl
/content/submission_enhanced.csv
```

Check download failures and the OCR chunk count before generating answers. OCR extraction also depends on the Docling version and image representation; the import additions alone do not establish full compatibility. Model downloads, API access, and source-document availability are required.

### Baseline Notebook

The [baseline notebook](notebook/v1_gemini_2_5pro.ipynb) contains several experiments. To prepare its data and run the Gemini answer-generation section:

1. Open the notebook in Google Colab. Run its Kaggle login, dependency-installation, and competition-download cells. You need access to the WattBot 2025 competition data. The download cell prints `Downloaded to: ...`; keep that path.
2. Replace the hard-coded `/root/.cache/kagglehub/competitions/WattBot2025` path in later cells with the path printed in step 1. In particular, update `INPUT_DIR` in the data-loading cell and in `main()` at the end. The PDF parsing test also sets `data_dir` to that hard-coded path.
3. Run the document-processing cells needed to create `/content/hybrid_chunks.json`. The final generation cell reads that file, plus `metadata.csv` and `test_Q.csv` from `INPUT_DIR`. Check that all three exist before running it.
4. Before the final generation cell, make sure `google-genai` is installed; the notebook imports it but has no installation cell for it. Replace the placeholder `GEMINI_API_KEY = "# ⚠️ 請替換成你的 API Key"` with a valid key, then run the cell to generate a submission CSV. It calls `main()` automatically and sends the test questions to the Gemini API. Keep your key out of any notebook you share.

The notebook also includes separate PDF parsing and retrieval tests. Those cells are experiments rather than additional steps in the final generation function. The dependency cells install Python packages and, for OCR, Poppler and Tesseract through `apt-get`. This setup is based on reading the notebook; a clean end-to-end run has not been verified.

## Team Members

**Team Attention Please**

- **Shao-Hua Wu**‡ - Document Processing & Retrieval System & Project Lead
- **Pei-Ju Hsieh**‡ - Document Processing & Retrieval System & Hybrid Search Implementation & Answer Generation & Prompt Engineering & Optimization
- **Bo-Hao Chen**‡ - Answer Generation & Prompt Engineering & Optimization
- **Yi-Chen Hsiao**∗ - Report Generation
- **Yi-Yang Xue**† - Retrieval System & Hybrid Search Implementation

<sup>‡ Equal contribution</sup>

## Citation

```bibtex
@misc{wattbot2025_hero,
  title = {WattBot 2025: Retrieval-Augmented Generation for ESG Intelligence},
  author = {Wu, Shao-Hua and Hsieh, Pei-Ju and Chen, Bo-Hao and Hsiao, Yi-Chen and Xue, Yi-Yang},
  year = {2025},
  url = {https://github.com/AnnieHsieh-12/WattBot-RAG-System}
}
```

## License

No `LICENSE` file is currently included in this repository. Licensing terms have not been specified in the repository.

## Acknowledgments

- The WattBot competition organizers at ML+X, University of Wisconsin–Madison.
- Google for Gemini API access.
- The open-source community for the tools and libraries supporting this project.

See the [technical report](docs/WattBot2025_AttentionPlease_Technical_Report.pdf) for related work, including the KohakuRAG case study that informed the team's approach.

---

## 中文版說明

## WattBot 2025：AI 環境影響的證據檢索與問答系統

**HERO（Hybrid Evidence Retrieval Optimization）** 是 Attention Please 團隊為 WattBot 2025 與政大自然語言處理期末專題開發的 RAG 系統，從學術論文與產業報告中擷取 AI 能耗、碳排、水資源與永續相關資訊。

[技術報告](docs/WattBot2025_AttentionPlease_Technical_Report.pdf) · [專案簡報](docs/WattBot2025_AttentionPlease_Presentation.pdf) · [Baseline notebook](notebook/v1_gemini_2_5pro.ipynb) · [HERO notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb) · [English](#table-of-contents)

## 中文目錄

- [專案介紹](#專案介紹)
- [技術架構](#技術架構)
- [系統架構](#系統架構)
  - [Baseline：BM25 檢索](#baselinebm25-檢索)
  - [HERO：結合視覺證據的混合檢索](#hero結合視覺證據的混合檢索)
  - [架構比較](#架構比較)
- [專案目錄](#專案目錄)
- [資料集與評分方式](#資料集與評分方式)
- [輸出格式](#輸出格式)
- [實驗成果](#實驗成果)
- [Notebook 執行設定](#notebook-執行設定)
- [團隊成員](#團隊成員)
- [引用格式](#引用格式)
- [授權](#授權)
- [致謝](#致謝)

## 專案介紹

WattBot 的核心任務是：**找出支持 AI 永續資訊的實際證據。** 答案可能藏在表格的一個數值、圖像中的標籤，或使用不同單位的文獻段落中。

系統除了回答問題，還需要提供標準化數值、單位、文件引用與支持內容；若資料不足，也必須辨識無法回答的情況。

本專案包含兩套方法：

- **BM25 Baseline**：結合文字與表格檢索、問題路由，以及兩階段答案生成。
- **HERO**：加入版面解析、OCR、稀疏與稠密檢索、加權排名融合、CrossEncoder 重排序及題型專屬提示詞。

研究的關鍵是**表格悖論（Table Paradox）**：加入原始表格可能干擾關鍵字檢索，刪除表格又會失去重要數值。HERO 保留內容類型，透過加權融合提高表格與 OCR 證據的可見度。

## 技術架構

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">項目</th>
      <th width="1500" align="left">使用技術</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>程式語言</td>
      <td>Python</td>
    </tr>
    <tr>
      <td>執行環境</td>
      <td>Jupyter Notebook / Google Colab</td>
    </tr>
    <tr>
      <td>Baseline 解析</td>
      <td>PyMuPDF + pdfplumber</td>
    </tr>
    <tr>
      <td>HERO 解析</td>
      <td>Docling + EasyOCR + OpenCV</td>
    </tr>
    <tr>
      <td>稀疏檢索</td>
      <td>BM25 (rank-bm25)</td>
    </tr>
    <tr>
      <td>稠密檢索</td>
      <td>Sentence-Transformers + BGE-large + FAISS</td>
    </tr>
    <tr>
      <td>重排序</td>
      <td>BGE-reranker-v2-m3</td>
    </tr>
    <tr>
      <td>答案生成</td>
      <td>Google Gemini API</td>
    </tr>
    <tr>
      <td>資料處理</td>
      <td>pandas + NumPy</td>
    </tr>
  </tbody>
</table>

上述設定以公開 notebook 為準；技術報告另包含其他實驗配置。

## 系統架構

### Baseline：BM25 檢索

![Baseline BM25 系統架構](docs/images/baseline-architecture.png)

*擷取自[技術報告第 5 頁 Figure 2](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=5)。*

Baseline 將文件處理與問答分成以下步驟：

1. **文件擷取**：解析 PDF，整理文字與表格區塊。
2. **建立索引**：使用 BM25 進行關鍵字與數值比對。
3. **問題路由**：判斷是否為表格相關問題。Notebook 的 `hybrid_search` 對這類問題合併 **Top-7 文字 + Top-3 表格**，表格分數乘上 **0.8** 後排序；其他問題使用文字檢索。
4. **答案生成**：先以較少證據作答，無法回答時再擴大上下文。
5. **結構化輸出**：產生答案、數值、單位、引用、支持內容與解釋。

報告原圖以純文字與混合索引呈現，notebook 的路由版本則分開建立文字與表格索引。此處的 hybrid 指內容類型混合，這條 baseline 路徑尚未結合向量檢索。

### HERO：結合視覺證據的混合檢索

![HERO 系統架構](docs/images/hero-architecture.png)

*擷取自[技術報告第 7 頁 Figure 4](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=7)。*

HERO 在 Baseline 基礎上加入五個階段：

1. **視覺證據擷取**：Docling 解析文件元素，EasyOCR 與 OpenCV 擷取圖像中的文字，保留區塊類型與文件 ID。
2. **雙路檢索**：BM25 比對精確詞彙與數值，稠密向量補足語意相近但用詞不同的證據。
3. **加權倒數排名融合**：透過 RRF 合併排名，設定**表格 ×2.0、OCR ×1.8、作者／年份匹配 ×2.0**。
4. **CrossEncoder 重排序**：聯合評估問題與候選區塊，優先保留相關證據。
5. **自適應生成**：依題型選擇提示詞，先用 Top-3 上下文回答，需要第二次嘗試時再擴大傳入的證據範圍。

原圖標示的模型為 **BGE-M3**；公開 notebook 實際使用 **BGE-large-en-v1.5**、正規化向量的 FAISS `IndexFlatIP` 與 **BGE-reranker-v2-m3**，並提供 MiniLM 載入失敗備援。

### 架構比較

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">元件</th>
      <th width="750" align="left">BM25 Baseline</th>
      <th width="750" align="left">HERO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">文件解析</td>
      <td align="left">PyMuPDF / pdfplumber</td>
      <td align="left">Docling + EasyOCR</td>
    </tr>
    <tr>
      <td align="left">證據來源</td>
      <td align="left">文字與表格</td>
      <td align="left">文字、表格與圖像 OCR</td>
    </tr>
    <tr>
      <td align="left">檢索方式</td>
      <td align="left">文字／表格 BM25 索引</td>
      <td align="left">BM25 + 稠密向量搜尋</td>
    </tr>
    <tr>
      <td align="left">表格處理</td>
      <td align="left">BM25 分數權重 ×0.8</td>
      <td align="left">RRF 權重 ×2.0</td>
    </tr>
    <tr>
      <td align="left">結果排序</td>
      <td align="left">加權 BM25 分數</td>
      <td align="left">加權 RRF + CrossEncoder</td>
    </tr>
    <tr>
      <td align="left">查詢調整</td>
      <td align="left">表格相關問題路由</td>
      <td align="left">單位擴展與元資料提示</td>
    </tr>
    <tr>
      <td align="left">答案生成</td>
      <td align="left">兩階段 fallback</td>
      <td align="left">題型提示詞與兩階段 fallback</td>
    </tr>
  </tbody>
</table>

## 專案目錄

```text
WattBot-RAG-System/
├── README.md
├── notebook/
│   ├── v1_gemini_2_5pro.ipynb                 # BM25 baseline experiments
│   └── Gemini_2_5_pro_新pipeline_0_821.ipynb  # HERO pipeline
├── data/
│   ├── metadata.csv                           # 32 source documents
│   ├── train_QA.csv                           # 41 training questions
│   └── test_Q.csv                             # 282 test questions
└── docs/
    ├── images/                                # Architecture figures
    ├── WattBot2025_AttentionPlease_Technical_Report.pdf
    └── WattBot2025_AttentionPlease_Presentation.pdf
```

## 資料集與評分方式

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">檔案</th>
      <th width="750" align="left">筆數</th>
      <th width="750" align="left">內容</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><code>metadata.csv</code></td>
      <td align="right">32</td>
      <td align="left">文件資訊與來源網址</td>
    </tr>
    <tr>
      <td align="left"><code>train_QA.csv</code></td>
      <td align="right">41</td>
      <td align="left">問題、標準答案、引用與證據</td>
    </tr>
    <tr>
      <td align="left"><code>test_Q.csv</code></td>
      <td align="right">282</td>
      <td align="left">測試問題與預期單位</td>
    </tr>
  </tbody>
</table>

原始文獻需依 metadata 網址另外下載；`docs/` 的 PDF 是專案報告與簡報。

<table width="100%">
  <thead>
    <tr>
      <th width="500" align="left">WattBot Score 項目</th>
      <th width="750" align="left">權重</th>
      <th width="750" align="left">評分依據</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">答案值</td>
      <td align="right">75%</td>
      <td align="left">數值誤差或標準化類別比對</td>
    </tr>
    <tr>
      <td align="left">引用文件</td>
      <td align="right">15%</td>
      <td align="left">引用集合的 Jaccard overlap</td>
    </tr>
    <tr>
      <td align="left">無法作答判定</td>
      <td align="right">10%</td>
      <td align="left">正確辨識缺乏證據的問題</td>
    </tr>
  </tbody>
</table>

報告設定數值答案的容許相對誤差為 **±0.1%**。

## 輸出格式

生成器先產生結構化 JSON，再轉換成競賽欄位：

<table width="100%">
  <thead>
    <tr>
      <th width="1000" align="left">欄位</th>
      <th width="1000" align="left">說明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><code>id</code>、<code>question</code></td>
      <td align="left">問題編號與內容</td>
    </tr>
    <tr>
      <td align="left"><code>answer</code></td>
      <td align="left">自然語言答案</td>
    </tr>
    <tr>
      <td align="left"><code>answer_value</code>、<code>answer_unit</code></td>
      <td align="left">標準化答案值與單位</td>
    </tr>
    <tr>
      <td align="left"><code>ref_id</code>、<code>ref_url</code></td>
      <td align="left">支持文件 ID 與網址</td>
    </tr>
    <tr>
      <td align="left"><code>supporting_materials</code></td>
      <td align="left">證據摘錄</td>
    </tr>
    <tr>
      <td align="left"><code>explanation</code></td>
      <td align="left">證據如何支持答案</td>
    </tr>
  </tbody>
</table>

## 實驗成果

### 競賽最終排名

<table width="100%">
  <thead><tr><th width="800" align="left">排行榜</th><th width="600" align="left">名次</th><th width="600" align="left">分數</th></tr></thead>
  <tbody>
    <tr><td>Public</td><td>第 2 名</td><td>0.886</td></tr>
    <tr><td>Private — 最終排名</td><td>第 4 名</td><td>0.840</td></tr>
  </tbody>
</table>

於 2026 年 10 月 14 日核對 **Attention Please** 的[官方 Public 排行榜](https://www.kaggle.com/competitions/WattBot2025/leaderboard?tab=public)與[官方 Private 排行榜](https://www.kaggle.com/competitions/WattBot2025/leaderboard?tab=private)。競賽已結束，**最終競賽名次為 Private 第 4 名**。Public 約使用 66% 測試資料，Private 使用其餘 34%。報告中的 0.886 是 Public 分數，最終 Private 分數為 0.840。

實驗圍繞三個問題：**加入表格是否有利於 BM25、HERO 是否改善證據檢索，以及不同語言模型是否都能受益？** 檢索評估使用內部問題集，模型比較沿用報告中的競賽分數，測試檔案共有 282 題；官方 Public 與 Private 分數各使用其中不同子集。

### 實驗配置

本專案從表格雜訊實驗出發，發展路由式 Baseline，再加入 HERO 的混合檢索。以下配置對應這些階段：

<table width="100%">
  <thead><tr><th width="600" align="left">配置</th><th width="1400" align="left">定義</th></tr></thead>
  <tbody>
    <tr><td>Hybrid BM25</td><td>文字加原始 Markdown 表格，用於表格悖論實驗。</td></tr>
    <tr><td>Text-Only BM25</td><td>移除表格，作為同一實驗的對照配置。</td></tr>
    <tr><td>路由式 BM25 baseline</td><td>前方架構介紹的版本：依問題混合文字／表格檢索，表格分數乘上 0.8。</td></tr>
  </tbody>
</table>

檢索表格依報告 Table 1、Table 16 列示，模型比較依 Figure 5 列示。評估範圍與 notebook 留存輸出的差異集中說明於本節末尾。

### 1. 表格悖論：純文字與混合 BM25 比較

Baseline 比較「文字加原始 Markdown 表格」與「純文字」兩種語料。在 **39 題評估**中，純文字版本的所有指標都優於或等於混合版本。

<table width="100%">
  <thead><tr>
    <th width="666" align="left">指標</th>
    <th width="666" align="left">混合：文字＋表格</th>
    <th width="666" align="left">純文字</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">Recall@1</td><td align="right">79.49%</td><td align="right">87.18%</td></tr>
    <tr><td align="left">Recall@3</td><td align="right">89.74%</td><td align="right">92.31%</td></tr>
    <tr><td align="left">Recall@5</td><td align="right">92.31%</td><td align="right">92.31%</td></tr>
    <tr><td align="left">Recall@10</td><td align="right">92.31%</td><td align="right">94.87%</td></tr>
    <tr><td align="left">nDCG@1</td><td align="right">0.7949</td><td align="right">0.8718</td></tr>
    <tr><td align="left">nDCG@3</td><td align="right">0.8497</td><td align="right">0.8859</td></tr>
    <tr><td align="left">nDCG@5</td><td align="right">0.8564</td><td align="right">0.8859</td></tr>
    <tr><td align="left">nDCG@10</td><td align="right">0.8617</td><td align="right">0.8950</td></tr>
    <tr><td align="left">MRR</td><td align="right">0.8526</td><td align="right">0.8974</td></tr>
  </tbody>
</table>

來源：[報告第 4 頁 Table 1](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=4)。這裡的 hybrid 指文字與表格混合，不是稀疏與稠密檢索。

移除原始表格後，Recall@1 提高 **7.69 個百分點**，MRR 增加 **0.0448**。報告將原因歸於表格格式雜訊：分隔符號與破碎結構稀釋關鍵詞的 BM25 訊號。然而，刪除表格也會失去數值證據。這個取捨促成 HERO 的結構化解析、語意檢索與表格加權設計。

### 2. HERO 檢索品質

HERO 透過結構化擷取、稀疏與稠密檢索及重排序改善證據選擇。下表呈現報告中的檢索比較，配置與評估樣本差異集中列於評估說明。

<table width="100%">
  <thead><tr>
    <th width="500" align="left">指標</th>
    <th width="500" align="left">Baseline（報告 Table 16）</th>
    <th width="500" align="left">HERO（報告 Table 16）</th>
    <th width="500" align="left">差異</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">Recall@1</td><td align="right">79.49%</td><td align="right">80.49%</td><td align="right">+1.00%</td></tr>
    <tr><td align="left">Recall@3</td><td align="right">89.74%</td><td align="right">89.80%</td><td align="right">+0.06%</td></tr>
    <tr><td align="left">Recall@5</td><td align="right">92.31%</td><td align="right">92.68%</td><td align="right">+0.37%</td></tr>
    <tr><td align="left">Recall@10</td><td align="right">92.31%</td><td align="right">95.12%</td><td align="right">+2.81%</td></tr>
    <tr><td align="left">nDCG@1</td><td align="right">0.7949</td><td align="right">0.8049</td><td align="right">+0.0100</td></tr>
    <tr><td align="left">nDCG@3</td><td align="right">0.8497</td><td align="right">0.8502</td><td align="right">+0.0005</td></tr>
    <tr><td align="left">nDCG@5</td><td align="right">0.8564</td><td align="right">0.8584</td><td align="right">+0.0020</td></tr>
    <tr><td align="left">nDCG@10</td><td align="right">0.8617</td><td align="right">0.8665</td><td align="right">+0.0048</td></tr>
    <tr><td align="left">MRR</td><td align="right">0.8526</td><td align="right">0.8560</td><td align="right">+0.0034</td></tr>
  </tbody>
</table>

來源：[報告第 9 頁 Table 16](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=9)。Recall 差異以 `%` 顯示兩個百分比直接相減的結果（百分點變化），並非相對成長率。上表忠實呈現報告值，與 notebook 留存輸出的差異另列於下方評估說明。

- **證據覆蓋率**：Recall@10 從 92.31% 提升至 95.12%，增加 **2.81 個百分點**。
- **首位命中**：Recall@1 增加 **1.00 個百分點**，代表更多問題能在第一個檢索位置找到相關引用。
- **排序品質**：報告中的 nDCG@10 增加 **0.0048**，MRR 增加 **0.0034**。

Text-Only BM25 的報告 Recall@1 與 MRR 較高，HERO 的 Recall@10 則略高；這些結果並非已驗證的同樣本比較。下方跨模型答案分數的提升，更直接呈現 HERO 在完整問答流程中的價值。

### 3. 視覺證據擷取後的語料組成

Docling 與 EasyOCR 將可搜尋內容從一般段落擴展至表格和圖像文字。報告的語料統計共 **8,407 個區塊**：

<table width="100%">
  <thead><tr>
    <th width="666" align="left">區塊類型</th>
    <th width="666" align="left">數量</th>
    <th width="666" align="left">占比</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">文字</td><td align="right">7,817</td><td align="right">93.0%</td></tr>
    <tr><td align="left">表格</td><td align="right">153</td><td align="right">1.8%</td></tr>
    <tr><td align="left">圖像 OCR</td><td align="right">437</td><td align="right">5.2%</td></tr>
    <tr><td align="left">總計</td><td align="right">8,407</td><td align="right">100%</td></tr>
  </tbody>
</table>

來源：[報告第 6 頁 Table 8](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=6)。其中 437 個區塊來自圖像 OCR，153 個區塊保留表格內容。類型標籤讓檢索階段能為不同證據來源配置權重。

數量統一採用 Table 8，反映可搜尋證據規模，不代表 OCR 的獨立準確率。

### 4. 不同模型與 Pipeline 比較

<table width="100%">
  <thead><tr>
    <th width="500" align="left">模型</th>
    <th width="500" align="left">BM25 Baseline</th>
    <th width="500" align="left">HERO</th>
    <th width="500" align="left">分數提升</th>
  </tr></thead>
  <tbody>
    <tr><td align="left">GPT-4o-mini</td><td align="right">0.753</td><td align="right">0.789</td><td align="right">+0.036</td></tr>
    <tr><td align="left">GPT-4o</td><td align="right">0.768</td><td align="right">0.811</td><td align="right">+0.043</td></tr>
    <tr><td align="left">GPT-5.1</td><td align="right">0.778</td><td align="right">0.807</td><td align="right">+0.029</td></tr>
    <tr><td align="left">Gemini 2.5 Pro</td><td align="right">0.807</td><td align="right">0.821</td><td align="right">+0.014</td></tr>
    <tr><td align="left">Gemini 3 Pro</td><td align="right">0.853</td><td align="right">0.886</td><td align="right">+0.033</td></tr>
  </tbody>
</table>

來源：[報告第 10 頁 Figure 5](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=10)。報告中五種模型搭配 HERO 均有成績提升；此表與上方最終 Private 排名分開呈現。

**GPT-4o 的絕對提升最大（+0.043）**，**Gemini 3 Pro 的報告分數最高（0.886，亦已由 Public 榜單確認）**。Gemini 3 Pro 從 0.853 提升至 0.886，說明即使搭配較強的生成模型，改善證據檢索仍有幫助。Gemini 2.5 Pro 則從 0.807 提升至 0.821，也是公開 HERO notebook 初始化使用的模型。

### 5. 結論與未來方向

實驗呈現從表格雜訊到檢索改造的完整脈絡：保留文字、表格與 OCR 類型，結合詞彙與語意搜尋，再重排序候選證據，依問題調整生成上下文。

報告提出未來可研究**學習式融合權重**、**多語言語料**與**大型文件集合的階層索引**。目前權重與問題路由仍是啟發式規則；兩階段生成在較小上下文與較廣 fallback 之間取捨，但上表未量化其獨立效果。

<details>
<summary>評估範圍與重現細節</summary>

Baseline 表格實驗使用 39 題；HERO 最後留存的評估使用 41 題，包含不可回答問題。其輸出記錄 Recall@3 為 87.80%、nDCG@3 為 0.8384、MRR 為 0.8519，與報告 Table 16 不同。因此報告比較應視為文件記錄的結果，而非本次驗證的同樣本實驗；純文字 baseline 在部分指標上也高於上述 HERO 數值。

Notebook 的 Recall@K 判斷前 K 個結果是否命中任一標準引用，實際上是命中率。多個區塊可來自同一文件，區塊排名與去重文件排名不能混用。留存評估也包含不可回答問題，會影響指標解讀。

公開 HERO 檢索預設保留八個最終區塊；生成器要求十個 fallback 區塊時，仍只能使用實際傳入的內容。目前所有問題均啟用重排序，拒答由規則與 LLM 回應決定，不是校準後的信心估計。本節數據來自報告與留存輸出，沒有重新執行模型或競賽評分。

</details>

## Notebook 執行設定

以下依公開程式整理，**尚未從乾淨環境驗證完整執行成功**。Notebook 保留實驗 cell，需要手動設定。

### HERO Notebook

1. 在 Google Colab 開啟 [HERO notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb)，將 `data/` 的三份 CSV 上傳到 `/content/`。
2. 下列套件清單依流程使用的匯入整理，尚未驗證版本相容性；請先檢查 notebook 已有的安裝 cell：

```python
%pip install docling easyocr opencv-python-headless rank-bm25 sentence-transformers faiss-cpu pandas numpy requests pdfminer.six pillow tqdm google-genai
```

3. 在 **Docling／OCR 處理 cell 之前**補上以下匯入。公開版本在較晚才匯入 `np`，生成階段則使用未匯入的 `time` 與 `types`：

```python
import numpy as np
import time
from google.genai import types
```

4. 在 Gemini 生成 cell 中，替換原本的佔位金鑰與 client 初始化。若使用 Colab Secrets，建立 `GOOGLE_API_KEY` 並允許 notebook 存取，再使用：

```python
from google.colab import userdata
from google import genai

client = genai.Client(api_key=userdata.get("GOOGLE_API_KEY"))
```

5. 依序執行：CSV 載入 → PDF 下載 → EasyOCR 初始化 → Docling／OCR 處理 → BM25／向量索引 → 檢索與重排序函式 → Gemini 生成與匯出。評估與清除快取 cell 為選用步驟。

程式使用的路徑如下：

```text
/content/metadata.csv
/content/train_QA.csv
/content/test_Q.csv
/content/papers/
/content/wattbot_working/chunks/structured_chunks_with_ocr.jsonl
/content/submission_enhanced.csv
```

生成答案前，請檢查文件下載失敗清單與 OCR 區塊數。OCR 也受 Docling 版本與圖像物件格式影響，補上匯入不代表已解決所有相容性問題。流程需要下載模型、可用的 API 存取與原始文獻。

### Baseline Notebook

[Baseline notebook](notebook/v1_gemini_2_5pro.ipynb) 收錄了多組實驗。若要準備資料並執行其中的 Gemini 答案生成程式，請依序檢查：

1. 在 Google Colab 開啟 notebook，執行 Kaggle 登入、安裝套件與下載競賽資料的程式格。你需要有 WattBot 2025 競賽資料的存取權。下載時會印出 `Downloaded to: ...`，請記下這個資料夾位置。
2. 後面的程式有幾處把資料夾寫死為 `/root/.cache/kagglehub/competitions/WattBot2025`。請改成第 1 步印出的實際位置，尤其是讀取資料的 `INPUT_DIR`，以及最後 `main()` 裡的 `INPUT_DIR`。PDF 解析測試的 `data_dir` 也使用同一個寫死的路徑。
3. 執行產生 `/content/hybrid_chunks.json` 所需的文件處理程式格。最後的答案生成程式會讀取這個檔案，以及 `INPUT_DIR` 中的 `metadata.csv` 和 `test_Q.csv`；執行前先確認三個檔案都存在。
4. 執行最後的答案生成程式格之前，先確認已安裝 `google-genai`；notebook 會匯入它，但沒有對應的安裝程式格。將 `GEMINI_API_KEY = "# ⚠️ 請替換成你的 API Key"` 這行的佔位文字換成有效金鑰，再執行該格產生提交用 CSV。這格會自動呼叫 `main()`，並將測試題送到 Gemini API。分享 notebook 前，請移除金鑰。

Notebook 另有 PDF 解析與檢索測試；它們是獨立實驗，不是最後答案生成函式的額外步驟。安裝程式格會安裝 Python 套件，OCR 所需的 Poppler 與 Tesseract 則透過 `apt-get` 安裝。以上步驟依公開程式整理，尚未從乾淨環境完整驗證。

## 團隊成員

**Team Attention Please**

- **Shao-Hua Wu**‡ - 文件處理、檢索系統、專案統籌
- **Pei-Ju Hsieh**‡ - 文件處理、檢索系統、混合搜尋實作、答案生成、提示詞設計與優化
- **Bo-Hao Chen**‡ - 答案生成、提示詞設計與優化
- **Yi-Chen Hsiao**∗ - 報告製作
- **Yi-Yang Xue**† - 檢索系統、混合搜尋實作

<sup>‡ 共同貢獻</sup>

## 引用格式

若本專案對你的研究或實作有幫助，歡迎使用以下格式引用：

```bibtex
@misc{wattbot2025_hero,
  title = {WattBot 2025: Retrieval-Augmented Generation for ESG Intelligence},
  author = {Wu, Shao-Hua and Hsieh, Pei-Ju and Chen, Bo-Hao and Hsiao, Yi-Chen and Xue, Yi-Yang},
  year = {2025},
  url = {https://github.com/AnnieHsieh-12/WattBot-RAG-System}
}
```

## 授權

目前 repo 未包含 `LICENSE` 檔案，尚未在 repository 中明定授權條款。

## 致謝

- WattBot 競賽主辦單位：威斯康辛大學麥迪遜分校 ML+X。
- Google 提供 Gemini API 存取。
- 提供本專案所使用工具與函式庫的開源社群。

相關研究與啟發團隊設計的 KohakuRAG 案例，可參閱[技術報告](docs/WattBot2025_AttentionPlease_Technical_Report.pdf)。

[回到英文版](#table-of-contents) · [回到中文目錄](#中文目錄)
