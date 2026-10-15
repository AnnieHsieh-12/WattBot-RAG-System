# WattBot 2025 - Team Name: Attention Please 🔋

<div align="center">

<img src="docs/images/wattbot-logo.png" alt="WattBot — Team Attention Please" width="800">

[![Competition](https://img.shields.io/badge/Kaggle-WattBot%202025-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/WattBot2025)
![Team](https://img.shields.io/badge/Team-Attention%20Please-FF6B6B?style=for-the-badge)
![System](https://img.shields.io/badge/System-HERO-9B59B6?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

### HERO: Hybrid Evidence Retrieval Optimization

*Evidence-based Energy Estimation for AI Workloads using Retrieval-Augmented Generation*

[📄 Technical Report](docs/WattBot2025_AttentionPlease_Technical_Report.pdf) | [📊 Presentation](docs/WattBot2025_AttentionPlease_Presentation.pdf)

[Baseline Notebook](notebook/v1_gemini_2_5pro.ipynb) · [HERO Notebook](notebook/Gemini_2_5_pro_新pipeline_0_821.ipynb) · [繁體中文](README.zh-TW.md)

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
- [Related Work](#related-work)
- [License](#license)
- [繁體中文版](README.zh-TW.md)

## Introduction

WattBot 2025 evaluates systems that answer questions about the environmental impact of AI workloads using evidence from technical documents. Many answers depend on values in tables or figures and require unit normalization.

This repository contains two approaches developed by Team Attention Please. Both return a normalized answer, document references, supporting text, and an abstention when the source documents do not contain enough evidence:

- **BM25 Baseline** — text and table retrieval with query routing and two-stage answer generation.
- **HERO** — layout-aware parsing and OCR, sparse–dense retrieval, weighted rank fusion, CrossEncoder reranking, and question-specific generation.

A baseline experiment found that adding raw Markdown tables reduced several BM25 retrieval metrics, while removing tables also removed useful numerical evidence. HERO keeps text, table, and OCR chunks separate and weights them independently during retrieval.

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

The entries above reflect the public notebooks. The technical report also records experimental configurations that are not included in those notebooks.

## System Architecture

### Baseline: BM25 Retrieval

![Baseline BM25 architecture](docs/images/baseline-architecture.png)

*Figure 2 from the [technical report, page 5](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=5).*

The baseline has five steps:

1. **Document extraction.** Parse PDFs and prepare text and table chunks.
2. **Sparse indexing.** Build BM25 indexes for lexical matching.
3. **Query routing.** Identify table-related questions using keywords. The notebook's `hybrid_search` combines **Top-7 text + Top-3 table** results for these questions, applying a **0.8× table-score weight**. Other questions use text retrieval.
4. **Answer generation.** First answer from a small evidence set; if unsuccessful, expand the context and try again.
5. **Structured output.** Return the answer, value, unit, references, supporting evidence, and explanation.

The report diagram presents text-only and hybrid indexes; the notebook routing implementation separates text and table indexes. Here, “hybrid” refers to content types. This baseline routing path does not combine BM25 with dense retrieval.

### HERO: Visual-Aware Hybrid Retrieval

![HERO architecture](docs/images/hero-architecture.png)

*Figure 4 from the [technical report, page 7](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=7).*

HERO adds five stages to the baseline:

1. **Visual-aware ingestion.** Docling extracts document elements; EasyOCR and OpenCV turn text inside figures into searchable evidence. Chunks retain their content type and document ID.
2. **Dual retrieval.** BM25 handles exact terms and numbers, while dense embeddings retrieve semantically related passages.
3. **Weighted reciprocal rank fusion.** Combine sparse and dense rankings, boosting **tables ×2.0**, **OCR ×1.8**, and **author/year matches ×2.0**.
4. **CrossEncoder reranking.** Jointly score each query–chunk pair to prioritize relevant evidence.
5. **Question-specific generation.** Choose prompts by question type, answer from Top-3 context first, and expand the supplied context when a second attempt is needed.

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
├── README.zh-TW.md
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

Source PDFs are not stored in this repository; they are downloaded from the URLs in `metadata.csv`. The PDFs in `docs/` are the project report and presentation.

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

The [official Public leaderboard](https://www.kaggle.com/competitions/WattBot2025/leaderboard?tab=public) and [official Private leaderboard](https://www.kaggle.com/competitions/WattBot2025/leaderboard?tab=private) were checked on October 14, 2026. The competition has ended, so **Private rank 4 is the final placement**. The report's 0.886 result is the Public score; the final Private score is 0.840. Public and Private use different portions of the 282-question test set.

The report evaluates three comparisons: BM25 with and without raw tables, baseline and HERO retrieval, and baseline and HERO answer scores across five language models. Retrieval metrics use internal question sets; the model comparison uses competition scores reported by the team.

### Experiment Configurations

The report uses three BM25 configurations:

<table width="100%">
  <thead><tr><th width="600" align="left">Configuration</th><th width="1400" align="left">Meaning</th></tr></thead>
  <tbody>
    <tr><td>Hybrid BM25</td><td>Text plus raw Markdown tables; used in the table-paradox experiment.</td></tr>
    <tr><td>Text-Only BM25</td><td>Tables removed; the control configuration in that experiment.</td></tr>
    <tr><td>Routed BM25 baseline</td><td>The architecture described above: question-dependent text/table retrieval with a 0.8× table-score weight.</td></tr>
  </tbody>
</table>

The retrieval tables below reproduce report Tables 1 and 16; the model comparison reproduces Figure 5. Differences between the report and the saved notebook output are noted at the end of this section.

### 1. The Table Paradox: Text-Only vs. Hybrid BM25

The baseline experiment compares text with raw Markdown tables against a text-only corpus. Across **39 queries**, the text-only version matches or exceeds the hybrid version on every reported metric.

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

Removing raw tables raises Recall@1 by **7.69 percentage points** and MRR by **0.0448**. The report attributes this to table delimiters and disrupted structure weakening BM25 term matching. Because tables still contain necessary numerical evidence, HERO parses and weights them separately instead of removing them.

### 2. HERO Retrieval Performance

HERO combines structured ingestion, sparse–dense retrieval, and reranking. The table below reproduces the report's retrieval comparison; sample differences are documented in the evaluation note.

<table width="100%">
  <thead><tr>
    <th width="500" align="left">Metric</th>
    <th width="500" align="left">Baseline</th>
    <th width="500" align="left">HERO</th>
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

The largest reported difference is Recall@10 at **+2.81 percentage points**. Recall@1 changes by **+1.00 percentage point**, nDCG@10 by **+0.0048**, and MRR by **+0.0034**. Text-Only BM25 reports higher Recall@1 and MRR than HERO, while HERO reports higher Recall@10. These runs are not a verified same-sample comparison.

### 3. Evidence Coverage After Visual-Aware Ingestion

Report Table 8 lists **8,407 searchable chunks** produced with Docling and EasyOCR:

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

Source: [report Table 8, page 6](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=6). The counts describe searchable evidence, not OCR accuracy. Content labels allow the retrieval stage to assign different weights to prose, tables, and figure text.

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
    <tr><td align="left">Gemini 3.0 Pro</td><td align="right">0.853</td><td align="right">0.886</td><td align="right">+0.033</td></tr>
  </tbody>
</table>

Source: [report Figure 5, page 10](docs/WattBot2025_AttentionPlease_Technical_Report.pdf#page=10). The report lists higher HERO scores for all five models. This comparison is separate from the final Private standings above.

The largest reported difference is **+0.043 for GPT-4o**. **Gemini 3.0 Pro has the highest reported score, 0.886**, which matches the Public leaderboard score. The public HERO notebook initializes Gemini 2.5 Pro; its reported score changes from 0.807 to 0.821.

### 5. Limitations and Future Work

HERO's fusion weights and question routing are heuristic. The reported experiments also do not isolate the effect of two-stage generation from the retrieval changes. The report lists **learned fusion weights**, **multilingual corpora**, and **hierarchical indexing for larger collections** as future work.

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
2. Install the dependencies used by the workflow. Version compatibility has not been tested, so review the notebook's existing installation cells first:

```python
%pip install docling easyocr opencv-python-headless rank-bm25 sentence-transformers faiss-cpu pandas numpy requests pdfminer.six pillow tqdm google-genai
```

3. In the Gemini generation cell, replace the existing placeholder key and client initialization. For Colab Secrets, create a secret named `GOOGLE_API_KEY`, allow notebook access, then use:

```python
from google.colab import userdata
from google import genai

client = genai.Client(api_key=userdata.get("GOOGLE_API_KEY"))
```

4. Execute the workflow in this order: CSV loading → PDF download → EasyOCR initialization → Docling/OCR processing → BM25/dense indexes → retrieval and reranking helpers → Gemini generator and submission. Evaluation and cache-cleanup cells are optional.

The code uses these locations:

```text
/content/metadata.csv
/content/train_QA.csv
/content/test_Q.csv
/content/papers/
/content/wattbot_working/chunks/structured_chunks_with_ocr.jsonl
/content/submission_enhanced.csv
```

Check download failures and the OCR chunk count before generating answers. A clean run also requires compatible Docling image objects, model downloads, API access, and the source documents.

### Baseline Notebook

The [baseline notebook](notebook/v1_gemini_2_5pro.ipynb) contains several experiments. To prepare its data and run the Gemini answer-generation section:

1. Open the notebook in Google Colab. Run its Kaggle login, dependency-installation, and competition-download cells. You need access to the WattBot 2025 competition data. The download cell prints `Downloaded to: ...`; keep that path.
2. Replace the hard-coded `/root/.cache/kagglehub/competitions/WattBot2025` path in later cells with the path printed in step 1. In particular, update `INPUT_DIR` in the data-loading cell and in `main()` at the end. The PDF parsing test also sets `data_dir` to that hard-coded path.
3. Run the document-processing cells needed to create `/content/hybrid_chunks.json`. The final generation cell reads that file, plus `metadata.csv` and `test_Q.csv` from `INPUT_DIR`. Check that all three exist before running it.
4. Before the final generation cell, make sure `google-genai` is installed; the notebook imports it but has no installation cell for it. Replace the placeholder `GEMINI_API_KEY = "# ⚠️ 請替換成你的 API Key"` with a valid key, then run the cell to generate a submission CSV. It calls `main()` automatically and sends the test questions to the Gemini API. Keep your key out of any notebook you share.

The notebook also includes separate PDF parsing and retrieval tests that are not part of the final generation function. Its setup cells install Python packages, Poppler, and Tesseract. A clean end-to-end run has not been verified.

## Team Members

**Team Attention Please**

- **Shao-Hua Wu** - Department of Money and Banking, National Chengchi University
- **Pei-Ju Hsieh** - Department of Money and Banking, National Chengchi University
- **Bo-Hao Chen** - Department of Money and Banking, National Chengchi University
- **Yi-Chen Hsiao** - Department of Finance, National Chengchi University
- **Yi-Yang Xue** - Department of Computer Science, National Chengchi University

## Related Work

The [technical report](docs/WattBot2025_AttentionPlease_Technical_Report.pdf) includes the literature review and a KohakuRAG case study.

## License

No `LICENSE` file is currently included in this repository. Licensing terms have not been specified in the repository.
