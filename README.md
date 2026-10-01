# WattBot 2025 - Team Attention Please 🔋

<div align="center">

[![Competition](https://img.shields.io/badge/Kaggle-WattBot%202025-20BEFF?style=for-the-badge&logo=kaggle)](https://kaggle.com/competitions/WattBot2025)
[![Team](https://img.shields.io/badge/Team-Attention%20Please-FF6B6B?style=for-the-badge)]()
[![System](https://img.shields.io/badge/System-HERO-9B59B6?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()

### **HERO: Hierarchical Evidence Retrieval & Optimization**
*Evidence-based Energy Estimation for AI Workloads using Retrieval Augmented Generation*

[📄 Technical Report](docs/WattBot2025_AttentionPlease_Technical_Report.pdf) | [📊 Presentation](docs/WattBot2025_AttentionPlease_Presentation.pdf)

</div>

---

## 📌 Competition Overview

**WattBot 2025** challenges participants to build RAG systems that extract credible environmental impact estimates from academic literature. Our system must provide:

- 🎯 Concise, citation-backed answers
- 📚 Document IDs and supporting evidence
- ❌ Explicit handling of unanswerable questions

### Dataset
- **32 scholarly articles** (2019-2025) on AI's environmental impact
- **Topics**: Energy consumption, carbon emissions, water usage, sustainability
- **Question Types**: Numeric values, categorical terms, True/False

### Evaluation Metrics
| Component | Weight | Criteria |
|-----------|--------|----------|
| `answer_value` | 75% | Numeric accuracy (±0.1% tolerance) or exact categorical match |
| `ref_id` | 15% | Jaccard overlap with ground truth citations |
| `is_NA` | 10% | Proper handling of unanswerable questions |

---

## 🏆 Results

| Metric | Score |
|--------|-------|
| **Final WattBot Score** | 0.886 |
| **Public Leaderboard Rank** | 1 |
| **Private Leaderboard Rank** | 1 |

---

## 🎯 Solution Architecture

Our **HERO (Hierarchical Evidence Retrieval & Optimization)** pipeline consists of four key components:

### 🌟 What Makes HERO Different?

| Feature | HERO Approach | Traditional RAG |
|---------|---------------|-----------------|
| **Document Structure** | Preserves hierarchy (Doc→Section→Leaf) | Flat chunking |
| **Retrieval Strategy** | Hybrid (BM25 + Dense) with reranking | Single-method |
| **Storage** | SQLite single-file | External vector DBs |
| **Visual Processing** | OCR + Table extraction | Text-only |
| **Citation Handling** | Multi-document evidence aggregation | Single-source |

### 1. **Document Processing** 📄
```
PDF Documents → Hierarchical Parsing → Structured Chunks
```
- **Parser**: PyMuPDF for text extraction
- **Structure Preservation**: Document → Section → Paragraph hierarchy
- **Visual Processing**: OCR for tables/figures containing critical metrics
- **Chunking Strategy**: ~150-180 words per chunk with overlap

### 2. **Hybrid Retrieval System** 🔍
```
Query → [BM25 + Dense Embeddings] → Top-K Chunks
```
- **BM25**: Keyword-based retrieval for precise term matching
- **Dense Retrieval**: Sentence-Transformers for semantic search
- **Reranking**: Weighted combination (α=0.6 for BM25, β=0.4 for Dense)
- **Storage**: SQLite for single-file reproducibility

### 3. **Answer Generation** 🤖
```
Retrieved Context + Question → Gemini 2.5 Pro → Structured Answer
```
- **Model**: Google Gemini 2.5 Pro (gemini-2.0-flash-exp)
- **Prompt Engineering**: Few-shot examples + structured output format
- **Rate Limiting**: Exponential backoff for API stability
- **Output**: JSON with answer, citations, and supporting evidence

### 4. **Post-Processing** ✅
```
Raw Answers → Validation → Unit Normalization → Final Submission
```
- Numeric validation (±0.1% tolerance)
- Citation format checking
- "Unable to answer" fallback handling

---

## 🔑 Key Technical Insights

Based on our experimental results:

### 1. **Structure > Chunking**
> 💡 **Lesson**: Do not treat PDFs as flat text strings. Preserving document hierarchy (Doc → Section → Leaf) allows precise targeting.

**Implementation**:
- Tracked section headers and metadata
- Maintained parent-child relationships in chunks
- Enabled context-aware retrieval

### 2. **Engineering Simplicity**
> 💡 **No Vector DB**: Using SQLite as a single-file datastore reduces complexity and ensures reproducibility.

**Benefits**:
- ✅ No external dependencies
- ✅ Easy version control
- ✅ Portable across environments
- ✅ Resilient to API rate limits with backoff logic

### 3. **Visuals are Data, Not Noise**
> 💡 **Lesson**: Critical energy metrics are often hidden in charts. We must process images (via Vision Models/OCR), not just filter them out.

**Strategy**:
- Extracted figures/tables as separate chunks
- Used OCR for table data
- Linked visual content to text context

---

## 📁 Repository Structure

```
.
├── notebooks/
│   ├── v1_gemini_2_5pro.ipynb              # Initial pipeline implementation
│   └── Gemini_2_5_pro_新pipeline_0_821.ipynb  # Optimized version (0.886 score)
├── docs/
│   ├── WattBot2025_AttentionPlease_Technical_Report.pdf
│   └── WattBot2025_AttentionPlease_Presentation.pdf
├── data/
│   ├── metadata.csv                         # Document index
│   ├── train_QA.csv                         # Training Q&A pairs
│   └── test_Q.csv                           # Test questions
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pymupdf rank-bm25 sentence-transformers google-generativeai pandas
```

### Environment Setup
```bash
# Set Google API Key
export GOOGLE_API_KEY="your-api-key-here"
```

### Run the Pipeline
```bash
# Open the notebook
jupyter notebook notebooks/Gemini_2_5_pro_新pipeline_0_821.ipynb

# Or run as script (if converted)
python src/main.py --input data/test_Q.csv --output outputs/submission.csv
```

### Expected Output
```
✅ Document Parsing: 32/32 papers processed
📊 Total chunks: 1778
🔍 Hybrid Search: BM25 + Dense retrieval ready
🤖 Generating answers...
Progress: 100% |████████████████████| 250/250
💾 Submission saved: outputs/submission.csv
```

---

## 📊 Performance Breakdown

### Retrieval Performance

| Metric | Baseline | HERO (Ours) | Δ |
|--------|----------|-------------|---|
| **Recall@K (Coverage)** |
| Recall@1 | 79.49% | **80.49%** | +1.00% |
| Recall@3 | 89.74% | **89.80%** | +0.06% |
| Recall@5 | 92.31% | **92.68%** | +0.37% |
| Recall@10 | 92.31% | **95.12%** | +2.81% |
| **nDCG@K (Ranking Quality)** |
| nDCG@1 | 0.7949 | **0.8049** | +0.0100 |
| nDCG@3 | 0.8497 | **0.8502** | +0.0005 |
| nDCG@5 | 0.8564 | **0.8584** | +0.0020 |
| nDCG@10 | 0.8617 | **0.8665** | +0.0048 |
| **Overall Accuracy** |
| MRR | 0.8526 | **0.8560** | +0.0034 |

### Answer Accuracy by Question Type
| Type | Count | Accuracy |
|------|-------|----------|
| Numeric | ~40% | [TBD] |
| Categorical | ~35% | [TBD] |
| True/False | ~15% | [TBD] |
| Unanswerable | ~10% | [TBD] |

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Document Parsing** | PyMuPDF, regex |
| **Keyword Retrieval** | BM25 (rank-bm25) |
| **Dense Retrieval** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Vector Storage** | SQLite |
| **LLM** | Google Gemini 2.5 Pro |
| **Development** | Python 3.10+, Jupyter Notebook |

---

## 🧪 Experimental Findings

### What Worked ✅
1. **Hybrid retrieval** outperformed single-method approaches by 12%
2. **Hierarchical chunking** improved citation accuracy by preserving context
3. **Few-shot prompting** reduced hallucinations in numeric answers
4. **Exponential backoff** handled API rate limits gracefully

### What Didn't Work ❌
1. Pure dense retrieval missed exact term matches (e.g., "BERT-base")
2. Large chunks (>300 words) diluted relevant information
3. Zero-shot prompting generated inconsistent citation formats

### Ablation Study
| Configuration | WattBot Score | Δ |
|---------------|---------------|---|
| BM25 only | 0.64 | -0.18 |
| Dense only | 0.71 | -0.11 |
| **HERO (Hybrid)** | **0.82** | **baseline** |
| HERO + Visual processing | 0.85 | +0.03 |

---

## 📚 Key References

1. **Competition Dataset**: Endemann, C., Paul, D. J., & Zhao, A. (2025). *WattBot 2025*. Kaggle.
2. **Retrieval Methods**: Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*.
3. **RAG Survey**: Gao, Y., et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*.

---

## 👥 Team Members

**Team Attention Please**

- **Shao-Hua Wu**‡ - Document Processing & Retrieval System & Project Lead
- **Xie-Pei Ju**‡ - Document Processing & Retrieval System & Hybrid Search Implementation & Answer Generation & Prompt Engineering & Optimization
- **Bo-Hao Chen**‡ - Answer Generation & Prompt Engineering & Optimization
- **Yi-Chen Hsiao**∗ - Report Generation
- **Yi-Yang Xue**† - Retrieval System & Hybrid Search Implementation 

<sup>‡ Equal contribution</sup>

---

## 📝 Citation

If you find our approach useful, please cite:

```bibtex
@misc{wattbot2025_hero,
  title={HERO: Hierarchical Evidence Retrieval \& Optimization for WattBot 2025},
  author={Wu, Shao-Hua and Ju, Xie-Pei and Chen, Bo-Hao and Hsiao, Yi-Chen and Xue, Yi-Yang},
  year={2025},
  howpublished={\url{https://github.com/your-repo/wattbot2025-hero}}
}
```

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Competition organizers at ML+X, University of Wisconsin-Madison
- Google for Gemini API access
- Open-source community for tools and libraries

---

<div align="center">

**⭐ If you found this helpful, please consider starring the repository!**

[![Star on GitHub](https://img.shields.io/github/stars/your-username/wattbot2025?style=social)](https://github.com/your-username/wattbot2025)

</div>
