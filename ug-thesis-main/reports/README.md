# Project Reports — Multi-Stream Deepfake Detection

## Md Noushad Jahan Ramim | April 18, 2026

---

| #   | Report                                                    | Contents                                                     | Audience      |
| --- | --------------------------------------------------------- | ------------------------------------------------------------ | ------------- |
| 01  | [Project Overview](report_01_project_overview.md)         | Complete summary: what we built, problems fixed, all results | Everyone      |
| 02  | [Code Explained](report_02_code_explained.md)             | Every important file explained in plain language             | Non-technical |
| 03  | [Figures Explained](report_03_figures_explained.md)       | Every chart, graph, heatmap and what it means                | Everyone      |
| 04  | [Architecture](report_04_architecture.md)                 | Full model architecture with diagrams and equations          | Technical     |
| 05  | [Defense Q&A](report_05_defense_qa.md)                    | 15 hard viva questions with model answers                    | Student       |
| 06  | [Dataset & Pipeline](report_06_dataset_pipeline.md)       | How data was collected, cleaned, and split                   | Technical     |
| 07  | [Experimental Results](report_07_experimental_results.md) | All numbers, all tables, all metrics                         | Researcher    |
| 08  | [Baseline Comparison](report_08_baseline_comparison.md)   | Our model vs CNNDetect vs UnivFD in detail                   | Researcher    |
| 09  | [Ablation Study](report_09_ablation_study.md)             | What each stream contributes and why                         | Researcher    |
| 10  | [Publication Guide](report_10_publication_guide.md)       | How to write the paper, target venues, timeline              | Student       |

---

**Key Results at a Glance:**

- Test AUC: **99.92%**
- Cross-generator (SDXL) AUC: **98.09%** (smallest drop: −1.91%)
- Outperforms CNNDetect (+14.72pp AUC) and UnivFD (+8.52pp AUC)
