# Amazon ML Challenge — Business Entity Resolution

High-performance solution for the Amazon ML Challenge: large-scale multi-source business entity resolution and entity linking, optimizing macro-averaged $F_{0.5}$.

## Overview

In real-world e-commerce catalogs and business directories, the same commercial entity often appears across multiple sources with distinct abbreviations, missing address components, and multilingual naming conventions. Merging two different entities (false positive) is twice as damaging as missing a link (false negative), which is why this solution is engineered to optimize macro-averaged $F_{0.5}$.

### Key Highlights
- **8-Channel Inverted Multi-View Blocking**: Achieves **98.66% candidate recall ceiling** over 10 million candidate records (Prefix-3, Exact Sorted Name, Double Metaphone, Postal Codes, Rare Name Tokens, Addr Num + First Word, Composite Landmark Indexing).
- **48-Dimensional Pairwise Feature Engineering**: Captures token-set ratios, RapidFuzz token sorting, Levenshtein distances, phonetic match flags, address house-number concordance, and postal conflict flags.
- **LightGBM GBDT with Stratified Hard-Negative Mining**: High-capacity tree ensemble trained on hard negative pairs.
- **Calibrated Tiered Rules & Contradiction Veto Filter**: Custom decision rules with relative margin pruning ($\Delta_p \le 0.15$), high-concordance boost ($\tau = 0.85$), and missing address protection ($\tau = 0.99$).

---

## Directory Structure

```text
├── code/
│   └── business_entity_resolution/
│       ├── cache/
│       │   ├── exp03_model.txt               # Trained 48-feature champion GBDT model
│       │   └── baseline_model.txt            # Baseline LightGBM model
│       ├── src/
│       │   ├── core.py                       # Data loading, normalization, macro F0.5 evaluator
│       │   ├── features.py                   # 48-dimensional pairwise feature extractors
│       │   ├── pipeline.py                   # End-to-end full training & inference pipeline
│       │   ├── exp01_fast_baseline.py        # Fast dev baseline experiment
│       │   ├── exp02_blocking_upgrade.py     # Blocking channel evaluation
│       │   ├── exp03_pipeline.py             # Feature extraction & GBDT training
│       │   ├── exp04_rule_calibration.py     # Tiered decision threshold calibration
│       │   ├── run_inference.py              # Test set prediction & submission generator
│       │   └── audit_metric.py               # Macro F0.5 unit test & verification suite
│       ├── experiments.csv                   # Experiment tracking log
│       ├── requirements.txt                  # Python dependencies
│       └── README.md                         # Detailed technical documentation
├── 6ab10eb3b23ba_student_resource/
│   └── student_resource/
│       ├── utils/validate_submission.py      # Official submission validator
│       ├── Documentation_template.md         # Solution methodology report
│       └── README.md                         # Official challenge rules & guidelines
└── README.md
```

---

## Quick Start

### 1. Installation
```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

### 2. Verify Evaluator
```bash
python src/audit_metric.py
```

### 3. Run Fast Baseline Experiment
```bash
python src/exp01_fast_baseline.py
```

### 4. Run Full Production Inference
```bash
python src/run_inference.py
```

### 5. Validate Output Format
```bash
python ../../6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../6ab10eb3b23ba_student_resource/student_resource/dataset/test
```