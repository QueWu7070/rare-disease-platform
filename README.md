# rare-disease-platform
Reference implementation for "Designing a Secure Rare Disease Data-Sharing Platform Leveraging Electronic Health Records: A System Architecture and Simulation Study". Features NLP data governance, differential privacy, and Hyperledger Fabric.
## Repository Structure

| File | Function |
|---|---|
| `governance/mice_imputation.py` | MICE imputation for missing clinical values |
| `governance/train_rare_roberta.py` | Fine-tune Rare-RoBERTa for clinical NER |
| `governance/ner_inference.py` | Run NER inference on clinical notes |
| `governance/terminology_mapping.py` | Hybrid edit-distance + cosine mapping to ORPHA/OMIM/HPO |
| `privacy/deidentification.py` | k-anonymity (k=5) + differential privacy de-identification |
| `blockchain/chaincode/data_sharing_cc.go` | Hyperledger Fabric access-control chaincode |
| `blockchain/caliper/` | Caliper v0.4.2 stress-test configuration |

## Environment

- Ubuntu 20.04 LTS · Python 3.8
- NVIDIA Tesla V100 (32GB VRAM) · CUDA 11.0
- Hyperledger Fabric v2.2· CouchDB v3.1· Caliper v0.4.2

## Installation

```bash
python -m venv venv && source venv/bin/activate

# GPU build (V100 / CUDA 11.0) — must be installed before requirements.txt
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

## Running the pipeline

### Option 1 — Smoke test (bundled synthetic sample, no external data)

Verifies the entire pipeline executes without error. Does **not** reproduce
the reported F1 = 87.3%, which requires the full training corpus.

```bash
python governance/mice_imputation.py
python governance/train_rare_roberta.py --epochs 1
python governance/ner_inference.py --text "患儿反复跌倒，近端肌无力"
python governance/terminology_mapping.py --term "杜氏肌营养不良"
python privacy/deidentification.py
```

### Option 2 — Full reproduction (paper metrics)

1. Download the **CBLUE** benchmark (15,000 clinical sentences):https://github.com/CBLUEbenchmark/CBLUE
   Convert to BIO CoNLL format and place at `data/cblue_train.conll`.

2. Download the 200-case synthetic test set:
   https://github.com/QueWu7070/Synthetic-Rare-Disease-Dataset-Sample
   Convert to BIO CoNLL format and place at `data/synthetic_200.conll`.

```bash
python governance/train_rare_roberta.py \
  --train data/cblue_train.conll \
  --evaldata/synthetic_200.conll \
  --epochs 10 --batch_size 32 --lr 2e-5 --max_len 512
```

## Data dictionary (`data/synthetic_sample.csv`)

| Column | Type | Description |
|---|---|---|
| `patient_id` | string | Synthetic patient identifier (no real PII) |
| `age_months` | float | Age in months; may be missing (NaN) |
| `location` | string | Address string; generalised to city level during de-identification |
| `diagnosis_date` | string | ISO date (YYYY-MM-DD); suppressed to YYYY-MM during de-identification |
| `disease_name_raw` | string | Non-standardised disease label to be mapped to ORPHA/OMIM |
| `lab_value` | float | Example numeric lab result; may be missing |
| `clinical_note` | string | Free-text Chinese clinical narrative; input to NER |

`data/ner_sample.conll` uses character-level BIO tagging
(`B-DIS`, `I-DIS`, `B-SYM`, `I-SYM`, `B-DRUG`, `I-DRUG`, `O`),
UTF-8 encoding, one token per line, blank lines between sentences.

## License

The code in this repository is licensed under the MIT License. Note that the third-party datasets (e.g., CBLUE) used in this study are subject to their own respective licenses and terms of use.
