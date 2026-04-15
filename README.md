# LexiNeo

Deep learning models for predicting immune epitope binding:

- **LexiNeoTCR**: Repertoire-aware MHC-II epitope immunogenicity prediction
- **LexiNeoBCR**: Repertoire-aware B-cell epitope binding prediction

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### LexiNeoTCR

```bash
python predict_lexineotcr.py input.tsv output.tsv
python predict_lexineotcr.py input.tsv output.tsv --cpu
```

Input TSV columns: `epitope`, `mhc_allele`

| epitope | mhc_allele |
|---------|------------|
| KLKSEYMTSWFYNEL | HLA-DRB1*01:01 |
| VFITLPCRIKIIML | HLA-DRB1*04:01 |

Supported allele formats: `HLA-DRB1*01:01`, `DRB1*01:01`, `HLA-DRB1-0101`, `HLA-DQB1*02:01`, `HLA-DPB1*01:01`

### LexiNeoBCR

```bash
python predict_lexineobcr.py input.txt output.tsv
python predict_lexineobcr.py input.txt output.tsv --cpu
```

Input: Plain text (one peptide per line) or TSV/CSV with peptide column.

## Output

| peptide | score |
|---------|-------|
| KLKSEYMTSWFYNEL | 0.8234 |
| VFITLPCRIKIIML | 0.1456 |

Score: 0.0 (non-binder) to 1.0 (binder), averaged across 5-fold ensemble.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- pandas, numpy, tqdm
