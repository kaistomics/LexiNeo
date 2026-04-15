#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lexineotcr.model import LexiNeoTCRModel, encode_seq

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / 'lexineotcr' / 'models'
PSEUDOSEQ_FILE = SCRIPT_DIR / 'lexineotcr' / 'data' / 'pseudosequence.dat'

CONFIG = {
    'num_protos': 256,
    'embed_dim': 256,
    'num_heads': 4,
    'pep_len': 15,
    'mhc_len': 34,
    'bn_dim': 128,
    'topk_protos': 32,
    'n_queries': 4,
    'n_refinement': 2,
}

N_FOLDS = 5
VALID_AA = set('ACDEFGHIKLMNPQRSTVWYX')


def normalize_allele(allele: str) -> str:
    a = allele.strip()
    if re.match(r'HLA-DQB1[\*\-]', a):
        suffix = re.sub(r'HLA-DQB1[\*\-]', '', a).replace(':', '').replace('-', '')
        return f'HLA-DQA10101-DQB1{suffix}'
    if re.match(r'HLA-DPB1[\*\-]', a):
        suffix = re.sub(r'HLA-DPB1[\*\-]', '', a).replace(':', '').replace('-', '')
        return f'HLA-DPA10103-DPB1{suffix}'
    if a.startswith('HLA-'):
        a = a[4:]
    return re.sub(r'[\*\-]', '_', a).replace(':', '')


def load_pseudoseqs(path: Path) -> dict:
    d = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                d[parts[0]] = parts[1]
    return d


def get_pseudoseq(allele: str, pseudoseqs: dict, mhc_len: int = 34) -> str:
    key = normalize_allele(allele)
    seq = pseudoseqs.get(key) or next(
        (v for k, v in pseudoseqs.items() if k.startswith(key[:7])), None
    )
    return (seq[:mhc_len].ljust(mhc_len, 'X') if seq else 'X' * mhc_len)


def load_fold(fold: int, device: torch.device) -> LexiNeoTCRModel:
    model_path = MODELS_DIR / f'fold{fold}_best.pt'
    proto_path = MODELS_DIR / 'prototypes.pt'
    for p in (model_path, proto_path):
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

    proto_raw = torch.load(proto_path, map_location='cpu', weights_only=False)
    proto_centroids = proto_raw['centroids'] if isinstance(proto_raw, dict) else proto_raw

    model = LexiNeoTCRModel(CONFIG, proto_centroids).to(device)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model: LexiNeoTCRModel, df: pd.DataFrame,
                  pseudoseqs: dict, device: torch.device) -> np.ndarray:
    pep_len = CONFIG['pep_len']
    mhc_len = CONFIG['mhc_len']
    pseudo_cache = {a: get_pseudoseq(a, pseudoseqs, mhc_len)
                    for a in df['mhc_allele'].unique()}
    scores = []
    batch_size = 256
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        pep_tok = torch.tensor(
            [encode_seq(e, pep_len) for e in batch['epitope']],
            dtype=torch.long, device=device)
        mhc_tok = torch.tensor(
            [encode_seq(pseudo_cache.get(a, 'X' * mhc_len), mhc_len)
             for a in batch['mhc_allele']],
            dtype=torch.long, device=device)
        scores.append(model(pep_tok, mhc_tok)['prob'].cpu().numpy())
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser(description='LexiNeoTCR: MHC-II T-cell epitope immunogenicity prediction')
    parser.add_argument('input', help='Input TSV (columns: epitope, mhc_allele)')
    parser.add_argument('output', help='Output TSV path')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"LexiNeoTCR | Device: {device}")

    df = pd.read_csv(args.input, sep='\t')
    missing = {'epitope', 'mhc_allele'} - set(df.columns)
    if missing:
        sys.exit(f"Error: Missing columns: {missing}")
    print(f"Input: {len(df)} rows")

    mask = (
        (df['epitope'].str.len() == CONFIG['pep_len']) &
        df['epitope'].apply(lambda x: all(c in VALID_AA for c in str(x)))
    )
    df_valid = df[mask].copy().reset_index(drop=True)
    skipped = len(df) - len(df_valid)
    if skipped:
        print(f"  Skipped {skipped} rows (not 15-mer or invalid AA)")
    print(f"  Valid: {len(df_valid)} samples")
    if len(df_valid) == 0:
        sys.exit("Error: No valid samples.")

    pseudoseqs = load_pseudoseqs(PSEUDOSEQ_FILE)
    print(f"Pseudosequences: {len(pseudoseqs)} alleles")

    fold_scores = []
    for fold in range(N_FOLDS):
        print(f"Fold {fold}...", end=' ', flush=True)
        model = load_fold(fold, device)
        s = run_inference(model, df_valid, pseudoseqs, device)
        fold_scores.append(s)
        print(f"[{s.min():.3f}, {s.max():.3f}]")
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    ensemble = np.mean(fold_scores, axis=0)
    out_df = df_valid[['epitope', 'mhc_allele']].copy()
    out_df['score'] = np.round(ensemble.astype(float), 4)
    out_df.to_csv(args.output, sep='\t', index=False)

    print(f"\nSaved: {args.output} ({len(out_df)} rows)")
    print(f"  score [{ensemble.min():.4f}, {ensemble.max():.4f}] mean={ensemble.mean():.4f}")


if __name__ == '__main__':
    main()
