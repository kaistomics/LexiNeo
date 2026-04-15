#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lexineobcr.model import LexiNeoBCRModel, tokenize_peptide, DEFAULT_CONFIG

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / 'lexineobcr' / 'models'
DATA_DIR = SCRIPT_DIR / 'lexineobcr' / 'data'

N_FOLDS = 5


class PeptideDataset(Dataset):
    def __init__(self, peptides):
        self.peptides = peptides

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        pep = self.peptides[idx]
        tokens = tokenize_peptide(pep)
        return {'pep_tokens': tokens, 'peptide': pep}


def collate_fn(batch):
    max_len = max(len(b['pep_tokens']) for b in batch)
    pep_tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        pep_tokens[i, :len(b['pep_tokens'])] = b['pep_tokens']
    return {'pep_tokens': pep_tokens, 'peptides': [b['peptide'] for b in batch]}


@torch.no_grad()
def predict_batch(model, loader, device):
    model.eval()
    all_preds, all_peptides = [], []

    for batch in tqdm(loader, desc='Predicting', leave=False):
        pep_idx = batch['pep_tokens'].to(device)
        pep_mask = (pep_idx != 0).float()
        probs = model.predict_proba(pep_idx, pep_mask)
        all_preds.extend(probs.cpu().numpy().flatten())
        all_peptides.extend(batch['peptides'])

    return np.array(all_preds), all_peptides


def load_peptides(input_path):
    input_path = Path(input_path)

    if input_path.suffix in ['.tsv', '.csv']:
        sep = '\t' if input_path.suffix == '.tsv' else ','
        df = pd.read_csv(input_path, sep=sep)
        for col in ['peptide', 'Peptide', 'epitope', 'Epitope', 'sequence', 'Sequence']:
            if col in df.columns:
                return df[col].tolist(), df
        raise ValueError(f"No peptide column found. Available: {list(df.columns)}")
    else:
        with open(input_path) as f:
            peptides = [line.strip() for line in f if line.strip()]
        return peptides, None


def main():
    parser = argparse.ArgumentParser(description='LexiNeoBCR: BCR epitope binding prediction')
    parser.add_argument('input', help='Input file (txt/tsv/csv)')
    parser.add_argument('output', help='Output TSV file')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"LexiNeoBCR | Device: {device}")

    ighv_path = DATA_DIR / 'ighv_archetypes.pt'
    cdr3_path = DATA_DIR / 'cdr3h_archetypes.pt'
    if not ighv_path.exists() or not cdr3_path.exists():
        sys.exit("Error: Archetype files not found in lexineobcr/data/")

    print(f"Loading peptides from {args.input}...")
    peptides, _ = load_peptides(args.input)
    print(f"Loaded {len(peptides)} peptides")

    valid_mask = [len(p.replace('O', '')) == 15 for p in peptides]
    valid_peptides = [p for p, v in zip(peptides, valid_mask) if v]
    skipped = len(peptides) - len(valid_peptides)
    if skipped:
        print(f"  Skipped {skipped} rows (not 15-mer)")
    print(f"  Valid: {len(valid_peptides)} peptides")

    if not valid_peptides:
        sys.exit("Error: No valid peptides found")

    dataset = PeptideDataset(valid_peptides)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_fn)

    ighv = torch.load(ighv_path, map_location='cpu', weights_only=False)['centroids']
    cdr3 = torch.load(cdr3_path, map_location='cpu', weights_only=False)['centroids']

    all_fold_preds = []

    for fold in range(N_FOLDS):
        ckpt_path = MODELS_DIR / f'fold{fold}.pt'
        if not ckpt_path.exists():
            print(f"Warning: Fold {fold} checkpoint not found, skipping")
            continue

        print(f"Fold {fold}...", end=' ', flush=True)
        model = LexiNeoBCRModel(DEFAULT_CONFIG, ighv, cdr3).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])

        preds, _ = predict_batch(model, loader, device)
        all_fold_preds.append(preds)
        print(f"[{preds.min():.3f}, {preds.max():.3f}]")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if not all_fold_preds:
        sys.exit("Error: No predictions generated")

    ensemble = np.mean(all_fold_preds, axis=0)

    result_df = pd.DataFrame({
        'peptide': valid_peptides,
        'score': np.round(ensemble, 4)
    })
    result_df.to_csv(args.output, sep='\t', index=False)

    print(f"\nSaved: {args.output} ({len(result_df)} rows)")
    print(f"  score [{ensemble.min():.4f}, {ensemble.max():.4f}] mean={ensemble.mean():.4f}")


if __name__ == '__main__':
    main()
