import torch
import torch.nn as nn
import torch.nn.functional as F

AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')
VOCAB = {aa: i + 1 for i, aa in enumerate(AA_LIST)}
VOCAB['<PAD>'] = 0
VOCAB['<UNK>'] = len(VOCAB)

DEFAULT_CONFIG = {
    'embed_dim': 256,
    'hidden_dim': 384,
    'num_heads': 2,
    'dropout': 0.1,
    'n_ighv_layers': 1,
    'n_cdr3_layers': 2,
}


class ChannelWiseGatedFusion(nn.Module):
    def __init__(self, hidden_dim, reduction=4, dropout=0.1):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // reduction),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // reduction, hidden_dim),
            nn.Sigmoid()
        )
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, raw_emb, branch_emb):
        concat = torch.cat([raw_emb, branch_emb], dim=-1)
        gate = self.gate_net(concat)
        fused = gate * raw_emb + (1 - gate) * branch_emb
        return self.norm(fused * self.scale + self.bias)


class CNNPooler(nn.Module):
    def __init__(self, in_dim, out_dim, kernels=[3, 5, 7], dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_dim, out_dim, k, padding=k // 2),
                nn.BatchNorm1d(out_dim),
                nn.GELU()
            )
            for k in kernels
        ])
        self.proj = nn.Linear(out_dim * len(kernels), out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x.transpose(1, 2)
        conv_outs = [F.adaptive_max_pool1d(conv(x), 1).squeeze(-1) for conv in self.convs]
        return self.norm(self.drop(self.proj(torch.cat(conv_outs, dim=-1))))


class MidBranch(nn.Module):
    def __init__(self, cfg, archetypes, n_layers, dropout=0.1):
        super().__init__()
        hd = cfg['hidden_dim']
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hd, cfg['num_heads'], dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(hd) for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hd, hd * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hd * 4, hd),
                nn.Dropout(dropout)
            )
            for _ in range(n_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hd) for _ in range(n_layers)])

        arch_dim = archetypes.shape[1]
        self.arch_proj = nn.Linear(arch_dim, hd) if arch_dim != hd else nn.Identity()
        self.register_buffer('archetypes', archetypes)

        self.cnn = CNNPooler(hd, hd, dropout=dropout)
        self.gated_fusion = ChannelWiseGatedFusion(hd, dropout=dropout)
        self.raw_pool = nn.Sequential(nn.Linear(hd, hd), nn.GELU(), nn.LayerNorm(hd))

    def forward(self, pep_emb, pep_mask):
        B = pep_emb.size(0)
        raw_emb = self.raw_pool(pep_emb.mean(dim=1))

        arch_ctx = self.arch_proj(self.archetypes)
        arch_kv = arch_ctx.unsqueeze(0).expand(B, -1, -1)

        x = pep_emb
        for attn, norm_a, ffn, norm_f in zip(
            self.cross_attn_layers, self.cross_norms, self.ffn_layers, self.ffn_norms
        ):
            attn_out, _ = attn(x, arch_kv, arch_kv)
            x = norm_a(x + attn_out)
            x = norm_f(x + ffn(x))

        x = x * pep_mask.unsqueeze(-1)
        branch_emb = self.cnn(x, pep_mask)
        return branch_emb, self.gated_fusion(raw_emb, branch_emb)


class LexiNeoBCRModel(nn.Module):
    def __init__(self, cfg, ighv_archetypes, cdr3h_archetypes):
        super().__init__()
        hd = cfg['hidden_dim']

        self.pep_embed = nn.Embedding(len(VOCAB), cfg['embed_dim'], padding_idx=0)
        self.pep_proj = nn.Linear(cfg['embed_dim'], hd)
        self.pep_norm = nn.LayerNorm(hd)

        self.ighv_branch = MidBranch(cfg, ighv_archetypes, cfg['n_ighv_layers'], cfg['dropout'])
        self.cdr3_branch = MidBranch(cfg, cdr3h_archetypes, cfg['n_cdr3_layers'], cfg['dropout'])

        self.fusion = nn.Sequential(
            nn.Linear(hd * 2, hd),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.LayerNorm(hd)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hd, hd),
            nn.GELU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(hd, 1)
        )
        self.raw_pool = CNNPooler(hd, hd, dropout=cfg['dropout'])

    def forward(self, pep_idx, pep_mask):
        pep_emb = self.pep_norm(self.pep_proj(self.pep_embed(pep_idx)))
        _, ighv_gated = self.ighv_branch(pep_emb, pep_mask)
        _, cdr3_gated = self.cdr3_branch(pep_emb, pep_mask)
        final_emb = self.fusion(torch.cat([ighv_gated, cdr3_gated], dim=-1))
        logits = self.classifier(final_emb)
        return logits

    def predict_proba(self, pep_idx, pep_mask):
        logits = self.forward(pep_idx, pep_mask)
        return torch.sigmoid(logits)


def tokenize_peptide(peptide):
    peptide = peptide.replace('O', '').upper()
    tokens = [VOCAB.get(aa, VOCAB['<UNK>']) for aa in peptide]
    return torch.tensor(tokens, dtype=torch.long)
