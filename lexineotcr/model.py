import math
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 256

AA_VOCAB = {aa: i + 1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
AA_VOCAB['X'] = 0


def encode_seq(seq, max_len):
    seq = seq[:max_len].ljust(max_len, 'X')
    return [AA_VOCAB.get(aa, 0) for aa in seq]


class PeptideEncoder(nn.Module):
    def __init__(self, embed_dim=256, vocab_size=22):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, tokens):
        return self.embed(tokens)


class MHCEncoder(nn.Module):
    def __init__(self, embed_dim=256, mhc_len=34, vocab_size=22):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(mhc_len, embed_dim)
        self.mhc_len = mhc_len

    def forward(self, tokens):
        B = tokens.size(0)
        pos = torch.arange(self.mhc_len, device=tokens.device).unsqueeze(0).expand(B, -1)
        return self.embed(tokens) + self.pos_embed(pos)


class LexiNeoTCRModel(nn.Module):
    def __init__(self, config, proto_centroids):
        super().__init__()
        K = config['num_protos']
        ed = config['embed_dim']
        nh = config['num_heads']

        self._pep_len = config['pep_len']
        self._mhc_len = config['mhc_len']
        self._topk_protos = config.get('topk_protos', 32)
        self._n_queries = config.get('n_queries', 4)
        self._n_refinement = config.get('n_refinement', 2)

        self.pep_encoder = PeptideEncoder(embed_dim=ed)
        self.mhc_encoder = MHCEncoder(embed_dim=ed, mhc_len=config['mhc_len'])

        BN = config.get('bn_dim', 128)
        self.bn_pep = nn.Sequential(nn.Linear(ed, BN), nn.LayerNorm(BN))
        self.bn_mhc = nn.Sequential(nn.Linear(ed, BN), nn.LayerNorm(BN))

        cnn_ch = 128
        self.cnn_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(ed, cnn_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(cnn_ch),
                nn.ReLU()
            ) for k in [5, 7, 9]
        ])
        self.cnn_proj = nn.Linear(cnn_ch * 3, HIDDEN_SIZE)
        self.cnn_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.cnn_drop = nn.Dropout(0.2)
        self.pep_len_emb = nn.Embedding(config['pep_len'] + 1, HIDDEN_SIZE)

        self.prototypes = nn.Parameter(proto_centroids.clone())
        self.query_embed = nn.Embedding(self._n_queries, HIDDEN_SIZE)
        self.proto_scale = nn.Parameter(torch.tensor(0.5))

        self.refine_cross = nn.ModuleList([
            nn.MultiheadAttention(HIDDEN_SIZE, nh, dropout=0.1, batch_first=True)
            for _ in range(self._n_refinement)
        ])
        self.refine_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE), nn.Sigmoid())
            for _ in range(self._n_refinement)
        ])

        self.query_pool = nn.Linear(self._n_queries * HIDDEN_SIZE, HIDDEN_SIZE)
        self.scorer = nn.Sequential(nn.LayerNorm(HIDDEN_SIZE), nn.Linear(HIDDEN_SIZE, 1))

    def encode_pmhc(self, pep_tokens, mhc_tokens):
        pep_mask = (pep_tokens != 0).float().unsqueeze(-1)
        pep_emb = self.pep_encoder(pep_tokens) * pep_mask
        mhc_emb = self.mhc_encoder(mhc_tokens)

        pep_bn = self.bn_pep(pep_emb)
        mhc_bn = self.bn_mhc(mhc_emb)
        compat = torch.bmm(pep_bn, mhc_bn.transpose(1, 2)) / math.sqrt(pep_bn.size(-1))

        pep_gate = torch.sigmoid(compat.max(dim=-1).values).unsqueeze(-1)
        align = F.softmax(compat, dim=-1)
        mhc_aln = torch.bmm(align, mhc_emb)
        pMHC = pep_emb * mhc_aln * pep_gate

        x = pMHC.transpose(1, 2)
        feats = [F.adaptive_max_pool1d(b(x), 1).squeeze(-1) for b in self.cnn_branches]
        antigen_emb = self.cnn_norm(self.cnn_drop(self.cnn_proj(torch.cat(feats, 1))))

        pep_lens = (pep_tokens != 0).sum(-1).clamp(max=self._pep_len)
        return antigen_emb + self.pep_len_emb(pep_lens)

    def forward(self, pep_tokens, mhc_tokens):
        B = pep_tokens.size(0)
        device = pep_tokens.device

        antigen_emb = self.encode_pmhc(pep_tokens, mhc_tokens)

        proto_norm = F.normalize(self.prototypes, dim=-1)

        queries = self.query_embed(torch.arange(self._n_queries, device=device))
        queries = queries.unsqueeze(0).expand(B, -1, -1) + antigen_emb.unsqueeze(1)
        context = queries

        for attn, gate_fn in zip(self.refine_cross, self.refine_gates):
            ctx_norm = F.normalize(context.mean(dim=1), dim=-1)
            topk_idx = torch.mm(ctx_norm, proto_norm.T).topk(self._topk_protos, dim=-1).indices
            selected_protos = self.prototypes[topk_idx]
            attn_out, _ = attn(query=context, key=selected_protos, value=selected_protos)
            context = context + gate_fn(torch.cat([context, attn_out], dim=-1)) * attn_out

        combined = antigen_emb + self.proto_scale * self.query_pool(context.view(B, -1))
        prob = torch.sigmoid(self.scorer(combined).squeeze(-1))

        return {'prob': prob}

    def predict_proba(self, pep_tokens, mhc_tokens):
        return self.forward(pep_tokens, mhc_tokens)['prob']
