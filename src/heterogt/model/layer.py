import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv

# multi-class classification task
class MultiPredictionHead(nn.Module):
    def __init__(self, hidden_size, label_size):
        super(MultiPredictionHead, self).__init__()
        self.cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), 
                nn.ReLU(), 
                nn.Linear(hidden_size, label_size)
            )

    def forward(self, input):
        return self.cls(input)
    
class BinaryPredictionHead(nn.Module):
    def __init__(self, hidden_size):
        super(BinaryPredictionHead, self).__init__()
        self.cls = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), 
                nn.ReLU(), 
                nn.Linear(hidden_size, 1)
            )
    def forward(self, input):
        return self.cls(input)
    
class HierTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, batch_first=True, norm_first=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.transformer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=batch_first, norm_first=norm_first)

    def forward(self, x, src_key_padding_mask, attn_mask):
        """
        src:                [B, L, d] (batch_first=True)
        attn_masks:         [B*num_heads, L, L]
        src_key_padding_mask: [B, L]; True=PAD
        """
        B, L, _ = x.shape
        H = self.num_heads
        out = self.transformer(src=x, src_key_padding_mask=src_key_padding_mask, src_mask=attn_mask)
        rows_use = self._rows_from_attn_mask(attn_mask, src_key_padding_mask, B, H)
        x = self._blend_update(x, out, rows_use)
        return x

    @staticmethod
    def _blend_update(x_old: torch.Tensor, x_new: torch.Tensor, update_rows: torch.BoolTensor):
        """
        Use x_new to overwrite only the rows where update_rows=True; 
        for all other rows, keep x_old.
        x_old, x_new: [B, L, d]
        update_rows: [B, L] (True = update)
        """
        mask = update_rows.unsqueeze(-1)              # [B, L, 1]
        return torch.where(mask, x_new, x_old)

    @staticmethod
    def _rows_from_attn_mask(attn_mask: torch.Tensor, src_key_padding_mask: torch.Tensor, B: int, H: int):
        """
        Infer the "rows allowed as Query" from the [B*H, L, L] attn_mask 
        (i.e., a row has at least one non-self unmasked column and is not a padding token).  
        Returns [B, L] as a boolean mask: True means the row will be updated in this forward pass.

        Conventions:
            attn_mask == True  → blocked
            attn_mask == False → allowed
            src_key_padding_mask == True → padding token

        Args:
            attn_mask: [B*H, L, L], torch.bool, attention mask
            src_key_padding_mask: [B, L], torch.bool, padding mask (True = padding)
            B: int, batch size
            H: int, number of heads

        Returns:
            row_updatable: [B, L], torch.bool, True indicates the token should be updated
        """
        BH, L, _ = attn_mask.shape
        assert BH == B * H, f"attn_mask first dimension must be B*num_heads, got {BH} vs {B}*{H}"
        assert src_key_padding_mask.shape == (B, L), \
            f"src_key_padding_mask shape must be [B, L], got {src_key_padding_mask.shape}"
        assert src_key_padding_mask.dtype == torch.bool, "src_key_padding_mask must be torch.bool type"
        assert src_key_padding_mask.device == attn_mask.device, \
            f"Device mismatch: src_key_padding_mask on {src_key_padding_mask.device}, attn_mask on {attn_mask.device}"

        m = attn_mask.view(B, H, L, L)  # [B, H, L, L]
        
        # Create diagonal mask to ignore self-attention positions
        diag_mask = torch.eye(L, dtype=torch.bool, device=attn_mask.device)  # [L, L]
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(0).expand(B, H, L, L)  # [B, H, L, L]
        
        # Set diagonal positions to True (blocked), so only non-diagonal False entries are considered
        m_non_diag = m | diag_mask  # [B, H, L, L]
        
        # Apply padding mask: mask out all columns corresponding to padding tokens
        padding_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, L, 1]
        m_non_diag = m_non_diag | padding_mask  # [B, H, L, L]

        # Check if each row (Query) is fully blocked (ignoring self and padding tokens)
        row_all_banned = m_non_diag.all(dim=-1)  # [B, H, L]
        
        # A row is updatable if it has at least one unmasked non-self column in any head
        row_updatable = (~row_all_banned).any(dim=1)  # [B, L]
        
        # Exclude padding tokens: they should not be updated
        row_updatable = row_updatable & (~src_key_padding_mask)  # [B, L]

        # Debug print
        print(f"[DEBUG] row_updatable shape: {row_updatable.shape}, dtype: {row_updatable.dtype}")

        return row_updatable
    
class DiseaseOccHetGNN(nn.Module):
    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d = d_model
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.ln_v1 = nn.LayerNorm(d_model)
        self.ln_o1 = nn.LayerNorm(d_model)
        self.ln_v2 = nn.LayerNorm(d_model)
        self.ln_o2 = nn.LayerNorm(d_model)

        self.alpha_v1 = nn.Parameter(torch.tensor(0.1))
        self.alpha_o1 = nn.Parameter(torch.tensor(0.1))
        self.alpha_v2 = nn.Parameter(torch.tensor(0.1))
        self.alpha_o2 = nn.Parameter(torch.tensor(0.1))

        self.conv1 = HeteroConv({
            ('visit','contains','occ'):   GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=False),
            ('occ','contained_by','visit'): GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=False),
            ('visit','next','visit'):     GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=True),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('visit','contains','occ'):   GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=False),
            ('occ','contained_by','visit'): GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=False),
            ('visit','next','visit'):     GATConv(d_model, d_model, heads=heads, concat=False, add_self_loops=True),
        }, aggr='sum')

        self.lin_v = nn.Linear(d_model, d_model)
        self.lin_o = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.lin_v.weight); nn.init.zeros_(self.lin_v.bias)
        nn.init.zeros_(self.lin_o.weight); nn.init.zeros_(self.lin_o.bias)

    def forward(self, hg):
        x_v = hg['visit'].x
        x_o = hg['occ'].x

        h1 = self.conv1({'visit': x_v, 'occ': x_o}, hg.edge_index_dict)
        dv = self.drop(h1['visit'])
        do = self.drop(h1['occ'])
        v1 = self.ln_v1(x_v + self.alpha_v1 * dv)
        o1 = self.ln_o1(x_o + self.alpha_o1 * do)

        h2 = self.conv2({'visit': v1, 'occ': o1}, hg.edge_index_dict)
        dv2 = self.drop(h2['visit'])
        do2 = self.drop(h2['occ'])
        v2 = self.ln_v2(v1 + self.alpha_v2 * dv2)
        o2 = self.ln_o2(o1 + self.alpha_o2 * do2)

        v_out = v2 + self.lin_v(v2)
        o_out = o2 + self.lin_o(o2)

        return {'visit': v_out, 'occ': o_out}
    
class CLSQueryMHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_token_type: list, dropout: float = 0.0, 
                 use_raw_value_agg: bool = True, fallback_to_cls: bool = True):
        super().__init__()
        self.d_model = d_model
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.use_raw_value_agg = use_raw_value_agg
        self.fallback_to_cls = fallback_to_cls
        self.drop = nn.Dropout(dropout)
        self.attn_token_types = attn_token_type

    def forward(self, x: torch.Tensor, token_type: torch.Tensor):
        """
        Args:
            x: [B, L, D]
            token_type: [B, L] (long/int)

        Returns:
            out: [B, 2D] = concat([CLS], agg)
            attn_probs: [B, H, 1, L]  (for debugging; per-head attention)
        """
        B, L, D = x.shape
        assert token_type.shape == (B, L)
        assert D == self.d_model

        # 1) CLS as the single query
        cls = x[:, 0, :]                 # [B, D]
        q   = cls.unsqueeze(1)           # [B, 1, D]
        k   = x                          # [B, L, D]
        v   = x                          # [B, L, D]

        # 2) Build key_padding_mask: True = ignore (mask out)
        allowed = torch.zeros_like(token_type, dtype=torch.bool)
        for t in self.attn_token_types:
            allowed |= (token_type == t)
        kv_mask = ~allowed   # mask out positions not in {attn_token_types}
        # Prevent entire row being True (no {6,7}) which causes softmax NaN: temporarily unmask CLS
        no_kv = kv_mask.all(dim=1)       # [B]
        if no_kv.any():
            kv_mask = kv_mask.clone()
            kv_mask[no_kv, 0] = False    # avoid NaN; aggregation result will be overwritten later

        # 3) Multi-head attention (retain weights; do not average across heads)
        attn_out, attn_probs = self.mha(
            q, k, v,
            key_padding_mask=kv_mask,           # [B, L]; True = ignore
            need_weights=True,
            average_attn_weights=False          # -> [B, H, 1, L]
        )  # attn_out: [B, 1, D]

        # 4) Aggregate vector from attention
        if self.use_raw_value_agg:
            # Compute weighted average explicitly in the "input space"
            w = attn_probs.mean(dim=1)          # [B, 1, L], averaged across heads
            # Zero out masked positions, renormalize within the subset (avoid leakage into non-{6,7})
            w = w.masked_fill(kv_mask.unsqueeze(1), 0.0)  # [B, 1, L]
            denom = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [B, 1, 1]
            w = w / denom
            agg = torch.bmm(w.reshape(B, 1, L), x).squeeze(1)      # [B, D]
        else:
            # Directly use MHA output (already in value-projection + out_proj space)
            agg = attn_out.squeeze(1)   # [B, D]

        # 5) Fallback strategy for samples with no {6,7}
        if no_kv.any():
            if self.fallback_to_cls:
                agg = agg.clone()
                agg[no_kv] = cls[no_kv]    # fallback to CLS
            else:
                agg = agg.clone()
                agg[no_kv] = 0.0           # fallback to zero vector
        assert agg.shape == (B, D), "CLS output shape mismatch"
        return agg  # [B, D]