"""
CL4SRec: Contrastive Learning for Sequential Recommendation
Paper: "Contrastive Learning for Sequential Recommendation" (WWW 2022)

Key components:
1. Sequence augmentation: crop, mask, reorder
2. Contrastive loss (InfoNCE) between augmented views  
3. Combined loss: L = L_rec + λ * L_cl
"""

import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np

from config import config, ModelConfig


# =============================================================================
# SEQUENCE AUGMENTATION
# =============================================================================

class SequenceAugmentor:
    """
    Sequence augmentation strategies for contrastive learning
    
    Augmentation types:
    - crop: Random contiguous subsequence
    - mask: Random item masking
    - reorder: Random segment reordering
    """
    
    def __init__(
        self,
        crop_ratio: float = 0.6,
        mask_ratio: float = 0.2,
        reorder_ratio: float = 0.2,
        mask_token: int = 0  # Use padding token for mask
    ):
        self.crop_ratio = crop_ratio
        self.mask_ratio = mask_ratio
        self.reorder_ratio = reorder_ratio
        self.mask_token = mask_token
    
    def crop(self, seq: List[int]) -> List[int]:
        """Random crop - take contiguous subsequence"""
        if len(seq) <= 2:
            return seq.copy()
        
        crop_len = max(2, int(len(seq) * self.crop_ratio))
        start = random.randint(0, len(seq) - crop_len)
        return seq[start:start + crop_len]
    
    def mask(self, seq: List[int]) -> List[int]:
        """Random mask - replace some items with mask token"""
        if len(seq) <= 1:
            return seq.copy()
        
        masked_seq = seq.copy()
        num_mask = max(1, int(len(seq) * self.mask_ratio))
        mask_positions = random.sample(range(len(seq)), min(num_mask, len(seq) - 1))
        
        for pos in mask_positions:
            masked_seq[pos] = self.mask_token
        
        return masked_seq
    
    def reorder(self, seq: List[int]) -> List[int]:
        """Random reorder - shuffle a segment"""
        if len(seq) <= 2:
            return seq.copy()
        
        reordered = seq.copy()
        segment_len = max(2, int(len(seq) * self.reorder_ratio))
        start = random.randint(0, max(0, len(seq) - segment_len))
        
        segment = reordered[start:start + segment_len]
        random.shuffle(segment)
        reordered[start:start + segment_len] = segment
        
        return reordered
    
    def augment(self, seq: List[int], aug_type: str = None) -> List[int]:
        """Apply random or specified augmentation"""
        if aug_type is None:
            aug_type = random.choice(['crop', 'mask', 'reorder'])
        
        if aug_type == 'crop':
            return self.crop(seq)
        elif aug_type == 'mask':
            return self.mask(seq)
        elif aug_type == 'reorder':
            return self.reorder(seq)
        else:
            return seq.copy()
    
    def get_pair(self, seq: List[int]) -> Tuple[List[int], List[int]]:
        """Generate two augmented views of the same sequence"""
        # Use different augmentation strategies for diversity
        view1 = self.augment(seq, 'crop')
        view2 = self.augment(seq, random.choice(['mask', 'reorder']))
        return view1, view2


# =============================================================================
# TRANSFORMER COMPONENTS (Same as before)
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Initialize with sinusoidal
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.position_embedding.weight.data = pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.dropout(x + self.position_embedding(positions))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.size()
        
        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Causal mask
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), -1e9)
        
        # Padding mask
        if mask is not None:
            pad_mask = (mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# =============================================================================
# CL4SREC MODEL
# =============================================================================

class CL4SRecEncoder(nn.Module):
    """SASRec Encoder for CL4SRec"""
    
    def __init__(self, num_items: int, cfg: ModelConfig = None):
        super().__init__()
        self.cfg = cfg or config.model
        self.num_items = num_items
        
        self.item_embedding = nn.Embedding(
            num_items + 1, self.cfg.embedding_dim, padding_idx=0
        )
        self.pos_encoding = PositionalEncoding(
            self.cfg.embedding_dim, self.cfg.max_seq_length + 1, self.cfg.dropout
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.cfg.embedding_dim,
                self.cfg.num_attention_heads,
                self.cfg.hidden_dim,
                self.cfg.dropout
            ) for _ in range(self.cfg.num_transformer_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(self.cfg.embedding_dim)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                std = 0.01 if m.weight.size(0) > 10000 else 0.02
                nn.init.normal_(m.weight, 0, std)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])
    
    def forward(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq = (seq + 1).clamp(0, self.num_items)
        x = self.item_embedding(seq)
        x = self.pos_encoding(x)
        
        for block in self.blocks:
            x = block(x, mask)
        
        return self.final_norm(x)
    
    def get_last_hidden(self, seq: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.forward(seq, mask)
        
        if mask is not None:
            lengths = mask.sum(dim=1).long() - 1
            lengths = lengths.clamp(min=0, max=hidden.size(1) - 1)
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_idx, lengths]
        return hidden[:, -1, :]


class CL4SRecModel(nn.Module):
    """
    CL4SRec: Contrastive Learning for Sequential Recommendation
    
    Loss = L_rec (BPR) + λ * L_cl (InfoNCE)
    
    Paper: WWW 2022
    """
    
    def __init__(
        self,
        num_items: int,
        cfg: ModelConfig = None,
        cl_weight: float = 0.1,      # λ for CL loss
        temperature: float = 1.0,     # Temperature for InfoNCE
        num_negatives: int = 256      # Negatives for InfoNCE (in-batch)
    ):
        super().__init__()
        self.cfg = cfg or config.model
        self.num_items = num_items
        self.cl_weight = cl_weight
        self.temperature = temperature
        self.num_negatives = num_negatives
        
        # Encoder
        self.encoder = CL4SRecEncoder(num_items, cfg)
        
        # Projection head for contrastive learning
        self.cl_projector = nn.Sequential(
            nn.Linear(self.cfg.embedding_dim, self.cfg.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.embedding_dim, self.cfg.embedding_dim)
        )
        
        # Augmentor
        self.augmentor = SequenceAugmentor()
    
    def contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE Contrastive Loss
        
        Args:
            z1, z2: (batch_size, dim) - projected representations
        """
        batch_size = z1.size(0)
        
        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=z1.device)
        
        # InfoNCE loss (both directions)
        loss_12 = F.cross_entropy(sim_matrix, labels)
        loss_21 = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_12 + loss_21) / 2
    
    def forward(
        self,
        input_seq: torch.Tensor,
        attention_mask: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
        aug_seq1: Optional[torch.Tensor] = None,
        aug_mask1: Optional[torch.Tensor] = None,
        aug_seq2: Optional[torch.Tensor] = None,
        aug_mask2: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with BPR + Contrastive Learning loss
        """
        # Get sequence representation
        seq_repr = self.encoder.get_last_hidden(input_seq, attention_mask)
        
        # === BPR Loss ===
        pos_emb = self.encoder.item_embedding(positive_items + 1)
        neg_emb = self.encoder.item_embedding(negative_items + 1)
        
        pos_scores = (seq_repr * pos_emb).sum(dim=-1)
        neg_scores = (seq_repr.unsqueeze(1) * neg_emb).sum(dim=-1)
        
        diff = (pos_scores.unsqueeze(1) - neg_scores).clamp(-80, 80)
        bpr_loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        
        result = {
            'bpr_loss': bpr_loss,
            'pos_scores': pos_scores,
            'neg_scores': neg_scores
        }
        
        # === Contrastive Loss ===
        if aug_seq1 is not None and aug_seq2 is not None:
            # Get representations for augmented views
            z1 = self.encoder.get_last_hidden(aug_seq1, aug_mask1)
            z2 = self.encoder.get_last_hidden(aug_seq2, aug_mask2)
            
            # Project through CL head
            z1 = self.cl_projector(z1)
            z2 = self.cl_projector(z2)
            
            cl_loss = self.contrastive_loss(z1, z2)
            result['cl_loss'] = cl_loss
            
            # Combined loss
            result['loss'] = bpr_loss + self.cl_weight * cl_loss
        else:
            result['loss'] = bpr_loss
            result['cl_loss'] = torch.tensor(0.0, device=bpr_loss.device)
        
        return result
    
    def predict(
        self,
        input_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict top-k items"""
        self.eval()
        with torch.no_grad():
            seq_repr = self.encoder.get_last_hidden(input_seq, attention_mask)
            item_emb = self.encoder.item_embedding.weight[1:]
            scores = torch.matmul(seq_repr, item_emb.T)
            return torch.topk(scores, k=min(top_k, self.num_items), dim=-1)


if __name__ == "__main__":
    print("Testing CL4SRec...")
    
    model = CL4SRecModel(num_items=1000)
    
    B, L = 4, 10
    seq = torch.randint(0, 1000, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)
    pos = torch.randint(0, 1000, (B,))
    neg = torch.randint(0, 1000, (B, 4))
    
    # Without CL
    out = model(seq, mask, pos, neg)
    print(f"BPR Loss: {out['bpr_loss'].item():.4f}")
    
    # With CL
    out = model(seq, mask, pos, neg, seq, mask, seq, mask)
    print(f"Total Loss: {out['loss'].item():.4f}")
    print(f"CL Loss: {out['cl_loss'].item():.4f}")
    
    # Predict
    scores, indices = model.predict(seq, mask, top_k=5)
    print(f"Predictions: {indices.shape}")
    print("CL4SRec test passed!")
