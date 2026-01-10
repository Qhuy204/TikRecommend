"""
SASRec + PhoBERT Fusion Model Architecture

Components:
1. SASRec: Self-Attentive Sequential Recommendation (Transformer-based)
2. PhoBERT Encoder: Vietnamese text understanding (frozen)
3. Fusion: Combines sequential + content signals
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import SASRecConfig, PhoBERTConfig, FusionConfig, default_config


# =============================================================================
# SASREC COMPONENTS
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking for SASRec"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len) - 1 for valid, 0 for padding
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Causal mask (prevent attending to future items)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, -1e9)  # Use large negative instead of -inf
        
        # Padding mask
        if attention_mask is not None:
            # Expand mask: (batch, seq) -> (batch, 1, 1, seq)
            padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask, -1e9)  # Use large negative instead of -inf
        
        # Softmax and dropout (add small epsilon for stability)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # Replace any NaN with 0
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(context)


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention + FFN"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), attention_mask)
        x = x + self.dropout(attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class SASRecEncoder(nn.Module):
    """
    SASRec: Self-Attentive Sequential Recommendation
    Uses transformer self-attention to model user behavior sequences
    """
    
    def __init__(self, config: SASRecConfig = None, num_items: int = None):
        super().__init__()
        self.config = config or default_config.sasrec
        
        if num_items is None:
            raise ValueError("num_items must be provided")
        
        self.num_items = num_items
        
        # Item embedding (add 1 for padding index 0)
        self.item_embedding = nn.Embedding(
            num_items + 1, 
            self.config.embedding_dim,
            padding_idx=0
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            self.config.embedding_dim,
            self.config.max_seq_length,
            self.config.dropout
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.config.embedding_dim,
                num_heads=self.config.num_attention_heads,
                d_ff=self.config.hidden_dim,
                dropout=self.config.dropout
            )
            for _ in range(self.config.num_transformer_blocks)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(self.config.embedding_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with better scaling for large vocabularies"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Use smaller std for large vocab to prevent gradient explosion
                std = 0.01 if module.weight.size(0) > 10000 else 0.02
                nn.init.normal_(module.weight, mean=0, std=std)
                # Zero out padding embedding
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
    
    def forward(
        self, 
        input_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_seq: (batch_size, seq_len) - item indices
            attention_mask: (batch_size, seq_len) - 1 for valid, 0 for padding
        Returns:
            (batch_size, seq_len, embedding_dim) - sequence representations
        """
        # Shift indices by 1 (0 is padding)
        input_seq = input_seq + 1
        input_seq = input_seq.clamp(0, self.num_items)
        
        # Embed items
        x = self.item_embedding(input_seq)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x
    
    def get_last_hidden(
        self, 
        input_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get the last valid hidden state for each sequence"""
        hidden = self.forward(input_seq, attention_mask)
        
        if attention_mask is not None:
            # Get last valid position (clamp to valid range)
            lengths = attention_mask.sum(dim=1).long() - 1
            lengths = lengths.clamp(min=0, max=hidden.size(1) - 1)  # Ensure valid indices
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            last_hidden = hidden[batch_idx, lengths]
        else:
            last_hidden = hidden[:, -1, :]
        
        return last_hidden


# =============================================================================
# PHOBERT ENCODER
# =============================================================================

class PhoBERTContentEncoder(nn.Module):
    """
    PhoBERT encoder for Vietnamese text content
    Outputs fixed-dim content representation
    """
    
    def __init__(self, config: PhoBERTConfig = None):
        super().__init__()
        self.config = config or default_config.phobert
        
        # Lazy loading to avoid import issues
        self.phobert = None
        self.tokenizer = None
        
        # Projection layer (trainable even if PhoBERT frozen)
        self.projection = nn.Sequential(
            nn.Linear(768, self.config.output_dim * 2),  # PhoBERT base is 768
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.output_dim * 2, self.config.output_dim),
            nn.LayerNorm(self.config.output_dim)
        )
    
    def load_phobert(self):
        """Load PhoBERT model and tokenizer"""
        if self.phobert is not None:
            return
        
        from transformers import AutoModel, AutoTokenizer
        
        print(f"📥 Loading PhoBERT from {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.phobert = AutoModel.from_pretrained(self.config.model_name)
        
        if self.config.freeze:
            for param in self.phobert.parameters():
                param.requires_grad = False
            print("   ❄️ PhoBERT weights frozen")
        
        print("   PhoBERT loaded successfully")
    
    def encode_text(self, texts: list, device: torch.device) -> torch.Tensor:
        """
        Encode list of texts to embeddings
        Args:
            texts: List of text strings
            device: Target device
        Returns:
            (batch_size, output_dim) embeddings
        """
        self.load_phobert()
        self.phobert = self.phobert.to(device)
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors='pt'
        ).to(device)
        
        # Get PhoBERT output
        with torch.no_grad() if self.config.freeze else torch.enable_grad():
            outputs = self.phobert(**inputs)
            # Use [CLS] token representation
            cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Project to target dimension
        return self.projection(cls_output)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-tokenized inputs
        """
        self.load_phobert()
        
        with torch.no_grad() if self.config.freeze else torch.enable_grad():
            outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
        
        return self.projection(cls_output)


# =============================================================================
# FUSION MODEL
# =============================================================================

class SASRecPhoBERTFusion(nn.Module):
    """
    Fusion model combining SASRec (sequential) + PhoBERT (content)
    
    Fusion strategies:
    - concat: Concatenate and project
    - attention: Cross-attention between sequence and content
    - gate: Gated fusion
    """
    
    def __init__(
        self, 
        num_items: int,
        sasrec_config: SASRecConfig = None,
        phobert_config: PhoBERTConfig = None,
        fusion_config: FusionConfig = None
    ):
        super().__init__()
        
        self.sasrec_config = sasrec_config or default_config.sasrec
        self.phobert_config = phobert_config or default_config.phobert
        self.fusion_config = fusion_config or default_config.fusion
        
        self.num_items = num_items
        
        # SASRec encoder
        self.sasrec = SASRecEncoder(self.sasrec_config, num_items)
        
        # PhoBERT encoder
        self.phobert_encoder = PhoBERTContentEncoder(self.phobert_config)
        
        # Fusion layers
        seq_dim = self.sasrec_config.embedding_dim
        content_dim = self.phobert_config.output_dim
        
        if self.fusion_config.fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(seq_dim + content_dim, self.fusion_config.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.fusion_config.dropout),
                nn.Linear(self.fusion_config.hidden_dim, self.fusion_config.output_dim)
            )
        elif self.fusion_config.fusion_type == "gate":
            self.gate = nn.Sequential(
                nn.Linear(seq_dim + content_dim, 1),
                nn.Sigmoid()
            )
            self.seq_proj = nn.Linear(seq_dim, self.fusion_config.output_dim)
            self.content_proj = nn.Linear(content_dim, self.fusion_config.output_dim)
        else:
            # Default: simple addition
            self.seq_proj = nn.Linear(seq_dim, self.fusion_config.output_dim)
            self.content_proj = nn.Linear(content_dim, self.fusion_config.output_dim)
        
        # Output projection for scoring
        self.output_layer = nn.Linear(self.fusion_config.output_dim, num_items)
        
    def get_sequence_representation(
        self, 
        input_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get user sequence representation from SASRec"""
        return self.sasrec.get_last_hidden(input_seq, attention_mask)
    
    def get_item_scores(
        self,
        input_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        content_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get scores for all items
        Args:
            input_seq: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            content_embeddings: (batch_size, content_dim) - optional content for context
        Returns:
            (batch_size, num_items) - scores for each item
        """
        # Get sequence representation
        seq_repr = self.get_sequence_representation(input_seq, attention_mask)
        
        if content_embeddings is not None:
            # Fuse with content
            if self.fusion_config.fusion_type == "concat":
                fused = self.fusion(torch.cat([seq_repr, content_embeddings], dim=-1))
            elif self.fusion_config.fusion_type == "gate":
                gate = self.gate(torch.cat([seq_repr, content_embeddings], dim=-1))
                fused = gate * self.seq_proj(seq_repr) + (1 - gate) * self.content_proj(content_embeddings)
            else:
                fused = self.seq_proj(seq_repr) + self.content_proj(content_embeddings)
        else:
            # Sequence only
            if hasattr(self, 'seq_proj'):
                fused = self.seq_proj(seq_repr)
            else:
                fused = self.fusion(torch.cat([seq_repr, torch.zeros_like(seq_repr)], dim=-1))
        
        # Score all items
        return self.output_layer(fused)
    
    def forward(
        self,
        input_seq: torch.Tensor,
        attention_mask: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass with BPR-style loss
        Returns: (loss, positive_scores, negative_scores)
        """
        # Get sequence representation
        seq_repr = self.get_sequence_representation(input_seq, attention_mask)
        
        # Get item embeddings
        pos_emb = self.sasrec.item_embedding(positive_items + 1)  # (batch, dim)
        neg_emb = self.sasrec.item_embedding(negative_items + 1)  # (batch, num_neg, dim)
        
        # Compute scores
        pos_scores = (seq_repr * pos_emb).sum(dim=-1)  # (batch,)
        neg_scores = (seq_repr.unsqueeze(1) * neg_emb).sum(dim=-1)  # (batch, num_neg)
        
        # BPR Loss: -log(sigmoid(pos - neg))
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8).mean()
        
        return loss, pos_scores, neg_scores


# =============================================================================
# LIGHTWEIGHT SASREC (WITHOUT PHOBERT FOR FASTER TRAINING)
# =============================================================================

class SASRecModel(nn.Module):
    """
    Pure SASRec model without PhoBERT for faster training
    PhoBERT can be added later for inference enhancement
    """
    
    def __init__(self, num_items: int, config: SASRecConfig = None):
        super().__init__()
        self.config = config or default_config.sasrec
        self.num_items = num_items
        
        # SASRec encoder
        self.encoder = SASRecEncoder(self.config, num_items)
    
    def forward(
        self,
        input_seq: torch.Tensor,
        attention_mask: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward with BPR loss"""
        # Get sequence representation
        seq_repr = self.encoder.get_last_hidden(input_seq, attention_mask)
        
        # Get item embeddings
        pos_emb = self.encoder.item_embedding(positive_items + 1)
        neg_emb = self.encoder.item_embedding(negative_items + 1)
        
        # Compute scores
        pos_scores = (seq_repr * pos_emb).sum(dim=-1)
        neg_scores = (seq_repr.unsqueeze(1) * neg_emb).sum(dim=-1)
        
        # BPR Loss with numerical stability
        diff = pos_scores.unsqueeze(1) - neg_scores
        diff = diff.clamp(-80, 80)  # Prevent overflow in sigmoid
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        
        # Additional check for NaN
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        return loss, pos_scores, neg_scores
    
    def predict(
        self, 
        input_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k items for given sequences
        Returns: (scores, indices)
        """
        self.eval()
        with torch.no_grad():
            # Get sequence representation
            seq_repr = self.encoder.get_last_hidden(input_seq, attention_mask)
            
            # Score all items
            all_item_emb = self.encoder.item_embedding.weight[1:]  # Exclude padding
            scores = torch.matmul(seq_repr, all_item_emb.T)
            
            # Get top-k
            top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)
        
        return top_scores, top_indices


if __name__ == "__main__":
    # Test model creation
    print("Testing SASRec model...")
    
    num_items = 1000
    batch_size = 4
    seq_len = 10
    
    model = SASRecModel(num_items)
    
    # Dummy input
    input_seq = torch.randint(0, num_items, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    positive = torch.randint(0, num_items, (batch_size,))
    negative = torch.randint(0, num_items, (batch_size, 4))
    
    loss, pos_scores, neg_scores = model(input_seq, attention_mask, positive, negative)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Pos scores shape: {pos_scores.shape}")
    print(f"Neg scores shape: {neg_scores.shape}")
    print("Model test passed!")
