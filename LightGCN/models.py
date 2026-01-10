"""
LightGCN Model
Simplified Graph Convolution for Recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import scipy.sparse as sp
import numpy as np

from config import config, ModelConfig


def sparse_to_torch(adj: sp.csr_matrix, device: str = 'cpu') -> torch.sparse.FloatTensor:
    """Convert scipy sparse to torch sparse tensor"""
    coo = adj.tocoo()
    indices = torch.LongTensor([coo.row, coo.col])
    values = torch.FloatTensor(coo.data)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, coo.shape)
    return sparse_tensor.to(device)


class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network
    
    Key features:
    - No feature transformation (no W matrices)
    - No nonlinear activation
    - Only neighborhood aggregation
    - Layer combination with mean pooling
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        adj_matrix: sp.csr_matrix,
        cfg: ModelConfig = None
    ):
        super().__init__()
        self.cfg = cfg or config.model
        self.num_users = num_users
        self.num_items = num_items
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, self.cfg.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, self.cfg.embedding_dim)
        
        # Initialize
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Normalized adjacency matrix (will be set by set_graph)
        self.register_buffer('adj_matrix', None)
        if adj_matrix is not None:
            self.set_graph(adj_matrix)
    
    def set_graph(self, adj_matrix: sp.csr_matrix):
        """Set the graph adjacency matrix"""
        sparse_tensor = sparse_to_torch(adj_matrix, self.user_embedding.weight.device)
        self.adj_matrix = sparse_tensor
    
    def get_all_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get final embeddings after LightGCN propagation
        
        Returns:
            user_embeddings: (num_users, dim)
            item_embeddings: (num_items, dim)
        """
        # Concatenate user and item embeddings
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        emb_list = [all_emb]
        
        # LightGCN propagation
        for layer in range(self.cfg.num_layers):
            all_emb = torch.sparse.mm(self.adj_matrix, all_emb)
            emb_list.append(all_emb)
        
        # Layer combination (mean)
        final_emb = torch.stack(emb_list, dim=0).mean(dim=0)
        
        user_emb = final_emb[:self.num_users]
        item_emb = final_emb[self.num_users:]
        
        return user_emb, item_emb
    
    def forward(
        self,
        user_indices: torch.Tensor,
        pos_item_indices: torch.Tensor,
        neg_item_indices: torch.Tensor
    ) -> dict:
        """
        Forward pass with BPR loss
        
        Args:
            user_indices: (batch_size,)
            pos_item_indices: (batch_size,)
            neg_item_indices: (batch_size,)
        """
        # Get propagated embeddings
        all_user_emb, all_item_emb = self.get_all_embeddings()
        
        # Get batch embeddings
        user_emb = all_user_emb[user_indices]
        pos_emb = all_item_emb[pos_item_indices]
        neg_emb = all_item_emb[neg_item_indices]
        
        # BPR loss
        pos_scores = (user_emb * pos_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_emb).sum(dim=-1)
        
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # L2 regularization on initial embeddings
        reg_loss = (
            self.user_embedding.weight[user_indices].norm(2).pow(2) +
            self.item_embedding.weight[pos_item_indices].norm(2).pow(2) +
            self.item_embedding.weight[neg_item_indices].norm(2).pow(2)
        ) / user_indices.size(0)
        
        return {
            'loss': bpr_loss + config.training.weight_decay * reg_loss,
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss
        }
    
    def predict(self, user_indices: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k recommendations for users"""
        self.eval()
        with torch.no_grad():
            all_user_emb, all_item_emb = self.get_all_embeddings()
            user_emb = all_user_emb[user_indices]
            scores = torch.matmul(user_emb, all_item_emb.T)
            return torch.topk(scores, k=min(top_k, self.num_items), dim=-1)
    
    def get_user_embedding(self, user_idx: int) -> torch.Tensor:
        """Get embedding for a single user"""
        with torch.no_grad():
            all_user_emb, _ = self.get_all_embeddings()
            return all_user_emb[user_idx]
    
    def get_item_embedding(self, item_idx: int) -> torch.Tensor:
        """Get embedding for a single item"""
        with torch.no_grad():
            _, all_item_emb = self.get_all_embeddings()
            return all_item_emb[item_idx]


if __name__ == "__main__":
    import scipy.sparse as sp
    
    print("Testing LightGCN...")
    
    num_users, num_items = 100, 200
    
    # Create dummy adjacency matrix
    interactions = [(i, i % num_items) for i in range(num_users)]
    interactions += [(i, (i + 1) % num_items) for i in range(num_users)]
    
    n_nodes = num_users + num_items
    rows, cols = [], []
    for u, i in interactions:
        rows += [u, num_users + i]
        cols += [num_users + i, u]
    
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    
    # Normalize
    degrees = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degrees + 1e-10, -0.5)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    
    # Create model
    model = LightGCN(num_users, num_items, norm_adj.tocsr())
    
    # Test forward
    users = torch.randint(0, num_users, (32,))
    pos_items = torch.randint(0, num_items, (32,))
    neg_items = torch.randint(0, num_items, (32,))
    
    output = model(users, pos_items, neg_items)
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"BPR: {output['bpr_loss'].item():.4f}")
    
    # Test predict
    scores, indices = model.predict(users[:4], top_k=5)
    print(f"Predictions: {indices.shape}")
    
    print("LightGCN test passed!")
