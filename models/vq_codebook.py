import torch
import torch.nn as nn

class VQCodebook(nn.Module):
    
    def __init__(self, token_dim, num_tokens: int=512):
        super().__init__()
        
        self.token_dim = token_dim
        self.code_embedding = nn.Embedding(num_tokens, token_dim)
        
    def extract_indices(self, x_in: torch.Tensor):
        
        indices_range = torch.linspace(0, self.token_dim, 1).unsqueeze(0).repeat((x_in.shape[0], 1))
        embeddings = self.code_embedding(indices_range)
        distances = torch.cdist(x_in, embeddings)
        indices = torch.argmin(distances, dim=2, keepdim=True)
        return indices
    
    def apply_codebook(self, indices: torch.Tensor):
        return self.code_embedding(indices)
        
        
    