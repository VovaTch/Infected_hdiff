import torch
import torch.nn as nn
import numpy as np

class VQCodebook(nn.Module):
    """
    Parameter holder for the embeddings for the VQVAE. This also references to the function that computes gradients past the quantizer.
    """
    def __init__(self, token_dim, num_tokens: int=512):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.code_embedding = nn.Parameter(torch.rand(num_tokens, token_dim))
        
    def apply_codebook(self, x_in: torch.Tensor, code_sg: bool=False):
        
        embedding_weights = self.code_embedding.transpose(0, 1)
        z_q, indices = vq_codebook_select(x_in, embedding_weights.detach() if code_sg else embedding_weights)
        
        return z_q, indices
        


class VQCodebookFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x_in: torch.Tensor, embedding_weights: torch.Tensor):
        """
        Autograd function for the index selection. According to the VQ-VAE paper, the gradient for x_in should be a mirror, to the x_out.

        Args:
            x_in (torch.Tensor): Input, should be a BS x emb_size x codes (16 x 147)
            embedding_weights (torch.Tensor): Embedding input tensor, should be BS x emb_size x num_codes (4 x 16 x 512)
        """
        
        ctx.batch_size = x_in.shape[0]
        
        embedding_batch = embedding_weights.unsqueeze(0).repeat((x_in.shape[0], 1, 1))
        x_in_t = x_in.transpose(1, 2).float()
        embedding_batch_t = embedding_batch.transpose(1, 2).float()
        embedding_batch_flat = embedding_batch_t.flatten(start_dim=0, end_dim=1)
        
        distances = torch.cdist(x_in_t, embedding_batch_t) # 4 x 147 x 512
        indices = torch.argmin(distances, dim=2, keepdim=True) # 4 x 147 x 1
        x_out = torch.index_select(embedding_batch_flat, dim=0, index=indices.flatten())
        
        x_out = x_out.view((x_in.shape[0], x_in.shape[2], x_in.shape[1]))
        x_out = x_out.transpose(1, 2)
        
        ctx.save_for_backward(embedding_weights, indices)
        
        return x_out, indices
        
    @staticmethod
    def backward(ctx, grad_outputs, indices):
        
        grad_input = None
        grad_emb = None
        
        if ctx.needs_input_grad[0]:
            
            grad_input = grad_outputs
            
        if ctx.needs_input_grad[1]:
            
            embedding_weights, indices = ctx.saved_variables
            grad_emb = torch.zeros_like(embedding_weights)
            
            # Feed the gradients into the grad_emb file
            
            for batch_idx, batch in enumerate(indices.flatten(start_dim=1)):
                running_idx = 0
                for idx in batch:
                    
                    idx_value = idx.item()
                    
                    grad_emb[:, idx_value] += grad_outputs[batch_idx, :, running_idx] / (indices.flatten().shape[0])
                    running_idx += 1
                    
        return grad_input, grad_emb, None, None
    

def vq_codebook_select(x_in, emb_batch):
    """
    Applies the vq codebook function, allowing to pass gradients through it.
    """
    return VQCodebookFunc.apply(x_in, emb_batch)