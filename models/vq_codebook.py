import torch
import torch.nn as nn
import numpy as np

class VQCodebook(nn.Module):
    
    def __init__(self, token_dim, num_tokens: int=512):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.code_embedding = nn.Parameter(torch.rand(num_tokens, token_dim))
        
    def apply_codebook(self, x_in: torch.Tensor, code_sg: bool=False):
        
        embedding_weights = self.code_embedding.transpose(0, 1)
        z_q, indices = vq_codebook_select(x_in, embedding_weights.detach() if code_sg else embedding_weights)
        
        return z_q, indices
        
        
    # def extract_indices(self, x_in: torch.Tensor):
        
    #     indices_range = torch.linspace(0, self.num_tokens - 1, self.num_tokens).unsqueeze(0).repeat((x_in.shape[0], 1)).to(x_in.device)
    #     embeddings = self.code_embedding(indices_range.int())
    #     distances = torch.cdist(x_in.transpose(1, 2), embeddings)
    #     indices = torch.argmin(distances, dim=2, keepdim=True)
    #     return indices
    
    # def apply_codebook(self, x_in: torch.Tensor):
    #     indices = self.extract_indices(x_in)
    #     return self.code_embedding(indices).squeeze(2).transpose(1, 2), indices
        
        

class NearestEmbedFunc(torch.autograd.Function):
    """
    Helper function for deriving the the nearest embeddings, as well as passing the gradients through the embeddings themselves.
    
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, emb: torch.Tensor):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.
                               format(input.size(1), emb.size(0)))

        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand to be broadcast-able
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(
                emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *
                         list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)
                                      ).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])

        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) ==
                           latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(
                ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) *
                                 idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None
    

def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


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
    return VQCodebookFunc.apply(x_in, emb_batch)