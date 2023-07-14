from typing import Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn

from loss import TotalLoss
from .base import BaseNetwork
from utils.other import SinusoidalPositionEmbeddings


class TransformerAutoregressor(BaseNetwork):
    def __init__(
        self,
        num_heads: int = 4,
        num_modules: int = 6,
        hidden_size: int = 256,
        codebook_size: int = 1024,
        max_num_songs: int = 100,
        input_channels: int = 8,
        masking_prob: float = 0.25,
        loss_obj: TotalLoss = None,
        **kwargs
    ) -> pl.LightningModule:
        super().__init__(**kwargs)

        # initialize class variables
        self.num_heads = num_heads
        self.num_modules = num_modules
        self.hidden_size = hidden_size
        self.input_channels = input_channels
        self.codebook_size = codebook_size
        self.masking_prob = masking_prob
        self.loss_obj = loss_obj

        # Initialize decoder-only transformer
        self.decoder_layer = nn.TransformerDecoderLayer(
            hidden_size, num_heads, dropout=0.0, activation="gelu", norm_first=True
        )
        self.decoder_stack = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_modules
        )

        # Song embedding
        self.song_embedding = nn.Embedding(
            num_embeddings=max_num_songs + 1, embedding_dim=hidden_size
        )
        self.song_cond_idx_matching = {}

        # Positional encoding
        self.positional_encoding = SinusoidalPositionEmbeddings(self.hidden_size)

        # Fully connected layers
        self.input_fc = nn.Linear(input_channels, hidden_size)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, codebook_size),
        )

    def forward(
        self, x: torch.Tensor, song_idx: torch.Tensor = None, mask: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        Forward method; assumes x in the shape of BS x V x in_dim, assumes song idx in the shape of BS x n_songs.
        Assumes empty index is -1, as the song_idx tensor must be a tensor. Mask in the shape of BS x V x V

        Args:
            x (torch.Tensor): _description_
            song_idx (torch.Tensor, optional): _description_. Defaults to None.
        """

        # If the song idx is empty, input empty embeddings
        if song_idx is None:
            song_idx = torch.zeros((x.shape[0], 1)).to(self.device)

        conditional_emb = self.song_embedding(song_idx + 1)

        x = self.input_fc(x)  # BS x V x hs
        positional_emb_idx = torch.arange(0, x.shape[1]).to(self.device)
        positional_emb = self.positional_encoding(positional_emb_idx)
        x += positional_emb

        # Activate transformer
        x = self.decoder_stack(x, conditional_emb, tgt_mask=mask)  # BS x V x hs

        # Classification head
        x = self.output_mlp(x)

        # Logits and loss
        output = {"logits": x}

        return output

    def training_step(self, batch, batch_idx):
        assert self.loss_obj is not None, "For training, there must be a loss object."

        slices, indices, track_idx = (
            batch["music slices"],
            batch["latent indices"],
            batch["track index"],
        )
        mask_prob = (
            torch.zeros((slices.shape[0], slices.shape[1], slices.shape[1])).to(
                self.device
            )
            + self.masking_prob
        )
        track_idx = track_idx.unsqueeze(-1)
        mask = torch.bernoulli(mask_prob).to(self.device)
        outputs = self(slices, track_idx, mask)
        targets = {"latent indices": indices}

        outputs.update(self.loss_obj(outputs, targets))

        return outputs["total_loss"]
