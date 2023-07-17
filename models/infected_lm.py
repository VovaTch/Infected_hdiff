from typing import Dict, Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from loss import TotalLoss
from .base import BaseNetwork
from utils.other import SinusoidalPositionEmbeddings
from .multi_level_vqvae import VQ1D


class TransformerAutoregressor(BaseNetwork):
    def __init__(
        self,
        codebook: VQ1D,
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
        self.codebook = codebook

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

    def _step(self, batch: torch.Tensor, batch_idx: torch.Tensor):
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

        return outputs

    def training_step(self, batch: torch.Tensor, batch_idx: torch.Tensor):
        outputs = self._step(batch, batch_idx)
        for key, value in outputs.items():
            if "loss" in key:
                self.log(key, value)
        return outputs["total_loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: torch.Tensor):
        outputs = self._step(batch, batch_idx)
        for key, value in outputs.items():
            if "loss" in key:
                self.log(key + "_val", value)

    def generate_sequence(
        self,
        preliminary_seq: torch.Tensor,
        temperature: float = 1.0,
        song_idx: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        seq_length = preliminary_seq.shape[1]
        current_sequence = preliminary_seq[:, 0, :].copy()
        for _ in tqdm.tqdm(range(seq_length - 1), "Creating latent sequence..."):
            output_logits = self(current_sequence, song_idx=song_idx)[:, -1]
            sampled_codes = self._sample_codes(output_logits, temperature)
            current_sequence = torch.cat((current_sequence, sampled_codes), dim=1)

        return {"sequence": current_sequence}

    def _sample_codes(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        categorical_dist = torch.distributions.Categorical(logits=logits / temperature)
        samples = categorical_dist.sample()
        sampled_codes = self.codebook.vq_codebook.code_embedding[samples]
        return sampled_codes
