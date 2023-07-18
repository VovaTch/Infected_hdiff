from typing import Dict, Any, Optional, List

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
        input_channels: int = 8,
        masking_prob: float = 0.25,
        loss_obj: TotalLoss = None,
        **kwargs,
    ) -> pl.LightningModule:
        super().__init__(**kwargs)

        # initialize class variables
        self.num_heads = num_heads
        self.num_modules = num_modules
        self.hidden_size = hidden_size
        self.input_channels = input_channels
        self.codebook_size = codebook.vq_codebook.num_tokens
        self.masking_prob = masking_prob
        self.loss_obj = loss_obj
        self.codebook = codebook

        # Initialize decoder-only transformer
        self.decoder_layer = nn.TransformerDecoderLayer(
            hidden_size,
            num_heads,
            dropout=0.0,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.decoder_stack = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_modules
        )

        # Empty embedding
        self.empty_embedding = nn.Parameter(torch.randn((1, input_channels, 512)))

        # Positional encoding
        self.positional_encoding = SinusoidalPositionEmbeddings(self.hidden_size)

        # Fully connected layers
        self.input_fc = nn.Linear(input_channels, hidden_size)
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.codebook_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        prev_seq: List[torch.Tensor] or torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> Dict[str, Any]:
        """
        Forward method; passes the inputs and conditionals through the decoder-only transformer and
        produces logits for each query.

        Args:
            x (torch.Tensor): Size BS x V x ldim, tensor input to the transformer decoder.
            prev_seq (List[torch.Tensor] | torch.Tensor, optional): Size BS x V x ldim Previous sequence that is used
            for conditional input, supports either a torch tensor or a list of torch tensors. Defaults to None.
            mask (torch.Tensor, optional): Size BS x V x V, attention mask for the decoder queries. Defaults to None.

        Raises:
            Exception: Unknown type for the previous sequence

        Returns:
            Dict[str, Any]: A dictionary the contains a Tensor
        """

        # If the conditional is empty
        if prev_seq is None:
            prev_seq = self.empty_embedding.repeat((x.shape[0], 1, 1))
            conditional = self.input_fc(prev_seq)
            positional_emb_idx = torch.arange(0, self.empty_embedding.shape[1]).to(
                self.device
            )
            positional_emb = self.positional_encoding(positional_emb_idx)
            conditional += positional_emb

        # If the conditional is a list
        elif isinstance(prev_seq, List):
            conditional = torch.zeros((x.shape[0], 0, x.shape[2])).to(self.device)
            for ind_seq in prev_seq:
                ind_cond = self.input_fc(ind_seq)
                positional_emb_idx = torch.arange(0, ind_seq.shape[1]).to(self.device)
                positional_emb = self.positional_encoding(positional_emb_idx)
                ind_cond += positional_emb
                conditional = torch.cat((conditional, ind_cond), dim=1)

        # If the conditional is a tensor
        elif isinstance(prev_seq, torch.Tensor):
            conditional = self.input_fc(prev_seq)
            positional_emb_idx = torch.arange(0, prev_seq.shape[1]).to(self.device)
            positional_emb = self.positional_encoding(positional_emb_idx)
            conditional += positional_emb

        # Else raise an error
        else:
            raise Exception(f"Unknown conditional type {type(prev_seq)}")

        x = self.input_fc(x)  # BS x V x hs
        positional_emb_idx = torch.arange(0, x.shape[1]).to(self.device)
        positional_emb = self.positional_encoding(positional_emb_idx)
        x += positional_emb

        # Activate transformer
        x = self.decoder_stack(x, conditional, tgt_mask=mask).transpose(
            0, 1
        )  # BS x V x hs

        # Classification head
        x = self.output_mlp(x)

        # Logits and loss
        output = {"logits": x}

        return output

    def _step(self, batch: torch.Tensor, batch_idx: torch.Tensor):
        assert self.loss_obj is not None, "For training, there must be a loss object."

        slices, indices, prev_seq = (
            batch["music slice"],
            batch["latent indices"],
            batch["back conditional slice"],
        )
        mask_prob = (
            torch.zeros((slices.shape[0], slices.shape[1], slices.shape[1]))
            .to(self.device)
            .repeat((self.num_heads, 1, 1))
            + self.masking_prob
        )

        mask = torch.bernoulli(mask_prob).to(self.device)
        outputs = self(slices.transpose(1, 2), prev_seq.transpose(1, 2), mask)
        targets = {"latent indices": indices}

        outputs.update(self.loss_obj(outputs, targets))

        return outputs

    def training_step(self, batch: torch.Tensor, batch_idx: torch.Tensor):
        outputs = self._step(batch, batch_idx)
        for key, value in outputs.items():
            if "loss" in key:
                prog_bar = True if key == "total_loss" else False
                displayed_key = key.replace("_", " ")
                self.log("Training " + displayed_key, value, prog_bar=prog_bar)
        return outputs["total_loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: torch.Tensor):
        outputs = self._step(batch, batch_idx)
        for key, value in outputs.items():
            if "loss" in key:
                prog_bar = True if key == "total_loss" else False
                displayed_key = key.replace("_", " ")
                self.log("Validation " + displayed_key, value, prog_bar=prog_bar)

    def generate_sequence(
        self,
        preliminary_seq: torch.Tensor,
        preliminary_mask: torch.Tensor = None,
        temperature: float = 1.0,
        prev_slice: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        seq_length = preliminary_seq.shape[1]
        current_sequence = preliminary_seq.clone()
        current_mask = (
            preliminary_mask.clone() if preliminary_mask is not None else None
        )
        for seq_idx in tqdm.tqdm(range(seq_length - 1), "Creating latent sequence..."):
            if current_mask is None or not current_mask[0, seq_idx]:
                output_logits = self(
                    current_sequence, prev_slice=prev_slice, mask=current_mask
                )[:, seq_idx + 1, :]
                sampled_codes = self._sample_codes(output_logits, temperature)
                current_sequence[:, seq_idx + 1, :] = sampled_codes
            if current_mask is not None:
                current_mask[:, seq_idx + 1] = True

        return {"sequence": current_sequence}

    def _sample_codes(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        categorical_dist = torch.distributions.Categorical(logits=logits / temperature)
        samples = categorical_dist.sample()
        sampled_codes = self.codebook.vq_codebook.code_embedding[samples]
        return sampled_codes
