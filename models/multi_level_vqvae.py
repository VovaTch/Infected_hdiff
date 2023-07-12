from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq_codebook import VQCodebook
from .base import BaseNetwork
from loss import TotalLoss
from utils.other import ACTIVATIONS


class Res1DBlock(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_res_conv: int,
        dilation_factor: int,
        kernel_size: int,
        activation_type: str = "gelu",
    ):
        """
        1D Conv res block, similar to Jukebox paper. This is a try because the transformer one didn't
        regress to the wanted waveform, and the 1d vqvae doesn't reconstruct the sound well enough.
        """

        super().__init__()

        self.activation = ACTIVATIONS[activation_type]

        # Create conv, activation, norm blocks
        self.res_block_modules = nn.ModuleList([])
        for idx in range(num_res_conv):
            # Keep output dimension equal to input dim
            dilation = dilation_factor**idx
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

            if idx != num_res_conv - 1:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation(),
                        nn.BatchNorm1d(num_channels),
                    )
                )

            else:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation(),
                    )
                )

    def forward(self, x: torch.Tensor):
        x_init = x.clone()
        for seq_module in self.res_block_modules:
            x = seq_module(x)

        return x + x_init


class Res1DBlockReverse(Res1DBlock):
    def __init__(
        self,
        num_channels: int,
        num_res_conv: int,
        dilation_factor: int,
        kernel_size: int,
        activation_type: str = "gelu",
    ):
        super().__init__(
            num_channels, num_res_conv, dilation_factor, kernel_size, activation_type
        )

        # Create conv, activation, norm blocks
        self.res_block_modules = nn.ModuleList([])
        for idx in range(num_res_conv):
            # Keep output dimension equal to input dim
            dilation = dilation_factor ** (num_res_conv - idx - 1)
            padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

            if idx != num_res_conv - 1:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation(),
                        nn.BatchNorm1d(num_channels),
                    )
                )

            else:
                self.res_block_modules.append(
                    nn.Sequential(
                        nn.Conv1d(
                            num_channels,
                            num_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=padding,
                        ),
                        self.activation(),
                    )
                )


class ConvDownsample(nn.Module):
    """
    A small module handling downsampling via a convolutional layer instead of e.g. Maxpool.
    """

    def __init__(
        self, kernel_size: int, downsample_divide: int, in_dim: int, out_dim: int
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.downsample_divide = downsample_divide
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.padding_needed = (kernel_size - 2) + (downsample_divide - 2)
        self.padding_needed = (
            0 if self.padding_needed < 0 else self.padding_needed
        )  # Safeguard against negative padding

        # Define the convolutional layer
        self.conv_down = nn.Conv1d(
            in_dim, out_dim, kernel_size=kernel_size, stride=downsample_divide
        )

    def forward(self, x):
        x = F.pad(x, (0, self.padding_needed))
        return self.conv_down(x)


class Encoder1D(nn.Module):
    """
    Encoder class for the level 1 auto-encoder, this is constructed in a VAE manner.
    """

    def __init__(
        self,
        channel_list: List[int],
        dim_change_list: List[int],
        input_channels: int = 1,
        kernel_size: int = 5,
        num_res_block_conv: int = 3,
        dilation_factor: int = 3,
        dim_change_kernel_size: int = 5,
        activation_type: str = "gelu",
    ):
        super().__init__()
        assert (
            len(channel_list) == len(dim_change_list) + 1
        ), "The channel list length must be greater than the dimension change list by 1"
        self.last_dim = channel_list[-1]
        self.activation = ACTIVATIONS[activation_type]()

        # Create the module lists for the architecture
        self.init_conv = nn.Conv1d(
            input_channels, channel_list[0], kernel_size=kernel_size, padding=1
        )
        self.conv_list = nn.ModuleList(
            [
                Res1DBlock(
                    channel_list[idx],
                    num_res_block_conv,
                    dilation_factor,
                    kernel_size,
                    activation_type,
                )
                for idx in range(len(dim_change_list))
            ]
        )
        self.dim_change_list = nn.ModuleList(
            [
                ConvDownsample(
                    kernel_size=dim_change_kernel_size,
                    downsample_divide=dim_change_param,
                    in_dim=channel_list[idx],
                    out_dim=channel_list[idx + 1],
                )
                for idx, dim_change_param in enumerate(dim_change_list)
            ]
        )

    def forward(self, x):
        x = self.init_conv(x)

        for idx, (conv, dim_change) in enumerate(
            zip(self.conv_list, self.dim_change_list)
        ):
            x = conv(x)
            x = dim_change(x)

            if idx != len(self.dim_change_list) - 1:
                x = self.activation(x)

        return x


class Decoder1D(nn.Module):
    def __init__(
        self,
        channel_list: List[int],
        dim_change_list: List[int],
        input_channels: int = 1,
        kernel_size: int = 5,
        dim_add_kernel_add: int = 12,
        num_res_block_conv: int = 3,
        dilation_factor: int = 3,
        activation_type: str = "gelu",
    ):
        """
        A simpler decoder than the old version, maybe will still need to push here some attention.
        """

        super().__init__()
        assert (
            len(channel_list) == len(dim_change_list) + 1
        ), "The channel list length must be greater than the dimension change list by 1"
        self.activation = ACTIVATIONS[activation_type]()

        # Create the module lists for the architecture
        self.end_conv = nn.Conv1d(
            channel_list[-1], input_channels, kernel_size=3, padding=1
        )
        self.conv_list = nn.ModuleList(
            [
                Res1DBlockReverse(
                    channel_list[idx],
                    num_res_block_conv,
                    dilation_factor,
                    kernel_size,
                    activation_type,
                )
                for idx in range(len(dim_change_list))
            ]
        )
        assert (
            dim_add_kernel_add % 2 == 0
        ), "dim_add_kernel_size must be an even number."
        self.dim_change_list = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    channel_list[idx],
                    channel_list[idx + 1],
                    kernel_size=dim_change_list[idx] + dim_add_kernel_add,
                    stride=dim_change_list[idx],
                    padding=dim_add_kernel_add // 2,
                )
                for idx in range(len(dim_change_list))
            ]
        )

    def forward(self, z):
        for _, (conv, dim_change) in enumerate(
            zip(self.conv_list, self.dim_change_list)
        ):
            z = conv(z)
            z = dim_change(z)
            z = self.activation(z)

        x_out = self.end_conv(z)

        return x_out


class VQ1D(nn.Module):
    def __init__(self, token_dim, num_tokens: int = 8192):
        super().__init__()
        self.vq_codebook = VQCodebook(token_dim, num_tokens=num_tokens)

    def forward(self, z_e: torch.Tensor, extract_losses: bool = False):
        z_q, indices = self.vq_codebook.apply_codebook(z_e, code_sg=True)
        output = {"indices": indices, "v_q": z_q}

        if extract_losses:
            emb, _ = self.vq_codebook.apply_codebook(z_e.detach())
            output.update({"emb": emb})

        return output


class MultiLvlVQVariationalAutoEncoder(BaseNetwork):
    """
    VQ VAE that takes a music sample and converts it into latent space, hopefully faithfully reconstructing it later.
    This latent space is then used for the lowest level sample generation in a DiT like fashion.
    """

    def __init__(
        self,
        sample_rate: int,
        slice_time: float,
        hidden_size: int,
        latent_depth: int,
        loss_obj: TotalLoss = None,
        vocabulary_size: int = 8192,
        input_channels: int = 1,
        channel_dim_change_list: List[int] = [2, 2, 2, 4, 4],
        encoder_kernel_size: int = 5,
        encoder_dim_change_kernel_size: int = 5,
        decoder_kernel_size: int = 7,
        decoder_dim_change_kernel_add: int = 12,
        activation_type: str = "gelu",
        num_res_block_conv: int = 4,
        dilation_factor: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Parse arguments
        self.loss_obj = loss_obj
        self.cfg = kwargs
        self.sample_rate = sample_rate
        self.slice_time = slice_time
        self.samples_per_slice = int(sample_rate * slice_time)
        self.hidden_size = hidden_size
        self.latent_depth = latent_depth
        self.vocabulary_size = vocabulary_size
        self.channel_dim_change_list = channel_dim_change_list

        # Encoder parameter initialization
        encoder_channel_list = [
            hidden_size * (2 ** (idx + 1))
            for idx in range(len(channel_dim_change_list))
        ] + [latent_depth]
        # encoder_channel_list = [hidden_size for _ in range(len(channel_dim_change_list))] + [latent_depth]
        encoder_dim_changes = channel_dim_change_list
        decoder_channel_list = [latent_depth] + [
            hidden_size * (2 ** (idx + 1))
            for idx in reversed(range(len(channel_dim_change_list)))
        ]
        # decoder_channel_list = [latent_depth] + [hidden_size for _ in reversed(range(len(channel_dim_change_list)))]
        decoder_dim_changes = list(reversed(channel_dim_change_list))

        # Initialize network parts
        self.input_channels = input_channels

        self.encoder = Encoder1D(
            channel_list=encoder_channel_list,
            dim_change_list=encoder_dim_changes,
            input_channels=input_channels,
            kernel_size=encoder_kernel_size,
            num_res_block_conv=num_res_block_conv,
            dilation_factor=dilation_factor,
            dim_change_kernel_size=encoder_dim_change_kernel_size,
            activation_type=activation_type,
        )

        self.decoder = Decoder1D(
            channel_list=decoder_channel_list,
            dim_change_list=decoder_dim_changes,
            input_channels=input_channels,
            kernel_size=decoder_kernel_size,
            dim_add_kernel_add=decoder_dim_change_kernel_add,
            num_res_block_conv=num_res_block_conv,
            dilation_factor=dilation_factor,
            activation_type=activation_type,
        )

        self.vq_module = VQ1D(latent_depth, num_tokens=vocabulary_size)

    def encode(self, x: torch.Tensor):
        x_reshaped = x.reshape((x.shape[0], -1, self.input_channels)).permute((0, 2, 1))
        z_e = self.encoder(x_reshaped)
        return z_e

    def decode(self, z_e: torch.Tensor, origin_shape=None):
        vq_block_output = self.vq_module(z_e, extract_losses=True)
        x_out = self.decoder(vq_block_output["v_q"])

        if origin_shape is None:
            origin_shape = (z_e.shape[0], self.input_channels, -1)

        x_out = x_out.permute((0, 2, 1)).reshape(origin_shape)

        total_output = {**vq_block_output, "output": x_out}

        return x_out, total_output

    def forward(self, x: torch.Tensor):
        origin_shape = x.shape
        z_e = self.encode(x)
        _, total_output = self.decode(z_e, origin_shape=origin_shape)

        # x_reshaped = x.reshape((x.shape[0], -1, self.input_channels)).permute((0, 2, 1))

        # z_e = self.encoder(x_reshaped)
        # vq_block_output = self.vq_module(z_e, extract_losses=True)
        # x_out = self.decoder(vq_block_output['v_q'])

        # total_output = {**vq_block_output,
        #                 'output': x_out.permute((0, 2, 1)).reshape(origin_shape)}

        loss_target = {"music_slice": x, "z_e": z_e}

        if self.loss_obj is not None:
            total_output.update(self.loss_obj(total_output, loss_target))

        return total_output

    def training_step(self, batch, batch_idx):
        music_slice = batch["music slice"]
        total_output = self.forward(music_slice)
        total_loss = total_output["total_loss"]

        for key, value in total_output.items():
            if "loss" in key.split("_"):
                displayed_key = key.replace("_", " ")
                self.log(f"Training {displayed_key}", value)

        return total_loss

    def on_train_epoch_end(self):
        self.vq_module.vq_codebook.random_restart()
        self.vq_module.vq_codebook.reset_usage()

    def validation_step(self, batch, batch_idx):
        music_slice = batch["music slice"]
        total_output = self.forward(music_slice)

        for key, value in total_output.items():
            if "loss" in key.split("_"):
                prog_bar = True if key == "total_loss" else False
                displayed_key = key.replace("_", " ")
                self.log(f"Validation {displayed_key}", value, prog_bar=prog_bar)
