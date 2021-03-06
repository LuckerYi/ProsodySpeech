# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Style encoder of GST-Tacotron."""

from typeguard import check_argument_types
from typing import Sequence

import torch

from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention as BaseMultiHeadedAttention,  # NOQA
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)

class StyleEncoder(torch.nn.Module):
    """Style encoder.

    This module is style encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the referece encoder.
        conv_kernel_size (int, optional):
            Kernal size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): The number of GRU layers in the reference encoder.
        gru_units (int, optional): The number of GRU units in the reference encoder.

    Todo:
        * Support manual weight specification in inference.

    """

    def __init__(
        self,
        idim: int = 80,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
        style_vector_type: str = 'mha',
        hparams=None
    ):
        """Initilize global style encoder module."""
        assert check_argument_types()
        super(StyleEncoder, self).__init__()

        self.ref_enc = ReferenceEncoder(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
            hparams=hparams
        )
        self.stl = StyleTokenLayer(
            ref_embed_dim=gru_units,
            gst_tokens=gst_tokens,
            gst_token_dim=gst_token_dim,
            gst_heads=gst_heads,
        )
        
        self.choosestl = ChooseStyleTokenLayer(
            ref_embed_dim=gst_token_dim,
            gst_token_dim=gst_token_dim,
            gst_heads=gst_heads,
        )        
        self.hparams = hparams
    def forward(self, speech: torch.Tensor, spembs_: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, odim).

        Returns:
            Tensor: Style token embeddings (B, token_dim).

        """
        ref_embs = self.ref_enc(speech, spembs_=spembs_ if self.hparams.style_vector_type == 'mha' else None, mask=mask)
        style_embs = self.stl(ref_embs)
        return style_embs


class ReferenceEncoder(torch.nn.Module):
    """Reference encoder module.

    This module is refernece encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the referece encoder.
        conv_kernel_size (int, optional):
            Kernal size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): The number of GRU layers in the reference encoder.
        gru_units (int, optional): The number of GRU units in the reference encoder.

    """

    def __init__(
        self,
        idim=80,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
        hparams=None
    ):
        """Initilize reference encoder module."""
        assert check_argument_types()
        super(ReferenceEncoder, self).__init__()

        # check hyperparameters are valid
        if hparams.gst_reference_encoder == 'convs':
            assert conv_kernel_size % 2 == 1, "kernel size must be odd."
            assert (
                len(conv_chans_list) == conv_layers
            ), "the number of conv layers and length of channels list must be the same."

            convs = []
            padding = (conv_kernel_size - 1) // 2
            for i in range(conv_layers):
                conv_in_chans = 1 if i == 0 else conv_chans_list[i - 1]
                conv_out_chans = conv_chans_list[i]
                convs += [
                    torch.nn.Conv2d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding,
                    # Do not use bias due to the following batch norm
                    bias=True,
                    ),
                    # torch.nn.BatchNorm2d(conv_out_chans),
                    torch.nn.ReLU(inplace=True),
                ]
            self.convs = torch.nn.Sequential(*convs)
            self.conv_layers = conv_layers
            self.kernel_size = conv_kernel_size
            self.stride = conv_stride
            self.padding = padding
            # get the number of GRU input units
            gru_in_units = idim
            for i in range(conv_layers):
                gru_in_units = (
                    gru_in_units - conv_kernel_size + 2 * padding
                ) // conv_stride + 1
            gru_in_units *= conv_out_chans

        if hparams.gst_reference_encoder == 'multiheadattention':
            def get_positionwise_layer(
                positionwise_layer_type="linear",
                attention_dim=256,
                linear_units=2048,#1536
                dropout_rate=0.0,
                positionwise_conv_kernel_size=1,
            ):
                """Define positionwise layer."""
                if positionwise_layer_type == "linear":
                    positionwise_layer = PositionwiseFeedForward
                    positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
                elif positionwise_layer_type == "conv1d":
                    positionwise_layer = MultiLayeredConv1d
                    positionwise_layer_args = (
                        attention_dim,
                        linear_units,
                        positionwise_conv_kernel_size,
                        dropout_rate,
                    )
                elif positionwise_layer_type == "conv1d-linear":
                    positionwise_layer = Conv1dLinear
                    positionwise_layer_args = (
                        attention_dim,
                        linear_units,
                        positionwise_conv_kernel_size,
                        dropout_rate,
                    )
                else:
                    raise NotImplementedError("Support only linear or conv1d.")
                return positionwise_layer, positionwise_layer_args            
            
            positionwise_layer, positionwise_layer_args = get_positionwise_layer(
                positionwise_layer_type=hparams.positionwise_layer_type,
                attention_dim=hparams.num_mels,
                linear_units=hparams.eunits,
                dropout_rate=0.0,
                positionwise_conv_kernel_size=hparams.positionwise_conv_kernel_size,
            )            
            
            self.style_encoders = repeat(
                hparams.gst_reference_encoder_mha_layers,
                lambda lnum: EncoderLayer(
                    hparams.num_mels,
                    MultiHeadedAttention(
                        q_dim=hparams.num_mels,
                        k_dim=hparams.num_mels,
                        v_dim=hparams.num_mels,
                        n_head=hparams.gst_heads,
                        n_feat=hparams.num_mels,
                        dropout_rate=0.0
                    ),
                    positionwise_layer(*positionwise_layer_args),
                    hparams.transformer_enc_dropout_rate,
                    hparams.encoder_normalize_before,
                    hparams.encoder_concat_after,
                    hparams=hparams,
                    elementwise_affine=True
                ),
            )

        self.hparams = hparams
        if self.hparams.style_vector_type == 'gru':
            self.gru = torch.nn.GRU(gru_in_units, gru_units, gru_layers, batch_first=True)
        if self.hparams.style_vector_type == 'mha':
            self.mha_linear = torch.nn.Linear(hparams.num_mels, gru_units)

    def forward(self, speech: torch.Tensor, spembs_: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, idim).

        Returns:
            Tensor: Reference embedding (B, gru_units)

        """
        batch_size = speech.size(0)
        if self.hparams.gst_reference_encoder == 'convs':
            xs = speech.unsqueeze(1)
            # torch.save(xs,'/blob/yuanhyi/Transformer/Student_model_style/espnet-master/checkpoint/Tensor.pt')
            # print('xs,shape=',xs.shape)
            # print('xs=',xs)  # (B, 1, Lmax, idim)
            # print('self.convs=',self.convs)
            hs = self.convs(xs).transpose(1, 2)  # (B, Lmax', conv_out_chans, idim')
            time_length = hs.size(1)
            hs = hs.contiguous().view(batch_size, time_length, -1)  # (B, Lmax', gru_in_units)
            # print('convs_hs.shape=',hs.shape)
            # print('convs_hs=',hs)
            # raise
        if self.hparams.gst_reference_encoder == 'multiheadattention':
            xs = speech
            hs,_,_ = self.style_encoders(xs,mask)
        # NOTE(kan-bayashi): We need to care the length?
        if self.hparams.style_vector_type == 'gru':
            self.gru.flatten_parameters()
            _, ref_embs = self.gru(hs)  # (gru_layers, batch_size, gru_units), automatically the last frame
            ref_embs = ref_embs[-1]# (B, gru_units)
            
        if self.hparams.style_vector_type == 'mha':
            if not self.hparams.is_partial_refine:
                print('Smile for every day :)')
            ref_embs = torch.tanh(self.mha_linear(hs)) # (B, Lmax', gru_units)
            ref_embs = torch.max(ref_embs,dim=1)[0]# (B, gru_units)
        # print('ref_embs.shape=',ref_embs.shape)
        # print('ref_embs=',ref_embs)        
        return ref_embs


class StyleTokenLayer(torch.nn.Module):
    """Style token layer module.

    This module is style token layer introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        ref_embed_dim (int, optional): Dimension of the input reference embedding.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        dropout_rate (float, optional): Dropout rate in multi-head attention.

    """

    def __init__(
        self,
        ref_embed_dim: int = 128,#128
        gst_tokens: int = 10,
        gst_token_dim: int = 256,#384
        gst_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        """Initilize style token layer module."""
        assert check_argument_types()
        super(StyleTokenLayer, self).__init__()

        gst_embs = torch.randn(gst_tokens, gst_token_dim // gst_heads)
        self.register_parameter("gst_embs", torch.nn.Parameter(gst_embs))
        self.mha = MultiHeadedAttention(
            q_dim=ref_embed_dim,
            k_dim=gst_token_dim // gst_heads,#384/4=96
            v_dim=gst_token_dim // gst_heads,#384/4=96
            n_head=gst_heads,
            n_feat=gst_token_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, ref_embs: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            ref_embs (Tensor): Reference embeddings (B, ref_embed_dim).

        Returns:
            Tensor: Style token embeddings (B, gst_token_dim).

        """
        batch_size = ref_embs.size(0)
        # (num_tokens, token_dim) -> (batch_size, num_tokens, token_dim)
        gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        # print('gst_embs.shape=',gst_embs.shape)
        # print('gst_embs[0]=',gst_embs[0])
        # NOTE(kan-bayashi): Shoule we apply Tanh?
        ref_embs = ref_embs.unsqueeze(1)  # (batch_size, 1 ,ref_embed_dim)
        style_embs, att = self.mha(ref_embs, gst_embs, gst_embs, None, get_att=True)
        # print('att.shape=',att.shape)
        # print('att=',att)
        return style_embs.squeeze(1)


class ChooseStyleTokenLayer(torch.nn.Module):
    """Style token layer module.

    This module is style token layer introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        ref_embed_dim (int, optional): Dimension of the input reference embedding.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        dropout_rate (float, optional): Dropout rate in multi-head attention.

    """

    def __init__(
        self,
        ref_embed_dim: int = 128,#384
        gst_token_dim: int = 256,#384
        gst_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        """Initilize style token layer module."""
        assert check_argument_types()
        super(ChooseStyleTokenLayer, self).__init__()

        self.choose_mha = MultiHeadedAttention(
            q_dim=ref_embed_dim,#384
            k_dim=gst_token_dim,#384
            v_dim=gst_token_dim,#384
            n_head=gst_heads,
            n_feat=gst_token_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, ref_embs: torch.Tensor, style_tokens: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            ref_embs (Tensor): Q (B, Tx, ref_embed_dim).
            style_tokens (Tensor): K,V (B, 17, ref_embed_dim).
        Returns:
            Tensor: Style token embeddings (B, Tx, ref_embed_dim).

        """
        # (num_tokens, token_dim) -> (batch_size, num_tokens, token_dim)
        # NOTE(kan-bayashi): Shoule we apply Tanh?
        # ref_embs = ref_embs.unsqueeze(1)  # (batch_size, 1 ,ref_embed_dim)
        style_embs, att = self.choose_mha(ref_embs, style_tokens, style_tokens, None, get_att=True)

        return style_embs, att
        
class MultiHeadedAttention(BaseMultiHeadedAttention):
    """Multi head attention module with different input dimension."""

    def __init__(self, q_dim, k_dim, v_dim, n_head, n_feat, dropout_rate=0.0):
        """Initialize multi head attention module."""
        # NOTE(kan-bayashi): Do not use super().__init__() here since we want to
        #   overwrite BaseMultiHeadedAttention.__init__() method.
        torch.nn.Module.__init__(self)
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = torch.nn.Linear(q_dim, n_feat)
        self.linear_k = torch.nn.Linear(k_dim, n_feat)
        self.linear_v = torch.nn.Linear(v_dim, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)

# class YYH_Attention(torch.nn.Module):
#     def __init__(self, gru_in_units, gru_out_units, hparams):
#         super(YYH_Attention, self).__init__()
#         self.hparams = hparams
#         self.gru = torch.nn.GRU(gru_in_units, gru_out_units, 1, batch_first=True)
