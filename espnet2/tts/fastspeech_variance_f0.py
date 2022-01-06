# -*- coding: utf-8 -*-

# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Fastspeech related modules for ESPnet2."""

from typing import Dict
from typing import Sequence
from typing import Tuple

from math import sqrt

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor, DurationPredictorLoss
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_gpu
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.gst.style_encoder import StyleEncoder, ChooseStyleTokenLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

class FastSpeechLoss(torch.nn.Module):
    """Loss function module for feed-forward Transformer."""

    def __init__(self, use_masking=True, use_weighted_masking=False, hparams=None):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to weighted masking in loss calculation.

        """
        super(FastSpeechLoss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.l2_criterion = torch.nn.MSELoss()
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)
        self.pitch_criterion = DurationPredictorLoss(reduction=reduction) if hparams.log_f0 else torch.nn.MSELoss()

    def forward(self, after_outs, before_outs, d_outs, ys, ds, ilens, olens, p_outs=None, ps=None):
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            d_outs (Tensor): Batch of outputs of duration predictor (B, Tmax).
            ys (Tensor): Batch of target features (B, Lmax, odim).
            ds (Tensor): Batch of durations (B, Tmax).
            ps (Tensor): Batch of pitchs (B, Tmax, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ys.device) if p_outs is not None else None
            d_outs = d_outs.masked_select(duration_masks)
            p_outs = p_outs.masked_select(pitch_masks) if p_outs is not None else None
            ds = ds.masked_select(duration_masks)
            ps = ps.masked_select(pitch_masks) if ps is not None else None
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            after_outs = (
                after_outs.masked_select(out_masks) if after_outs is not None else None
            )
            ys = ys.masked_select(out_masks)

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        l2_loss = self.l2_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss += self.l1_criterion(after_outs, ys)
            l2_loss += self.l2_criterion(after_outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.pitch_criterion(p_outs, ps) if ps is not None else None

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= ys.size(0) * ys.size(2)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            duration_weights = (
                duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            )
            duration_weights /= ds.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = (
                duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            )

        return l1_loss, duration_loss, l2_loss, pitch_loss

class VariancePredictor(torch.nn.Module):
    """Variance predictor module.

    This is a module of variacne predictor described in `FastSpeech 2:
    Fast and High-Quality End-to-End Text to Speech`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558

    """

    def __init__(
        self,
        idim: int,
        n_layers: int = 2,
        n_chans: int = 384,
        kernel_size: int = 3,
        bias: bool = True,
        dropout_rate: float = 0.5,
        hparams=None,
        elementwise_affine=False
    ):
        """Initilize duration predictor module.

        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.

        """
        assert check_argument_types()
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.norm = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                )
            ]
            self.norm += [LayerNorm(n_chans,hparams=hparams,dim=1,elementwise_affine=elementwise_affine)]
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, xs: torch.Tensor, x_masks: torch.Tensor = None, spembs_: torch.Tensor = None) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).

        """
        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for i, f in enumerate(self.conv):
            xs = self.dropout(self.norm[i](f(xs), spembs_))  # (B, C, Tmax)

        xs = self.linear(xs.transpose(1, 2))  # (B, Tmax, 1)

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs

class FastSpeech(AbsTTS):
    """FastSpeech module for end-to-end text-to-speech.

    This is a module of FastSpeech, feed-forward Transformer with duration predictor
    described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_, which
    does not require any auto-regressive processing during inference, resulting in
    fast decoding compared with auto-regressive Transformer.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    Args:
        idim (int): Dimension of the inputs.
        odim (int): Dimension of the outputs.
        elayers (int, optional): Number of encoder layers.
        eunits (int, optional): Number of encoder hidden units.
        dlayers (int, optional): Number of decoder layers.
        dunits (int, optional): Number of decoder hidden units.
        use_scaled_pos_enc (bool, optional):
            Whether to use trainable scaled positional encoding.
        encoder_normalize_before (bool, optional):
            Whether to perform layer normalization before encoder block.
        decoder_normalize_before (bool, optional):
            Whether to perform layer normalization before decoder block.
        encoder_concat_after (bool, optional): Whether to concatenate attention
            layer's input and output in encoder.
        decoder_concat_after (bool, optional): Whether to concatenate attention
            layer's input and output in decoder.
        duration_predictor_layers (int, optional): Number of duration predictor layers.
        duration_predictor_chans (int, optional): Number of duration predictor channels.
        duration_predictor_kernel_size (int, optional):
            Kernel size of duration predictor.
        spk_embed_dim (int, optional): Number of speaker embedding dimensions.
        spk_embed_integration_type: How to integrate speaker embedding.
        use_gst (str, optional): Whether to use global style token.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        gst_conv_layers (int, optional): The number of conv layers in GST.
        gst_conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in GST.
        gst_conv_kernel_size (int, optional): Kernal size of conv layers in GST.
        gst_conv_stride (int, optional): Stride size of conv layers in GST.
        gst_gru_layers (int, optional): The number of GRU layers in GST.
        gst_gru_units (int, optional): The number of GRU units in GST.
        reduction_factor (int, optional): Reduction factor.
        transformer_enc_dropout_rate (float, optional):
            Dropout rate in encoder except attention & positional encoding.
        transformer_enc_positional_dropout_rate (float, optional):
            Dropout rate after encoder positional encoding.
        transformer_enc_attn_dropout_rate (float, optional):
            Dropout rate in encoder self-attention module.
        transformer_dec_dropout_rate (float, optional):
            Dropout rate in decoder except attention & positional encoding.
        transformer_dec_positional_dropout_rate (float, optional):
            Dropout rate after decoder positional encoding.
        transformer_dec_attn_dropout_rate (float, optional):
            Dropout rate in deocoder self-attention module.
        init_type (str, optional):
            How to initialize transformer parameters.
        init_enc_alpha (float, optional):
            Initial value of alpha in scaled pos encoding of the encoder.
        init_dec_alpha (float, optional):
            Initial value of alpha in scaled pos encoding of the decoder.
        use_masking (bool, optional):
            Whether to apply masking for padded part in loss calculation.
        use_weighted_masking (bool, optional):
            Whether to apply weighted masking in loss calculation.

    """

    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6,
        eunits: int = 1536,
        dlayers: int = 6,
        dunits: int = 1536,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_normalize_before: bool = False,
        decoder_normalize_before: bool = False,
        is_spk_layer_norm: bool = False,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        reduction_factor: int = 1,
        spk_embed_dim: int = None,
        spk_embed_integration_type: str = "add",
        use_gst: bool = False,
        gst_tokens: int = 10,
        gst_heads: int = 4,
        gst_conv_layers: int = 6,
        gst_conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        gst_conv_kernel_size: int = 3,
        gst_conv_stride: int = 2,
        gst_gru_layers: int = 1,
        gst_gru_units: int = 128,
        # training related
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        duration_predictor_dropout_rate: float = 0.1,
        postnet_dropout_rate: float = 0.5,
        hparams=None,
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        init_dec_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
    ):
        """Initialize FastSpeech module."""
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.reduction_factor = reduction_factor
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.use_gst = use_gst
        self.spk_embed_dim = spk_embed_dim
        self.hparams = hparams
        if self.hparams.is_multi_speakers:
            self.spk_embed_integration_type = spk_embed_integration_type

        # use idx 0 as padding idx
        self.padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        # define encoder
        # print(idim)
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx
        )

        if self.hparams.is_multi_speakers:
            self.speaker_embedding = torch.nn.Embedding(
                    hparams.n_speakers, self.spk_embed_dim)
            std = sqrt(2.0 / (hparams.n_speakers + self.spk_embed_dim))
            val = sqrt(3.0) * std  # uniform bounds for std
            self.speaker_embedding.weight.data.uniform_(-val, val)
            self.spkemb_projection = torch.nn.Linear(hparams.spk_embed_dim, hparams.spk_embed_dim) 


        self.encoder = Encoder(
            idim=idim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=encoder_input_layer,
            dropout_rate=transformer_enc_dropout_rate,
            positional_dropout_rate=transformer_enc_positional_dropout_rate,
            attention_dropout_rate=transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=encoder_normalize_before,
            is_spk_layer_norm=is_spk_layer_norm,
            concat_after=encoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            hparams=hparams
        )

        # define GST
        if self.use_gst:
            self.gst = StyleEncoder(
                idim=odim,  # the input is mel-spectrogram
                gst_tokens=gst_tokens,
                gst_token_dim=adim,
                gst_heads=gst_heads,
                conv_layers=gst_conv_layers,
                conv_chans_list=gst_conv_chans_list,
                conv_kernel_size=gst_conv_kernel_size,
                conv_stride=gst_conv_stride,
                gru_layers=gst_gru_layers,
                gru_units=gst_gru_units,
                hparams=hparams
            )
            if self.hparams.style_embed_integration_type == "concat":
                self.gst_projection = torch.nn.Linear(adim + adim, adim)

        # define additional projection for speaker embedding
        if self.hparams.is_multi_speakers:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
            hparams=hparams
        )

        if self.hparams.use_gaussian_upsampling:
            self.sigma_predictor = DurationPredictor(
                idim=adim + 1,
                n_layers=duration_predictor_layers,
                n_chans=duration_predictor_chans,
                kernel_size=duration_predictor_kernel_size,
                dropout_rate=duration_predictor_dropout_rate,
                hparams=hparams
                )

        if self.hparams.use_f0:
            self.pitch_predictor = VariancePredictor(
                idim=adim,
                n_layers=hparams.pitch_predictor_layers,
                n_chans=hparams.pitch_predictor_chans,
                kernel_size=hparams.pitch_predictor_kernel_size,
                dropout_rate=hparams.pitch_predictor_dropout,
                hparams=hparams,
                elementwise_affine=False
            )
            # NOTE(kan-bayashi): We use continuous pitch + FastPitch style avg
            self.pitch_embed = torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=1,
                    out_channels=adim,
                    kernel_size=hparams.pitch_embed_kernel_size,
                    padding=(hparams.pitch_embed_kernel_size - 1) // 2,
                ),
                torch.nn.Dropout(hparams.pitch_embed_dropout),
            )            

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        # NOTE: we use encoder as decoder
        # because fastspeech's decoder is the same as encoder
        self.decoder = Encoder(
            idim=0,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=dunits,
            num_blocks=dlayers,
            input_layer=None,
            dropout_rate=transformer_dec_dropout_rate,
            positional_dropout_rate=transformer_dec_positional_dropout_rate,
            attention_dropout_rate=transformer_dec_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=decoder_normalize_before,
            is_spk_layer_norm=is_spk_layer_norm,
            concat_after=decoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            hparams=hparams
        )

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)

        # define postnet
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=postnet_dropout_rate,
            )
        )

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha,
        )

        # define criterions
        self.criterion = FastSpeechLoss(
            use_masking=use_masking, use_weighted_masking=use_weighted_masking, hparams=hparams
        )

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        ys: torch.Tensor = None,
        olens: torch.Tensor = None,
        ds: torch.Tensor = None,
        ps: torch.Tensor = None,
        spk_ids: torch.Tensor = None,
        style_ids: torch.Tensor = None,
        utt_mels: list = None,
        is_inference: bool = False,
        alpha: float = 1.0,
        open_choosestl: bool = False,
        is_MI_step: bool = False,
        style_embs_generated: torch.Tensor = None,
    ) -> Sequence[torch.Tensor]:
        # integrate speaker embedding
        att = None
        hs_random_choosed = None
        style_tokens = style_embs_generated

        if self.hparams.is_multi_speakers:
            # print('spk_ids.shape=',spk_ids.shape)
            spembs_ = self.spkemb_projection(self.speaker_embedding(spk_ids))        
        # forward encoder
        x_masks = self._source_mask(ilens)
        # forward decoder
        if olens is not None and not is_inference:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None        
        # print(spembs_)
        hs, _ = self.encoder(xs, x_masks, spembs_=spembs_)  # (B, Tmax, adim)

        hs = self._integrate_with_spk_embed(hs, spembs_) if self.hparams.is_multi_speakers else hs

        hs_random_choosed = hs[:,torch.randperm(hs.size()[1])][:,0] if self.hparams.use_mutual_information else None #(B, gst_token_dim)
        
        if self.use_gst:
            if self.hparams.is_partial_refine and self.hparams.is_refine_style:
                if style_embs_generated is None:
                    style_embs = []
                    print('len(utt_mels)=',len(utt_mels))
                    for i in range(len(utt_mels)):
                        print('utt_mels[i].shape=',utt_mels[i].shape)
                        # style_embs.append(self.gst(to_gpu(utt_mels[i]), spembs_=spembs_[0].unsqueeze(0)))#(1, gst_token_dim)
                        style_embs.append(self.gst.eval()(to_gpu(utt_mels[i]), spembs_=None))#(1, gst_token_dim)
                    style_embs = torch.cat(style_embs,dim=0)#(17, gst_token_dim)
                    print('style_embs.shape=',style_embs.shape)
                    print('style_embs=',style_embs)
                    style_tokens = style_embs.unsqueeze(0).expand(hs.size(0), -1, -1)#(B, 17, gst_token_dim)
                # print('style_tokens.shape=',style_tokens.shape)
                style_embs, att = self.gst.choosestl(hs, style_tokens)#(B, Tx, gst_token_dim)
                # integrate with GST
                if self.hparams.style_embed_integration_type == "concat":
                    hs = self.gst_projection(torch.cat([hs,style_embs],dim=-1))
                    print('spembs_.shape=',spembs_.shape)
                else:
                    hs = hs + style_embs
            else:
                style_embs = self.gst(ys, spembs_=None, mask=h_masks) if not open_choosestl else self.gst.eval()(ys, spembs_=None, mask=h_masks)#(B, gst_token_dim)
                print('style_embs.shape=',style_embs.shape)
                print('style_embs=',style_embs)

                if self.hparams.use_mutual_information and is_MI_step:
                    return hs_random_choosed, style_embs

                if self.hparams.gst_train_att and open_choosestl:
                    style_embs = style_embs[torch.randperm(style_embs.size()[0])] if self.hparams.shuffle else style_embs
                    style_tokens = style_embs.unsqueeze(0).expand(hs.size(0), -1, -1)#(B, B, gst_token_dim)                   
                    style_embs, att = self.gst.choosestl(hs, style_tokens.detach())#(B, Tx, gst_token_dim)
                    print('here')
                # integrate with GST
                if self.hparams.style_embed_integration_type == "concat":
                    if self.hparams.gst_train_att and open_choosestl:
                        hs = self.gst_projection(torch.cat([hs,style_embs],dim=-1))#(B, Tx, gst_token_dim)
                    else:   
                        hs = self.gst_projection(torch.cat([hs,style_embs.unsqueeze(1).expand(-1, hs.size(1), -1)],dim=-1))#(B, Tx, gst_token_dim)
                    print('params.gst_train_att=',self.hparams.gst_train_att)
                    print('open_choosestl=',open_choosestl)
                    print('spembs_.shape=',spembs_.shape)
                else:
                    hs = hs + (style_embs.unsqueeze(1) if self.hparams.gst_train_att else style_embs)

        # forward duration predictor and length regulator
        d_masks = make_pad_mask(ilens).to(xs.device)

        p_outs = self.pitch_predictor(hs, d_masks.unsqueeze(-1)) if self.hparams.use_f0 else None

        if is_inference:
            d_outs = self.duration_predictor.inference(hs, d_masks, spembs_=spembs_)  # (B, Tmax)
            hs = hs + (self.pitch_embed(p_outs.transpose(1, 2).exp() - 1.0).transpose(1, 2) if self.hparams.use_f0 else 0) # (B, Tmax)
            sigma = self.sigma_predictor(torch.cat([hs,d_outs.float().unsqueeze(-1)],dim=-1), d_masks, spembs_=spembs_) if self.hparams.use_gaussian_upsampling else None # (B, Tmax)
            hs = self.length_regulator(hs, d_outs, alpha, sigma=sigma, is_inference=True)  # (B, Lmax, adim)
        else:
            d_outs = self.duration_predictor(hs, d_masks, spembs_=spembs_)  # (B, Tmax)
            hs = hs + (self.pitch_embed(ps.unsqueeze(-1).transpose(1, 2)).transpose(1, 2) if self.hparams.use_f0 else 0) # (B, Tmax)
            sigma = self.sigma_predictor(torch.cat([hs,ds.float().unsqueeze(-1)],dim=-1), d_masks, spembs_=spembs_) if self.hparams.use_gaussian_upsampling else None
            hs = self.length_regulator(hs, ds, sigma=sigma)  # (B, Lmax, adim)

        zs, _ = self.decoder(hs, h_masks, spembs_=spembs_)  # (B, Lmax, adim)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, Lmax, odim)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return before_outs, after_outs, d_outs, att, hs_random_choosed, style_embs, style_tokens, p_outs

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        durations: torch.Tensor,
        durations_lengths: torch.Tensor,
        pitch: torch.Tensor = None,
        pitch_lengths: torch.Tensor = None,
        spk_ids: torch.Tensor = None,
        style_ids: torch.Tensor = None,
        utt_mels: list = None,
        open_choosestl: bool = False,
        is_MI_step: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            durations (LongTensor): Batch of padded durations (B, Tmax).
            pitch (LongTensor): Batch of padded pitch (B, Tmax, 1).
            durations_lengths (LongTensor): Batch of duration lengths (B, Tmax).
            pitch_lengths (LongTensor): Batch of pitch lengths (B, Tmax).
            spembs (Tensor, optional): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value.

        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        speech = speech[:, : speech_lengths.max()]  # for data-parallel
        durations = durations[:, : durations_lengths.max()]  # for data-parallel
        pitch = pitch[:, : pitch_lengths.max()] if pitch is not None else None# for data-parallel

        batch_size = text.size(0)
        xs = text
        ilens = text_lengths
        ys, ds = speech, durations
        olens = speech_lengths

        # forward propagation
        if self.hparams.use_mutual_information and is_MI_step:
            hs_random_choosed, style_embs = self._forward(
                xs, ilens, ys, olens, ds, ps=pitch, spk_ids=spk_ids, style_ids=style_ids, utt_mels=utt_mels, is_inference=False, open_choosestl=open_choosestl, is_MI_step=is_MI_step
            )
            return hs_random_choosed, style_embs

        else:
            before_outs, after_outs, d_outs, att, hs_random_choosed, style_embs, _, p_outs = self._forward(
                xs, ilens, ys, olens, ds, ps=pitch, spk_ids=spk_ids, style_ids=style_ids, utt_mels=utt_mels, is_inference=False, open_choosestl=open_choosestl
            )

        # modifiy mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]

        # calculate loss
        if self.postnet is None:
            after_outs = None
        l1_loss, duration_loss, l2_loss, pitch_loss = self.criterion(
            after_outs, before_outs, d_outs, ys, ds, ilens, olens, p_outs, pitch.unsqueeze(-1) if pitch is not None else None#(B, Tmax, 1)
        )

        if self.hparams.use_ssim_loss:
            import pytorch_ssim
            ssim_loss = pytorch_ssim.SSIM()
            ssim_out = 3.0 * (2.0 - ssim_loss(before_outs, after_outs, ys, olens))
        if self.hparams.loss_type == "L1":
            loss = l1_loss + duration_loss + (ssim_out if self.hparams.use_ssim_loss else 0)
        if self.hparams.loss_type == "L2":
            loss = l2_loss + duration_loss + (ssim_out if self.hparams.use_ssim_loss else 0)        
        if self.hparams.loss_type == "L1_L2":
            loss = l1_loss + duration_loss + (ssim_out if self.hparams.use_ssim_loss else 0) + l2_loss
        if self.hparams.use_f0:
            loss += pitch_loss

        stats = dict(
            L1=l1_loss.item() if self.hparams.loss_type == "L1" else 0,
            L2=l2_loss.item() if self.hparams.loss_type == "L2" else 0,
            L1_L2=l1_loss.item() + l2_loss.item() if self.hparams.loss_type == "L1_L2" else 0,
            duration_loss=duration_loss.item(),
            pitch_loss=pitch_loss.item() if self.hparams.use_f0 else 0,
            loss=loss.item(),
            ssim_loss=ssim_out.item() if self.hparams.use_ssim_loss else 0,
        )

        # report extra information
        if self.use_scaled_pos_enc:
            stats.update(
                encoder_alpha=self.encoder.embed[-1].alpha.data.item(),
                decoder_alpha=self.decoder.embed[-1].alpha.data.item(),
            )

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight, after_outs, ys, olens, att, hs_random_choosed, style_embs

    def inference(
        self,
        text: torch.Tensor,
        speech: torch.Tensor = None,
        spk_ids: torch.Tensor = None,
        durations: torch.Tensor = None,
        alpha: float = 1.0,
        use_teacher_forcing: bool = False,
        utt_mels: list = None,
        style_embs_generated: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            spk_ids (Tensor, optional): Speaker embedding vector (spk_embed_dim,).
            durations (LongTensor, optional): Groundtruth of duration (T + 1,).
            alpha (float, optional): Alpha to control the speed.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.

        Returns:
            Tensor: Output sequence of features (L, odim).
            None: Dummy for compatibility.
            None: Dummy for compatibility.

        """
        x, y = text, speech
        spk_ids, d = spk_ids, durations

        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs, ys = x.unsqueeze(0), None
        # if y is not None:
        #     ys = y.unsqueeze(0)
        # if spk_ids is not None:
        #     spk_ids = spk_ids.unsqueeze(0)

        if use_teacher_forcing:
            # use groundtruth of duration, pitch, and energy
            ds = d.unsqueeze(0)
            _, outs, *_, att, _, _, style_embs_generated, _ = self._forward(
                xs, ilens, ys, ds=ds, spk_ids=spk_ids,
            )  # (1, L, odim)
        else:
            # inference
            _, outs, _, att, _, _, style_embs_generated, _ = self._forward(
                xs, ilens, ys, spk_ids=spk_ids, is_inference=True, alpha=alpha, utt_mels=utt_mels, style_embs_generated=style_embs_generated
            )  # (1, L, odim)

        return outs[0], None, None, att, _, _, style_embs_generated

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def _reset_parameters(
        self, init_type: str, init_enc_alpha: float, init_dec_alpha: float
    ):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)

    @staticmethod
    def _parse_batch(batch, hparams, utt_mels=None):
        text_padded, input_lengths, mel_padded, output_lengths, duration_padded, duration_lengths= batch[:6]
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        mel_padded = to_gpu(mel_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        duration_padded = to_gpu(duration_padded).long()
        duration_lengths = to_gpu(duration_lengths).long()

        idx = 6
        speaker_ids = None
        style_ids = None
        utt_mels = utt_mels
        pitchs = None

        if hparams.is_multi_speakers:
            speaker_ids = batch[idx]
            speaker_ids = to_gpu(speaker_ids).long()
            idx += 1

        if hparams.is_multi_styles:
            style_ids = batch[idx]
            style_ids = to_gpu(style_ids).long()#(B,)
            idx += 1

        if hparams.use_f0:
            pitchs = batch[idx]
            pitchs = to_gpu(pitchs).float()#(B,)
            idx += 1        

        return (text_padded, input_lengths, mel_padded, output_lengths, duration_padded,
                duration_lengths, pitchs, input_lengths, speaker_ids, style_ids, utt_mels)