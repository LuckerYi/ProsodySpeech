#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Layer normalization module."""

import torch


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.

    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, hparams, dim=-1, elementwise_affine=False):
        """Construct an LayerNorm object."""
        is_spk_layer_norm = hparams.is_spk_layer_norm
        if is_spk_layer_norm:
            super(LayerNorm, self).__init__(nout, eps=1e-12, elementwise_affine=elementwise_affine)
            if not elementwise_affine:
                self.w = torch.nn.Linear(hparams.spk_embed_dim, nout)
                self.b = torch.nn.Linear(hparams.spk_embed_dim, nout)
        else:
            super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim
        self.is_spk_layer_norm = is_spk_layer_norm

    def forward(self, x, spembs_=None):
        """Apply layer normalization.

        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.is_spk_layer_norm and spembs_ is not None:
            if self.dim == -1:
                # print('super(LayerNorm, self).forward(x).shape=',super(LayerNorm, self).forward(x).shape)
                # print('self.w(spembs_).shape=',self.w(spembs_).unsqueeze(1).shape)
                # print('self.b(spembs_).shape=',self.b(spembs_).unsqueeze(1).shape)
                return super(LayerNorm, self).forward(x) * self.w(spembs_).unsqueeze(1) + self.b(spembs_).unsqueeze(1)
            # print('super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1).shape=',super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1).shape)
            return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1) * self.w(spembs_).unsqueeze(-1) + self.b(spembs_).unsqueeze(-1)
        else:
            if self.dim == -1:
                return super(LayerNorm, self).forward(x)
            return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)         
