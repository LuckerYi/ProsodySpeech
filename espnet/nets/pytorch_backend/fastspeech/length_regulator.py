#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Length regulator related modules."""

import logging

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list
import torch.nn.functional as F

class LengthRegulator(torch.nn.Module):

    def __init__(self):
        """Initilize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.

        """
        super(LengthRegulator, self).__init__()
    
    def gaussian_function_with_softplus(self, x, mu, sigma):

        sigma = F.softplus(sigma)

        # print('sigma.shape=',sigma.shape)
        # print('sigma[:,0]=',sigma[:,0])

        result = torch.exp(-((x - mu)**2)*0.5*sigma**2)*sigma

        return result

    def forward(self, xs, ds, alpha=1.0, sigma=None, is_inference=False):
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
            ds (LongTensor): Batch of durations of each frame (B, T).
            alpha (float, optional): Alpha value to control speed of speech.
            eg.
            ds = tensor([[3, 4, 5],
                        [4, 5, 6]])
            sigma = tensor([[1, 2, 3],
                            [1, 2, 3]])
        Returns:
            Tensor: replicated input tensor based on durations (B, T*, D).

        """
        assert alpha > 0
        if alpha != 1.0:
            ds = torch.round(ds.float() * alpha).long()
        phone_feats = xs
        durations = ds
        length = torch.cumsum(durations, dim=-1)
        target_length = torch.max(length + 1)
        cum_length = torch.cumsum(F.one_hot(length, target_length),dim=-1)
        shifted_length = F.pad(length, pad=(1, 0), mode = 'constant', value = 0)[:, 0:-1]
        shifted_cum_length = torch.cumsum(F.one_hot(shifted_length, target_length),dim=-1)
        diff = (shifted_cum_length - cum_length)[:, :, :target_length - 1]
        diff = diff.transpose(1,2).float()

        if sigma is not None:
            # print('durations.shape=',durations.shape)
            # print('durations=',durations)            
            # if durations.lt(0).int().sum() > 0:
            #     print('durations.lt(0).int().sum() > 0')
            #     raise          
            center_durations = durations.float()/2 + shifted_length#shape = torch.Size([2, 3])
            # print('center_durations.shape=',center_durations.shape)
            # print('center_durations=',center_durations)
            # change arange to F.onehot and use cumsum
            t_range = 1 + torch.arange(target_length - 1).to(target_length.device)#tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
            # print('t_range.shape=',t_range.shape)
            # print('t_range=',t_range)
            t_range_expand = t_range.unsqueeze(1).expand_as(diff)#shape = torch.Size([2, 15, 3])
            # print('t_range_expand.shape=',t_range_expand.shape)
            # print('t_range_expand=',t_range_expand)
            center_durations_expand = center_durations.unsqueeze(1).expand_as(t_range_expand)#shape = torch.Size([2, 15, 3])
            # print('center_durations_expand.shape=',center_durations_expand.shape)
            # print('center_durations_expand=',center_durations_expand)
            sigma_expand = sigma.unsqueeze(1).expand_as(t_range_expand)
            # print('sigma_expand.shape=',sigma_expand.shape)
            # print('sigma_expand=',sigma_expand)
            numerator = self.gaussian_function_with_softplus(t_range_expand.clone(),center_durations_expand.clone(),sigma_expand.clone())
            # print('numerator.shape=',numerator.shape)
            # print('numerator=',numerator)

            numerator_div_denominator_expand = F.normalize(numerator,p=1,dim=-1)

            # denominator_expand = torch.sum(numerator,dim=-1).unsqueeze(-1).expand_as(diff)
            # # print('denominator_expand.shape=',denominator_expand.shape)
            # # print('denominator_expand=',denominator_expand)
            # numerator_div_denominator_expand = numerator/denominator_expand.clamp_min(1e-12)# in case of div zeros
            # print('numerator_div_denominator_expand.shape=',numerator_div_denominator_expand.shape)
            # print('numerator_div_denominator_expand=',numerator_div_denominator_expand)
            diff_expand = torch.sum(diff,dim=-1).unsqueeze(-1).expand_as(diff)
            # print('diff_expand.shape=',diff_expand.shape)
            # print('diff_expand=',diff_expand)
            diff = numerator_div_denominator_expand*diff_expand
            # print('diff.shape=',diff.shape)
            # print('diff=',diff)

        final = torch.matmul(diff,phone_feats)
        return final

# class LengthRegulator(torch.nn.Module):
#     """Length regulator module for feed-forward Transformer.

#     This is a module of length regulator described in
#     `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
#     The length regulator expands char or
#     phoneme-level embedding features to frame-level by repeating each
#     feature based on the corresponding predicted durations.

#     .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
#         https://arxiv.org/pdf/1905.09263.pdf

#     """

#     def __init__(self, pad_value=0.0):
#         """Initilize length regulator module.

#         Args:
#             pad_value (float, optional): Value used for padding.

#         """
#         super(LengthRegulator, self).__init__()
#         self.pad_value = pad_value

#     def forward(self, xs, ds, ilens, alpha=1.0):
#         """Calculate forward propagation.

#         Args:
#             xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
#             ds (LongTensor): Batch of durations of each frame (B, T).
#             ilens (LongTensor): Batch of input lengths (B,).
#             alpha (float, optional): Alpha value to control speed of speech.

#         Returns:
#             Tensor: replicated input tensor based on durations (B, T*, D).

#         """
#         assert alpha > 0
#         if alpha != 1.0:
#             ds = torch.round(ds.float() * alpha).long()
#         xs = [x[:ilen] for x, ilen in zip(xs, ilens)]# zip in Batch Size
#         ds = [d[:ilen] for d, ilen in zip(ds, ilens)]
#         xs = [self._repeat_one_sequence(x, d) for x, d in zip(xs, ds)]

#         return pad_list(xs, self.pad_value)

#     def _repeat_one_sequence(self, x, d):
#         """Repeat each frame according to duration.

#         Examples:
#             >>> x = torch.tensor([[1], [2], [3]])
#             tensor([[1],
#                     [2],
#                     [3]])
#             >>> d = torch.tensor([1, 2, 3])
#             tensor([1, 2, 3])
#             >>> self._repeat_one_sequence(x, d)
#             tensor([[1],
#                     [2],
#                     [2],
#                     [3],
#                     [3],
#                     [3]])

#         """
#         if d.sum() == 0:
#             logging.warning("all of the predicted durations are 0. fill 0 with 1.")
#             d = d.fill_(1)
#         return torch.cat(
#             [x_.repeat(int(d_), 1) for x_, d_ in zip(x, d) if d_ != 0], dim=0
#         )
