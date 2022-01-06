import random

import torch
from tensorboardX import SummaryWriter

from .plotting_utils import (plot_alignment_to_numpy,
                                plot_gate_outputs_to_numpy,
                                plot_spectrogram_to_numpy)


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration, loss_dict=None, mi_lb=None):
        #self.add_scalar("train/training_loss", reduced_loss, iteration)
        self.add_scalar("grad_norm", grad_norm, iteration)
        self.add_scalar("learning_rate", learning_rate, iteration)
        self.add_scalar("training_time", duration, iteration)
        if loss_dict:
            for loss_name, loss in loss_dict.items():
                self.add_scalar("train/"+loss_name, loss, iteration)
        else:
            self.add_scalar("train/training_loss", reduced_loss, iteration)
        if mi_lb is not None:
            self.add_scalar("mutual_information", mi_lb, iteration)

    def log_validation(self, reduced_loss, model, iteration):
        self.add_scalar("validation_loss", reduced_loss, iteration)
        mel_targets = model.mel_targets.transpose(1, 2)
        gate_targets = model.gate_targets
        mel_outputs = model.mel_outputs.transpose(1, 2)
        gate_outputs = model.gate_outputs

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, gate_targets.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration)
