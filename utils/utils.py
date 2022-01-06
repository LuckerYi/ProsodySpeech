import os
import logging
import time
import librosa
import numpy as np
import struct
import array
import torch
from scipy.io.wavfile import read as wavread
# import distoptim as dist


def print_rank(str):
    # time_stamp = datetime.datetime.now().strftime("%I %M %p %B %d %Y")
    # str = "{} | rank {}: {}".format(time.ctime(), dist.rank(), str)
    str = "{} : {}".format(time.ctime(), str)
    # print to log
    logging.info(str)
    # print to stdout
    print(str, flush=True)


class AverageMeter(object):
    """
    Will calculate running micro and macro averages for various
    (error/efficiency) rates. 
    """
    def __init__(self, metric_name):
        self.numerators, self.denominators = list(), list()
        self.metric_name = metric_name
    
    def add(self, top, bottom):
        # print_rank("{} : {}".format(self.metric_name, 
        #                             float(top) / bottom))
        self.numerators.append(top)
        self.denominators.append(bottom)
    
    def get_macro_average(self):
        scores = [float(self.numerators[i]) / self.denominators[i] \
                            for i in range(len(self.denominators))]
        return self.get_average(scores)
    
    def get_micro_average(self):
        return float(sum(self.numerators)) / sum(self.denominators)
    
    def get_average(self, l):
        # accepts a list and returns average
        return sum(l) / float(len(l))
    
    def reset(self):
        self.numerators, self.denominators = list(), list()
    
    def display_results(self):
        print_rank("{} Macro average: {}".format(self.metric_name, 
                                                self.get_macro_average()))
        print_rank("{} Micro average: {}".format(self.metric_name, 
                                                self.get_micro_average()))


# define function for plot prob and att_ws
def _plot_and_save(array, figname, figsize=(16, 4), dpi=150):
    import matplotlib.pyplot as plt
    shape = array.shape
    if len(shape) == 1:
        # for eos probability
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(array)
        plt.xlabel("Frame")
        plt.ylabel("Probability")
        plt.ylim([0, 1])
    elif len(shape) == 2:
        # for tacotron 2 attention weights, whose shape is (out_length, in_length)
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(array, aspect="auto", origin='lower',
                   interpolation='none')
        plt.xlabel("Input")
        plt.ylabel("Output")
    elif len(shape) == 4:
        # for transformer attention weights, whose shape is (#leyers, #heads, out_length, in_length)
        plt.figure(figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi)
        for idx1, xs in enumerate(array):
            for idx2, x in enumerate(xs, start=1):
                plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
                plt.imshow(x, aspect="auto", origin='lower',
                   interpolation='none')
                plt.xlabel("Input")
                plt.ylabel("Output")
    else:
        raise NotImplementedError("Support only from 1D to 4D array.")
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(figname)):
        # NOTE: exist_ok = True is needed for parallel process decoding
        os.makedirs(os.path.dirname(figname), exist_ok=True)
    plt.savefig(figname)
    plt.close()


# define function to calculate focus rate (see section 3.3 in https://arxiv.org/abs/1905.09263)
def _calculate_focus_rete(att_ws):
    if att_ws is None:
        # fastspeech case -> None
        return 1.0
    elif len(att_ws.shape) == 2:
        # tacotron 2 case -> (L, T)
        return float(att_ws.max(dim=-1)[0].mean())
    elif len(att_ws.shape) == 4:
        # transformer case -> (#layers, #heads, L, T)
        return float(att_ws.max(dim=-1)[0].mean(dim=-1).max())
    else:
        raise ValueError("att_ws should be 2 or 4 dimensional tensor.")


# define function to convert attention to duration
def _convert_att_to_duration(att_ws):
    if len(att_ws.shape) == 2:
        # tacotron 2 case -> (L, T)
        pass
    elif len(att_ws.shape) == 4:
        # transformer case -> (#layers, #heads, L, T)
        # get the most diagonal head according to focus rate
        att_ws = torch.cat([att_w for att_w in att_ws], dim=0)  # (#heads * #layers, L, T)
        diagonal_scores = att_ws.max(dim=-1)[0].mean(dim=-1)  # (#heads * #layers,)
        diagonal_head_idx = diagonal_scores.argmax()
        att_ws = att_ws[diagonal_head_idx]  # (L, T)
    else:
        raise ValueError("att_ws should be 2 or 4 dimensional tensor.")
    # calculate duration from 2d attention weight
    durations = torch.stack([att_ws.argmax(-1).eq(i).sum() for i in range(att_ws.shape[1])])
    return durations.view(-1, 1).float()


def get_checkpoint_path(output_directory):
    # os.makedirs(output_directory) in prepare_directories_and_logger
    file_list = os.listdir(output_directory)
    file_list.sort(key=lambda fn: os.path.getmtime(os.path.join(output_directory, fn))
                    if fn.startswith("checkpoint_") else 0)
    if file_list and file_list[-1].startswith("checkpoint_"):
        return os.path.join(output_directory, file_list[-1])
    else:
        return None


def learning_rate_decay(step, hp):
    if hp.learning_rate_decay_scheme == "noam":
        ret = 5000.0 * hp.adim**-0.5 * min((step + 1) * hp.warmup_steps**-1.5, (step + 1)**-0.5)
        optimizer_correction = 0.002
        lr = ret * optimizer_correction * hp.initial_learning_rate
    else:
        lr = hp.initial_learning_rate
        step += 1.
        if step > hp.warmup_steps:
            # lr *= (hp.decay_rate ** ((step - hp.warmup_steps) / (hp.decay_end-hp.warmup_steps)))
            lr *= (hp.decay_rate ** ((step - hp.warmup_steps) / hp.decay_steps))
            lr = max(lr, hp.final_learning_rate)

    return lr

def load_wav_to_torch(full_path):
    #data, sampling_rate = librosa.core.load(full_path, sr=None)
    sampling_rate, data = wavread(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def save_htk_data(feature, file_name):
    (nframes, vec_size) = np.shape(feature)
    byte = 4 * vec_size
    htktype = 9
    frameshift = 50000
    hdr = struct.pack("<2l2h", nframes, frameshift, byte, htktype)
    with open(file_name, 'wb') as f:
        f.write(hdr)
        sdata = np.reshape(feature, [-1])
        s = array.array('f', sdata)
        s.tofile(f)



def _convert_duration_to_attn(dur, max_len=None, dtype=torch.float):
    """generate alignment matrix according to duration of phoneme.

    If `lengths` has shape `[B, T_in]` the resulting tensor `alignment` has
    dtype `dtype` and shape `[B, T_in, T_out]`, with

    ```
    lengths = torch.cumsum(dur, -1)
    alignment[i_1, i_2, j] = (lengths[i_1, i_2-1] <= j < lengths[i_1, i_2])
    ```

    Examples:

    ```python
    gen_alignment([[1, 2], [2, 0]])  # [[[1, 0, 0],
                                     #   [0, 1, 1]],
                                     #  [[1, 1, 0],
                                     #   [0, 0, 0]]]
    ```

    Args:
    dur: integer tensor, all its values <= maxlen. [B, T_in]
    maxlen: scalar integer tensor, size of last dimension of returned tensor.
        Default is the maximum value in `lengths`.
    dtype: output type of the resulting tensor.

    Returns:
    A mask tensor of shape `lengths.shape + (maxlen,)`, cast to specified dtype.

    Raises:
    ValueError: if `maxlen` is not a scalar.
    """
    assert len(dur.shape) == 2
    lengths = torch.cumsum(dur, -1)
    if max_len is None:
        max_len = torch.max(lengths).int()
    row_vec = torch.arange(max_len, device=dur.device).expand([lengths.shape[0], lengths.shape[1], -1])
    mask1 = (row_vec < lengths.unsqueeze(-1)).int()
    mask2 = torch.cat([mask1.new_zeros([mask1.shape[0], 1, max_len]), mask1[:, :-1, :]], 1)
    alignment = mask1 - mask2
    if dtype is not None:
        alignment = alignment.type(dtype)
    return alignment