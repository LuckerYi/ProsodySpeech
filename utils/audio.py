import os
import re

import librosa
import numpy as np
import scipy
import soundfile
import tensorflow as tf
from scipy import signal


'''
hparams = tf.contrib.training.HParams(
    # Audio:
    num_mels=80,
    num_freq=1025,
    min_mel_freq=0,
    max_mel_freq=8000,
    sample_rate=16000,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    preemphasize=0.97,
    min_level_db=-100,
    ref_level_db=0,  # suggest use 20 for griffin-lim and 0 for wavenet
    max_abs_value=1,
    symmetric_specs=False,  # if true, suggest use 4 as max_abs_value

    # Eval:
    griffin_lim_iters=60,
    power=1.5,  # Power to raise magnitudes to prior to Griffin-Lim
)
'''

class Audio():
    def __init__(self, hparams):
        self.hparams = hparams

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.hparams.sample_rate)[0]


    def save_wav(self, wav, path):
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        scipy.io.wavfile.write(path, self.hparams.sample_rate, wav.astype(np.int16))


    def spectrogram(self, y, clip_norm=True):
        D = self._stft(y)
        S = self._amp_to_db(np.abs(D)) - self.hparams.ref_level_db
        if clip_norm:
            S = self._normalize(S)
        return S


    def inv_spectrogram(self, spectrogram):
        '''Converts spectrogram to waveform using librosa'''
        S = self._db_to_amp(self._denormalize(spectrogram) +
                    self.hparams.ref_level_db)  # Convert back to linear
        return self._griffin_lim(S**self.hparams.power)  # Reconstruct phase


    def inv_spectrogram_tensorflow(self, spectrogram):
        '''Builds computational graph to convert spectrogram to waveform using TensorFlow.

        Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
        inv_preemphasis on the output after running the graph.
        '''
        S = self._db_to_amp_tensorflow(
            self._denormalize_tensorflow(spectrogram) + self.hparams.ref_level_db)
        return self._griffin_lim_tensorflow(tf.pow(S, self.hparams.power))


    def melspectrogram(self, y, clip_norm=True):
        D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.hparams.ref_level_db
        if clip_norm:
            S = self._normalize(S)
        return S


    def inv_mel_spectrogram(self, mel_spectrogram):
        S = self._mel_to_linear(self._db_to_amp(
            self._denormalize(mel_spectrogram) + self.hparams.ref_level_db))  # Convert back to linear
        return self._griffin_lim(S**self.hparams.power)  # Reconstruct phase


    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        window_length = int(self.hparams.sample_rate * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(threshold_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x:x + window_length]) < threshold:
                return x + hop_length
        return len(wav)


    def _griffin_lim(self, S):
        '''librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        '''
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(self.hparams.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y


    def _griffin_lim_tensorflow(self, S):
        '''TensorFlow implementation of Griffin-Lim
        Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
        '''
        with tf.variable_scope('griffinlim'):
            # TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
            S = tf.expand_dims(S, 0)
            S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
            y = self._istft_tensorflow(S_complex)
            for i in range(self.hparams.griffin_lim_iters):
                est = self._stft_tensorflow(y)
                angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
                y = self._istft_tensorflow(S_complex * angles)
            return tf.squeeze(y, 0)


    def _stft(self, y):
        n_fft, hop_length, win_length = self._stft_parameters()
        return librosa.stft(
            y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


    def _istft(self, y):
        _, hop_length, win_length = self._stft_parameters()
        return librosa.istft(y, hop_length=hop_length, win_length=win_length)


    def _stft_tensorflow(self, signals):
        n_fft, hop_length, win_length = self._stft_parameters()
        return tf.contrib.signal.stft(
            signals, win_length, hop_length, n_fft, pad_end=False)


    def _istft_tensorflow(self, stfts):
        n_fft, hop_length, win_length = self._stft_parameters()
        return tf.contrib.signal.inverse_stft(stfts, win_length, hop_length, n_fft)


    def _stft_parameters(self):
        n_fft = (self.hparams.num_freq - 1) * 2
        hop_length = int(self.hparams.frame_shift_ms / 1000 * self.hparams.sample_rate)
        win_length = int(self.hparams.frame_length_ms / 1000 * self.hparams.sample_rate)
        return n_fft, hop_length, win_length


    def _linear_to_mel(self, spectrogram):
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)


    def _mel_to_linear(self, mel_spectrogram):
        _inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


    def _build_mel_basis(self):
        n_fft = (self.hparams.num_freq - 1) * 2
        return librosa.filters.mel(
            self.hparams.sample_rate,
            n_fft,
            n_mels=self.hparams.num_mels,
            fmin=self.hparams.min_mel_freq,
            fmax=self.hparams.max_mel_freq)


    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))


    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)


    def _db_to_amp_tensorflow(self, x):
        return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


    def _normalize(self, S):
        if self.hparams.symmetric_specs:
            return np.clip(
                (2 * self.hparams.max_abs_value) * ((S - self.hparams.min_level_db) / (-self.hparams.min_level_db)) - self.hparams.max_abs_value,
                -self.hparams.max_abs_value, self.hparams.max_abs_value)
        else:
            return np.clip(self.hparams.max_abs_value * ((S - self.hparams.min_level_db) / (-self.hparams.min_level_db)), 0, self.hparams.max_abs_value)


    def _denormalize(self, S):
        if self.hparams.symmetric_specs:
            return (((np.clip(S, -self.hparams.max_abs_value,
                            self.hparams.max_abs_value) + self.hparams.max_abs_value) * -self.hparams.min_level_db / (
                            2 * self.hparams.max_abs_value))
                    + self.hparams.min_level_db)
        else:
            return ((np.clip(S, 0,
                            self.hparams.max_abs_value) * -self.hparams.min_level_db / self.hparams.max_abs_value) + self.hparams.min_level_db)


    def _denormalize_tensorflow(self, S):
        return (tf.clip_by_value(S, 0, 1) *
                -self.hparams.min_level_db) + self.hparams.min_level_db


    def _preemphasize(self, x):
        return signal.lfilter([1, -self.hparams.preemphasize], [1], x)


    def inv_preemphasize(self, x):
        if self.hparams.preemphasize is not None:
            x = signal.lfilter([1], [1, -self.hparams.preemphasize], x)
        return x


    def _magnitude_spectrogram(self, audio, clip_norm):
        preemp_audio = self._preemphasize(audio)
        mel_spec = self.melspectrogram(preemp_audio, clip_norm)
        linear_spec = self.spectrogram(preemp_audio, clip_norm)

        return mel_spec.T, linear_spec.T


    def _energy_spectrogram(self, audio):
        preemp_audio = self._preemphasize(audio)
        linear_spec = np.abs(self._stft(preemp_audio))**2
        mel_spec = self._linear_to_mel(linear_spec)

        return mel_spec.T, linear_spec.T


    def _extract_min_max(self, wav_path, spectrogram, post_fn=lambda x: x):
        num_mels = self.hparams.num_mels
        num_linears = self.hparams.num_freq

        wavs = []
        for root, dirs, files in os.walk(wav_path):
            for f in files:
                if re.match(r'.+\.wav', f):
                    wavs.append(os.path.join(root, f))

        num_wavs = len(wavs)

        mel_mins_per_wave = np.zeros((num_wavs, num_mels))
        mel_maxs_per_wave = np.zeros((num_wavs, num_mels))
        linear_mins_per_wave = np.zeros((num_wavs, num_linears))
        linear_maxs_per_wave = np.zeros((num_wavs, num_linears))

        for i, wav in enumerate(post_fn(wavs)):
            audio, sr = soundfile.read(wav)
            if spectrogram == 'magnitude':
                mel, linear = self._magnitude_spectrogram(audio, clip_norm=False)
            elif spectrogram == 'energy':
                mel, linear = self._energy_spectrogram(audio)
            else:
                raise Exception("only magnitude or energy is supported")

            mel_mins_per_wave[i, ] = np.amin(mel, axis=0)
            mel_maxs_per_wave[i, ] = np.amax(mel, axis=0)
            linear_mins_per_wave[i, ] = np.amin(linear, axis=0)
            linear_maxs_per_wave[i, ] = np.amax(linear, axis=0)

        mel_mins = np.reshape(np.amin(mel_mins_per_wave, axis=0), (1, num_mels))
        mel_maxs = np.reshape(np.amax(mel_maxs_per_wave, axis=0), (1, num_mels))
        linear_mins = np.reshape(
            np.amin(linear_mins_per_wave, axis=0), (1, num_linears))
        linear_maxs = np.reshape(
            np.amax(linear_maxs_per_wave, axis=0), (1, num_linears))

        min_max = {
            "mel_min": mel_mins,
            "mel_max": mel_maxs,
            "linear_min": linear_mins,
            "linear_max": linear_maxs
        }

        return min_max


    def _normalize_min_max(self, spec, maxs, mins, max_value=1.0, min_value=0.0):
        spec_dim = len(spec.T)
        num_frame = len(spec)

        max_min = maxs - mins
        max_min = np.reshape(max_min, (1, spec_dim))
        max_min[max_min <= 0.0] = 1.0

        target_max_min = np.zeros((1, spec_dim))
        target_max_min.fill(max_value - min_value)
        target_max_min[max_min <= 0.0] = 1.0

        spec_min = np.tile(mins, (num_frame, 1))
        target_min = np.tile(min_value, (num_frame, spec_dim))
        spec_range = np.tile(max_min, (num_frame, 1))
        norm_spec = np.tile(target_max_min, (num_frame, 1)) / spec_range
        norm_spec = norm_spec * (spec - spec_min) + target_min
        return norm_spec


    def _denormalize_min_max(self, spec, maxs, mins, max_value=1.0, min_value=0.0):
        spec_dim = len(spec.T)
        num_frame = len(spec)

        max_min = maxs - mins
        max_min = np.reshape(max_min, (1, spec_dim))
        max_min[max_min <= 0.0] = 1.0

        target_max_min = np.zeros((1, spec_dim))
        target_max_min.fill(max_value - min_value)
        target_max_min[max_min <= 0.0] = 1.0

        spec_min = np.tile(mins, (num_frame, 1))
        target_min = np.tile(min_value, (num_frame, spec_dim))
        spec_range = np.tile(max_min, (num_frame, 1))
        denorm_spec = spec_range / np.tile(target_max_min, (num_frame, 1))
        denorm_spec = denorm_spec * (spec - target_min) + spec_min
        return denorm_spec
