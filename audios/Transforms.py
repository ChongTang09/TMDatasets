import torch
import random
from torchvision import transforms

import torch.nn.functional as F
import torchaudio.transforms as T

class AudioTransform(torch.nn.Module):
    def __init__(self, resample_freq=16000, segment_len=16000*2, n_mfcc=13, transform_type='mfcc'):
        super().__init__()

        assert isinstance(segment_len, (int, float)), "segment_len must be a number"
        assert isinstance(resample_freq, (int, float)), "resample_freq must be a number"
        assert transform_type in ['mfcc'], 'Transform type must be either "mfcc"'

        # Resample to desired frequency
        self.resample_freq = resample_freq
        self.transform = None

        # MFCC transform
        if transform_type == 'mfcc':
            self.transform = T.MFCC(n_mfcc=n_mfcc, log_mels=True,
                    melkwargs={"n_fft": int(0.16*resample_freq), "hop_length": int(0.03*resample_freq), 
                               "center": False, 'n_mels': 13})
        
        self.segment_len = segment_len

    def forward(self, waveform, sampling_rate):
        resampler = T.Resample(orig_freq=sampling_rate, new_freq=self.resample_freq)
        # Resample the input
        resampled_wave = resampler(waveform)
        # check how many channels the waveform has;
        # for urbansound, it might have 2 due to torchaudio feature, we can take the average over the channels;
        if resampled_wave.shape[0] > 1:
            resampled_wave = torch.mean(resampled_wave, dim=0, keepdim=True) 

        # Pad or Trim waveform
        if resampled_wave.size(1) < self.segment_len:
            # Pad waveform
            resampled_wave = F.pad(input=resampled_wave, pad=(0, self.segment_len - resampled_wave.size(1)))
        else:
            # Trim waveform: randomly choose starting point
            max_audio_start = resampled_wave.size(1) - self.segment_len
            audio_start = random.randint(0, max_audio_start)
            resampled_wave = resampled_wave[:, audio_start : audio_start + self.segment_len]
            # resampled = resampled[:, :self.pad_len]
            
        # Compute features
        features = self.transform(resampled_wave)

        return features

class MFCCStandardizer:
    def __init__(self):
        self.mean_vals = None
        self.std_vals = None

    def fit(self, X):
        """
        Compute mean and standard deviation from the given data.

        :param X: N_samples x 13 x 29 matrix of MFCC features
        """
        self.mean_vals = np.mean(X, axis=(0,2), keepdims=True)
        self.std_vals = np.std(X, axis=(0,2), keepdims=True)

    def transform(self, X):
        """
        Standardize the given MFCC data using pre-computed mean and std values.

        :param X: N_samples x 13 x 29 matrix of MFCC features
        :return: Standardized MFCC matrix
        """
        if self.mean_vals is None or self.std_vals is None:
            raise ValueError("Must fit the standardizer before transforming data")

        return (X - self.mean_vals) / (self.std_vals + 1e-9)

    def inverse_transform(self, X_standardized):
        """
        Reverse the standardization for given standardized MFCC data.

        :param X_standardized: Standardized N_samples x 13 x 29 matrix of MFCC features
        :return: Original MFCC matrix
        """
        if self.mean_vals is None or self.std_vals is None:
            raise ValueError("Must fit the standardizer before inverse transforming data")

        return X_standardized * (self.std_vals + 1e-9) + self.mean_vals

    def fit_transform(self, X):
        """
        Compute mean and std from the data and then standardize it.

        :param X: N_samples x 13 x 29 matrix of MFCC features
        :return: Standardized MFCC matrix
        """
        self.fit(X)
        return self.transform(X)