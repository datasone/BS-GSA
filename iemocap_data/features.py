from enum import Enum
from typing import List, Dict

import librosa
import numpy as np
import torch
from python_speech_features import logfbank, fbank, sigproc
from tqdm import tqdm


class FeatureType(Enum):
    FBank = "fbank"
    LogFBank = "logfbank"
    MFCC = "mfcc"
    Spectrogram = "spectrogram"
    MelSpectrogram = "melspectrogram"

    def __str__(self):
        return self.value


class FeatureExtractor(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def get_features(self, features: List[FeatureType], x: List[torch.Tensor]) -> Dict[FeatureType, List[np.ndarray]]:
        return {feature: self.get_feature(feature, x) for feature in features}

    def get_feature(self, feature: FeatureType, x: List[torch.Tensor]) -> List[np.ndarray]:
        features = []
        for x_item in x:
            if x_item.shape[0] == 0:
                features.append(torch.empty((0, *self.feature_item_size[1:])))
            else:
                feature_x = self.__get_feature(feature, x_item.numpy())
                features.append(feature_x)
                self.feature_item_size = feature_x.shape

        return features

    def __get_feature(self, feature: FeatureType, x: np.ndarray):
        if feature == FeatureType.FBank:
            return self.get_fbank(x)
        if feature == FeatureType.LogFBank:
            return self.get_logfbank(x)
        if feature == FeatureType.MFCC:
            return self.get_mfcc(x, 26)
        if feature == FeatureType.Spectrogram:
            return self.get_spectrogram(x)
        if feature == FeatureType.MelSpectrogram:
            return self.get_melspectrogram(x)

    def get_logfbank(self, X):
        def _get_logfbank(x):
            out = logfbank(signal=x, samplerate=self.sample_rate, winlen=0.040, winstep=0.010, nfft=1024, highfreq=4000,
                           nfilt=40)
            return out

        X_features = np.apply_along_axis(_get_logfbank, 1, X)
        return X_features

    def get_mfcc(self, X, n_mfcc=13, use_delta=False, use_delta_delta=False):
        def _get_mfcc(x):
            mfcc_data = librosa.feature.mfcc(x, sr=self.sample_rate, n_mfcc=n_mfcc)

            delta = np.array([])
            delta_delta = np.array([])
            if use_delta:
                delta = librosa.feature.delta(mfcc_data)
            if use_delta_delta:
                delta_delta = librosa.feature.delta(mfcc_data, order=2)
            out = np.vstack([delta, mfcc_data]) if delta.size else mfcc_data
            out = np.vstack([delta_delta, out]) if delta_delta.size else out
            return out

        X_features = np.apply_along_axis(_get_mfcc, 1, X)
        return X_features

    def get_fbank(self, X):
        def _get_fbank(x):
            out, _ = fbank(signal=x, samplerate=self.sample_rate, winlen=0.040, winstep=0.010, nfft=1024)
            return out

        X_features = np.apply_along_axis(_get_fbank, 1, X)
        return X_features

    def get_melspectrogram(self, X):
        # feature shape (128, 1 + 40 * t)

        def _get_melspectrogram(x):
            mel = librosa.feature.melspectrogram(y=x, sr=self.sample_rate, n_fft=800, hop_length=400)[np.newaxis, :]
            out = np.log10(mel).squeeze()
            return out

        X_features = np.apply_along_axis(_get_melspectrogram, 1, X)
        return X_features

    def get_spectrogram(self, X):
        def _get_spectrogram(x):
            frames = sigproc.framesig(x, 640, 160)
            out = sigproc.logpowspec(frames, NFFT=3198)
            out = out.swapaxes(0, 1)
            return out[:][:400]

        X_features = np.apply_along_axis(_get_spectrogram, 1, X)
        return X_features
