import argparse
from enum import Enum, auto
from typing import List, Optional, Union

from iemocap_data.data_module import IEMOCAPSubset
from iemocap_data.features import FeatureType


class ConfigParseResult(Enum):
    OK = auto()
    SPLIT_RATE_ERR = auto()
    SPLIT_RATE_ERR_NO_TEST = auto()


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', "--data_dir")
        parser.add_argument('-s', "--seeds", nargs='*', type=int)
        parser.add_argument('-e', "--epochs", type=int, default=100)
        parser.add_argument('-b', "--batch_size", type=int, default=32)
        parser.add_argument('-l', "--learning_rate", type=float, default=1e-4)
        parser.add_argument('-dr', "--dropout_rate", type=float, default=0.5)
        parser.add_argument('-t', "--frame_length", type=float, default=2)
        parser.add_argument('-o', "--overlap", type=float, default=1)
        parser.add_argument("--features", type=FeatureType, choices=list(FeatureType), nargs='*',
                            default=[FeatureType.MelSpectrogram])
        parser.add_argument("--subset", type=IEMOCAPSubset, choices=list(IEMOCAPSubset))
        parser.add_argument("--gpus", type=int)
        parser.add_argument("--gpu_list", nargs='*', type=int)
        parser.add_argument("--aacnn_mode", nargs='*')
        parser.add_argument("--train_rate", type=float, default=0.8)
        parser.add_argument("--val_rate", type=float, default=0.2)
        parser.add_argument("--test_rate", type=float, default=0)
        parser.add_argument("--val_is_not_test", action='store_false', default=True)
        parser.add_argument("--disable_frame_pooling", action='store_false', default=True)
        parser.add_argument("--disable_blend_sample", action='store_false', default=True)
        parser.add_argument("--ee_only", action='store_true', default=False)
        parser.add_argument("--k_fold", type=int)
        parser.add_argument("--vtlp", action='store_true', default=False)
        parser.add_argument('--feature_file', type=str)

        args = parser.parse_args()

        # The directory of IEMOCAP data
        self.data_dir: str = args.data_dir
        # The seed list to run
        self.seeds: List[Optional[int]] = args.seeds
        # Training epochs
        self.epochs: int = args.epochs
        # Batch size
        self.batch_size: int = args.batch_size
        # Learning rate
        self.lr: float = args.learning_rate
        # Dropout rate
        self.dropout_rate: float = args.dropout_rate
        # The frame length
        self.t: float = args.frame_length
        # The frame overlap
        self.overlap: float = args.overlap
        # Features to be used
        self.features: List[FeatureType] = args.features
        # The subset of data to be used (i.e. `improvised`, `scripted` or both)
        self.data_subset: IEMOCAPSubset = args.subset
        # The number of gpu to be used, conflict with `gpu_list`
        self._gpus: int = args.gpus
        # The GPU ID list, conflict with `gpus`
        self._gpu_list: List[int] = args.gpu_list
        # Use the CNN structure like CA
        self._aacnn_mode = args.aacnn_mode
        # Data split: train set rate
        self.train_rate: float = args.train_rate
        # Data split: valid set rate
        self.val_rate: float = args.val_rate
        # Data split: test set rate
        self.test_rate: float = args.test_rate
        # Data split: merge valid and test
        self.val_is_test: bool = args.val_is_not_test
        # Frame pooling model
        self.frame_pooling: bool = args.disable_frame_pooling
        # BS augmentation
        self.blend_sample: bool = args.disable_blend_sample
        # Only apply augmentation to emotion and emotion (no neutral and emotion)
        self.ee_only: bool = args.ee_only
        # K-fold cross validation
        self.k_fold: Optional[int] = args.k_fold
        # Whether to use VTLP augmentation
        self.vtlp: bool = args.vtlp
        # Pre-extracted feature file, currently only used in replicating (k-fold mode)
        self.feature_file: Optional[str] = args.feature_file

    def validate(self) -> ConfigParseResult:
        if self.k_fold is None:
            if self.val_is_test:
                if self.train_rate + self.val_rate != 1:
                    return ConfigParseResult.SPLIT_RATE_ERR_NO_TEST
            else:
                if self.train_rate + self.val_rate + self.test_rate != 1:
                    return ConfigParseResult.SPLIT_RATE_ERR

        return ConfigParseResult.OK

    def aacnn_mode(self) -> bool:
        if self._aacnn_mode is None:
            return False
        else:
            return True

    def gpu_num_or_list(self) -> Union[int, List[int]]:
        if self._gpu_list is not None:
            return self._gpu_list
        else:
            return self._gpus
