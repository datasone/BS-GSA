from functools import partial
import glob
import itertools
import os
import pickle
import random
import shutil
from enum import Enum
from typing import Optional, List, Tuple, Dict, Union

import librosa
import pytorch_lightning as pl
import soundfile
import torch
from nlpaug.augmenter.audio import VtlpAug
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import numpy as np

from .features import FeatureType, FeatureExtractor

LABEL = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
}

IEMOCAP_LABEL = {
    '01': 'neutral',
    # '02': 'frustration',
    # '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    # '06': 'fearful',
    '07': 'happy',  # excitement->happy
    # '08': 'surprised'
}


class IEMOCAPSubset(Enum):
    Improvised = "impro"
    Script = "script"
    All = "all"

    def __str__(self):
        return self.value


def process_iemocap(data_dir):
    if not os.path.isdir(data_dir + '/generated_data'):
        os.mkdir(data_dir + '/generated_data')
    csv_path = data_dir + '/generated_data/IEMOCAP.csv'
    if os.path.isfile(csv_path):
        return

    wavs = glob.glob('{}/wav/*.wav'.format(data_dir))
    transcripts = glob.glob('{}/transcript/*.txt'.format(data_dir))
    write_list = []
    print("Processing dataset info...")
    for wav in tqdm(wavs):
        wav_name = os.path.basename(wav)
        wav_name_split = wav_name.split('.')[0].split('-')
        if wav_name_split[2] not in IEMOCAP_LABEL:
            continue
        if 'script' in wav_name:
            txt_name = "{}_{}_{}.txt".format(wav_name_split[0], wav_name_split[1], wav_name_split[-1].split('_')[0])
        else:
            txt_name = "{}_{}.txt".format(wav_name_split[0], wav_name_split[1])
        transcript_filename = None
        for transcript in transcripts:
            if os.path.basename(transcript) == txt_name:
                transcript_filename = transcript
                break
        if transcript_filename is not None:
            with open(transcript_filename) as transcript_file:
                transcript_content = transcript_file.readlines()
                find = False
                for line in transcript_content:
                    if line.split(' ')[0] == wav_name_split[0] + '_' + wav_name_split[1] + '_' + wav_name_split[-1]:
                        write_list.append(
                            (line.split(' ')[0], line.split(':')[-1].replace('\n', ''), wav_name, wav_name_split[2]))
                        find = True
                        break
                if not find:
                    print('Cannot find :' + wav_name)
        else:
            print('Cannot find :' + txt_name)
    with open(csv_path, 'w') as f:
        f.write("\n".join(["\t".join(s) for s in write_list]))


def aug_step(wav_name: str, aug: VtlpAug, target_dir: str):
    wav, _ = librosa.load(wav_name, sr=16000)
    wav_aug = aug.augment(wav)
    wav_name = os.path.basename(wav_name)
    soundfile.write(target_dir + wav_name + '.5', wav_aug[0], 16000, format='WAV')


def audio_augment(data_dir, vtlp):
    target_dir = "{}/augwav/".format(data_dir)
    wav_list = glob.glob('{}/wav/*.wav'.format(data_dir))

    if not (os.path.isdir(target_dir) and len(glob.glob("{}/*.wav".format(target_dir))) > 0):
        try:
            shutil.rmtree(target_dir)
        except Exception as e:
            print(e)
        os.mkdir(target_dir)
        os.system("cp {}/wav/*.wav {}".format(data_dir, target_dir))

    if vtlp:
        aug = VtlpAug(16000, zone=(0.0, 1.0), coverage=1, fhi=4800, factor=(0.8, 1.2))

        print("Processing audio augmentation...")
        process_map(partial(aug_step, aug=aug, target_dir=target_dir), wav_list, max_workers=64)


def data_process(files_paths: Dict[str, Dict[str, str]], sample_rate: int, t, overlap, val_overlap, subset: IEMOCAPSubset, vtlp: bool) -> \
        Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
    data_x = {}
    data_y = {}

    for name in tqdm(files_paths.keys()):
        normal_filename = files_paths[name]["normal"]
        label = str(os.path.basename(normal_filename).split('-')[2])
        if label not in IEMOCAP_LABEL:
            continue
        if subset != IEMOCAPSubset.All and (str(subset) not in normal_filename):
            continue
        label = IEMOCAP_LABEL[label]

        item_x = {}
        item_y = {}

        for type in ["train", "val"]:
            for part in ["normal", "additional"]:
                if not vtlp and part == "additional":
                    continue
                wav_filename = files_paths[name][part]
                wav_data, _ = librosa.load(wav_filename, sr=sample_rate)

                x = []
                y = []
                index = 0
                while index + t * sample_rate < len(wav_data):
                    x.append(wav_data[int(index):int(index + t * sample_rate)])
                    y.append(LABEL[label])
                    index += int((t - overlap) * sample_rate)

                x = np.array(x)
                y = np.array(y)
                x = torch.tensor(x, device='cpu')
                y = torch.tensor(y, device='cpu').long()

                if part not in item_x.keys():
                    item_x[part] = {}
                if part not in item_y.keys():
                    item_y[part] = {}
                item_x[part][type] = x
                item_y[part][type] = y

        if item_x["normal"]["train"].nelement() != 0 and item_y["normal"]["train"].nelement() != 0:
            data_x[name] = item_x
            data_y[name] = item_y

    return data_x, data_y


def data_process_additional_only(files_paths: Dict[str, Dict[str, str]], sample_rate: int, t, overlap, val_overlap, subset: IEMOCAPSubset, target_dir: str) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
    with open("{}/data_x_{}_{}_{}".format(target_dir, t, overlap, val_overlap), "rb") as f:
        data_x = pickle.load(f)
    with open("{}/data_y_{}_{}_{}".format(target_dir, t, overlap, val_overlap), "rb") as f:
        data_y = pickle.load(f)

    for type in ["train", "val"]:
        for name in tqdm(data_x.keys()):
            normal_filename = files_paths[name]["normal"]
            label = str(os.path.basename(normal_filename).split('-')[2])
            label = IEMOCAP_LABEL[label]

            wav_filename = files_paths[name]["additional"]
            wav_data, _ = librosa.load(wav_filename, sr=sample_rate)

            x = []
            y = []
            index = 0
            while index + t * sample_rate < len(wav_data):
                x.append(wav_data[int(index):int(index + t * sample_rate)])
                y.append(LABEL[label])
                if type == "train":
                    index += int((t - overlap) * sample_rate)
                else:
                    index += int((t - val_overlap) * sample_rate)

            x = np.array(x)
            y = np.array(y)
            x = torch.tensor(x, device='cpu')
            y = torch.tensor(y, device='cpu').long()

            data_x[name]["additional"][type] = x
            data_y[name]["additional"][type] = y

    return data_x, data_y


def generate_data(data_dir: str, sample_rate: int, t: float, overlap: float, val_overlap: float, subset: IEMOCAPSubset, vtlp: bool):
    exists_data = False
    target_dir = "{}/generated_data/dataset_data".format(data_dir)
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    filenames = ["data_x_{}_{}_{}".format(t, overlap, val_overlap), "data_y_{}_{}_{}".format(t, overlap, val_overlap), "names_{}_{}_{}".format(t, overlap, val_overlap)]
    if os.path.isdir(target_dir) and all([os.path.isfile("{}/{}".format(target_dir, f)) for f in filenames]):
        exists_data = True

    files_paths = {}

    with open(data_dir + '/generated_data/IEMOCAP.csv', 'r') as f:
        for line in f.readlines():
            line_segments = line.split('\t')
            name = line_segments[0]
            wav_path = "{}/augwav/{}".format(data_dir, line_segments[2])
            train_additional_wav_path = "{}/augwav/{}.{}".format(data_dir, line_segments[2], 5)  # additional is augmented

            files_paths[name] = {"normal": wav_path, "additional": train_additional_wav_path}

    print("Generating data...")

    if exists_data and vtlp:
        data_x, data_y = data_process_additional_only(files_paths, sample_rate, t, overlap, val_overlap, subset, target_dir)
    else:
        data_x, data_y = data_process(files_paths, sample_rate, t, overlap, val_overlap, subset, vtlp)

    #with open("{}/data_x_{}_{}_{}".format(target_dir, t, overlap, val_overlap), "wb") as f:
    #    pickle.dump(data_x, f)
    #with open("{}/data_y_{}_{}_{}".format(target_dir, t, overlap, val_overlap), "wb") as f:
    #    pickle.dump(data_y, f)

    #with open('{}/names_{}_{}_{}'.format(target_dir, t, overlap, val_overlap), "wb") as f:
    #    pickle.dump(list(data_x.keys()), f)

    return data_x, data_y


def generate_features(data_dir: str, features: List[FeatureType], sample_rate: int, t: float, overlap: float, data_x, data_y, vtlp: bool):
    # target_dir = "{}/generated_data/dataset_features".format(data_dir)
    # if not os.path.isdir(target_dir):
    #     os.mkdir(target_dir)
    # file_paths = ["{}/data_x_{}_{}_feature_{}".format(target_dir, t, overlap, feature) for feature in features]
    # if os.path.isdir(target_dir) and all([os.path.isfile(f) for f in file_paths]):
    #     return

    extractor = FeatureExtractor(sample_rate)

    print("Generating data features...")
    # with open("{}/generated_data/dataset_data/data_x_{}_{}".format(data_dir, t, overlap), "rb") as f:
    for feature in features:
        data_feature = {}
        for name in tqdm(data_x.keys()):
            for part in ["normal", "additional", "bsaug"]:
                if not vtlp and part == "additional":
                    continue
                if part == "bsaug" and part not in data_x[name].keys():
                    continue
                else:
                    for type in ["train", "val"]:
                        if part == "bsaug" and type == "val":
                            continue
                        feature_arr = extractor.get_feature(feature, [data_x[name][part][type]])

                        if name not in data_feature.keys():
                            data_feature[name] = {}

                        if part not in data_feature[name].keys():
                            data_feature[name][part] = {}
                        data_feature[name][part][type] = torch.tensor(feature_arr[0], device="cpu").unsqueeze(1)

    return data_feature
            # with open("{}/data_x_{}_{}_feature_{}".format(target_dir, t, overlap, feature), "wb") as f:
            #     pickle.dump(data_feature, f)


def random_labels(n: int, length: int):
    labels = sorted(random.sample(range(length), n))
    return labels, [i for i in range(length) if i not in labels]


def random_labels_kfold(k: int, length: int) -> list[list[int]]:
    ns = list(range(length))
    random.shuffle(ns)
    split = length // k
    
    ids = [ns[split * i : split * (i + 1)] for i in range(k)]
    ids[-1] = ns[split * (k - 1):]
    return ids

class IEMOCAPModule(pl.LightningDataModule):

    class ListDataset(Dataset):

        def __init__(self, x: List[Tensor], y: List[Tensor]):
            assert len(x) == len(y)
            self.x = x
            self.y = y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            return self.x[i], self.y[i]

    class PackType(Enum):
        AllInOne = "aio"
        BySample = "bys"

        def __str__(self):
            return self.value

    def __init__(self, data_dir: str, features: List[FeatureType], k_fold: Optional[int], vtlp: bool, batch_size: int = 32,
                 train_rate: float = 0.8, val_rate: float = 0.2, test_rate: float = 0, val_is_test: bool = True,
                 sample_rate=16000, t=2, overlap=1, val_overlap=1.6, subset=IEMOCAPSubset.Improvised, train_pack_type: PackType = PackType.AllInOne, blend_sample_enabled: bool = True, ee_only: bool = False, feature_file: Optional[str] = None):
        super().__init__()
        self.val_dataset: torch.utils.data.TensorDataset
        self.train_dataset: torch.utils.data.TensorDataset

        self.data_dir = data_dir
        self.features = features
        self.batch_size = batch_size
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.test_rate = test_rate
        self.val_is_test = val_is_test
        self.sample_rate = sample_rate
        self.t = t
        self.overlap = overlap
        self.val_overlap = val_overlap
        self.subset = subset
        self.train_pack_type = train_pack_type
        self.blend_sample_enabled = blend_sample_enabled
        self.ee_only = ee_only
        self.data_x = None
        self.data_y = None
        self.data_feature = None
        self.k_fold = k_fold
        self.vtlp = vtlp
        self.partition = 0
        self.partitioned_data = None
        self.feature_file = feature_file

    def prepare_data(self):
        if self.feature_file is None:
            process_iemocap(self.data_dir)
            audio_augment(self.data_dir, self.vtlp)

    @staticmethod
    def pack_data(data: Dict[str, Dict[str, torch.Tensor]], labels: List[str], data_slices: List[str], pack_type: PackType, train_or_val, add_dim: bool = False) -> Union[torch.Tensor, list]:
        if pack_type == IEMOCAPModule.PackType.AllInOne:
            result = []
            for data_slice in data_slices:
                result += torch.cat([data[n][data_slice][train_or_val] for n in labels if data_slice in data[n].keys()], dim=0)
            if len(result[0].size()) == 0:
                result = [x.view(1) for x in result]
            result = torch.cat(result, dim=0)
            if add_dim:
                result = torch.unsqueeze(result, dim=1)
            return result
        elif pack_type == IEMOCAPModule.PackType.BySample:
            result = []
            for data_slice in data_slices:
                result.extend([data[n][data_slice][train_or_val] for n in labels if data_slice in data[n].keys()])
            result = list(filter(lambda x: len(x) != 0, result))
            return result

    @staticmethod
    def blend_sample_augment(data_x, data_y, ee_only: bool):
        print("Blend-Sample augmenting...")

        emotion_count = [0, 0, 0, 0]
        emotion_labels = [[], [], [], []]

        for k in data_y.keys():
            emotion_class = data_y[k]["normal"]["train"][0]
            emotion_count[emotion_class] += 1
            emotion_labels[emotion_class].append(k)

        neutral_labels = emotion_labels[0]
        emotional_labels = emotion_labels[1] + emotion_labels[2] + emotion_labels[3]

        pairs_ne = list(itertools.product(neutral_labels, emotional_labels))
        pairs_ne = random.sample(pairs_ne, round(len(neutral_labels) / 2))

        pairs_ee = []
        for emotion in [1, 2, 3]:
            pairs = list(itertools.product(emotion_labels[emotion], emotion_labels[emotion]))
            if ee_only:
                pairs_ee.extend(random.sample(pairs, emotion_count[0] - emotion_count[emotion]))
            else:
                pairs_ee.extend(random.sample(pairs, round(len(neutral_labels) / 6)))

        if ee_only:
            pairs_ne = pairs_ee
        else:
            pairs_ne.extend(pairs_ee)

        for n, e in tqdm(pairs_ne):
            if "bsaug" not in data_x[e].keys():
                data_x[e]["bsaug"] = {}
                data_x[e]["bsaug"]["train"] = torch.cat([data_x[e]["normal"]["train"], data_x[n]["normal"]["train"]])
                new_len = data_x[e]["bsaug"]["train"].size()[0]
                data_y[e]["bsaug"] = {}
                data_y[e]["bsaug"]["train"] = torch.empty(new_len).long()
                data_y[e]["bsaug"]["train"].fill_(data_y[e]["normal"]["train"][0])

        return data_x, data_y

    def prepare_data_after(self):
        self.data_x, self.data_y = generate_data(self.data_dir, self.sample_rate, self.t, self.overlap, self.val_overlap, self.subset, self.vtlp)
        if self.blend_sample_enabled:
            self.data_x, self.data_y = self.blend_sample_augment(self.data_x, self.data_y, ee_only=self.ee_only)
        self.data_feature = generate_features(self.data_dir, self.features, self.sample_rate, self.t, self.overlap,
                                              self.data_x, self.data_y, self.vtlp)

    def select_partition(self, partition: int):
        self.partition = partition

    def k_fold_setup(self, stage: Optional[str], names: list):
        if self.feature_file is not None:
            with open(self.feature_file, 'rb') as f:
                self.partitioned_data = pickle.load(f)
                x_data, y_data = self.partitioned_data
        elif self.partitioned_data is None:
            ids = random_labels_kfold(self.k_fold, len(names))
            labels = list(map(lambda idx: list(map(lambda i: names[i], idx)), ids))

            data_y = self.data_y

            x_data = {}
            y_data = {}
            
            data_slices = ["normal"]
            if self.vtlp:
                data_slices.append("additional")
            if self.blend_sample_enabled:
                data_slices.append("bsaug")

            for partition in range(self.k_fold):
                partition_train_y = self.pack_data(data_y, labels[partition], data_slices, self.train_pack_type, train_or_val='train')
                partition_val_y = self.pack_data(data_y, labels[partition], ['normal'], self.train_pack_type, train_or_val='val')
                partition_test_y = self.pack_data(data_y, labels[partition], ['normal'], self.PackType.BySample, train_or_val='val')

                if partition not in y_data.keys():
                    y_data[partition] = {}

                y_data[partition]['train'] = partition_train_y
                y_data[partition]['val'] = partition_val_y

                if len(partition_test_y) != 0:
                    y_data[partition]['test'] = partition_test_y

                for feature in self.features:
                    data_feature = self.data_feature

                    partition_train_x = self.pack_data(data_feature, labels[partition], data_slices, self.train_pack_type, add_dim=True, train_or_val='train')
                    partition_val_x = self.pack_data(data_feature, labels[partition], ['normal'], self.train_pack_type, add_dim=True, train_or_val='val')
                    partition_test_x = self.pack_data(data_feature, labels[partition], ['normal'], self.PackType.BySample, train_or_val='val')

                    if partition not in x_data.keys():
                        x_data[partition] = {}

                    x_data[partition]['train'] = partition_train_x
                    x_data[partition]['val'] = partition_val_x

                    if len(partition_test_x) != 0:
                        x_data[partition]['test'] = partition_test_x # temp: we are now only using one feature type

            self.partitioned_data = (x_data, y_data)
        else:
            x_data, y_data = self.partitioned_data

        partitions = set(range(self.k_fold))
        train_partitions = partitions - set([self.partition])

        train_x = [x_data[p]['train'] for p in train_partitions]
        train_x = torch.cat([item for item in train_x])
        val_x = [x_data[p]['val'] for p in train_partitions]
        val_x = torch.cat([item for item in val_x])
        test_x = x_data[self.partition]['test']
        train_y = [y_data[p]['train'] for p in train_partitions]
        train_y = torch.cat([item for item in train_y])
        val_y = [y_data[p]['val'] for p in train_partitions]
        val_y = torch.cat([item for item in val_y])
        test_y = y_data[self.partition]['test']

        if stage in (None, "fit"):
            if self.train_pack_type == self.PackType.AllInOne:
                self.train_dataset = TensorDataset(train_x, train_y.long())
                self.val_dataset = TensorDataset(val_x, val_y.long())
            elif self.train_pack_type == self.PackType.BySample:
                self.train_dataset = IEMOCAPModule.ListDataset(train_x, train_y)
                self.val_dataset = IEMOCAPModule.ListDataset(test_x, test_y)

        if stage in (None, "test"):
            self.test_dataset = IEMOCAPModule.ListDataset(test_x, test_y)

        return

    def setup(self, stage: Optional[str] = None):
        if self.feature_file is not None:
            return self.k_fold_setup(stage, None)

        if self.partitioned_data is None:
            self.prepare_data_after()
        names = list(self.data_x.keys())
        # with open("{}/generated_data/dataset_data/names_{}_{}".format(self.data_dir, self.t, self.overlap), "rb") as f:
        #     names = pickle.load(f)

        if self.k_fold is not None:
            return self.k_fold_setup(stage, names)

        val_num = int(len(names) * self.val_rate)
        test_num = int(len(names) * self.test_rate)
        train_num = len(names) - val_num - test_num

        train_labels, val_test_labels = random_labels(train_num, len(names))
        val_labels, test_labels = random_labels(val_num, len(val_test_labels))
        train_labels = [names[l] for l in train_labels]
        val_labels = [names[nl] for nl in (val_test_labels[l] for l in val_labels)]

        if self.val_is_test:
            test_labels = val_labels
        else:
            test_labels = [names[nl] for nl in (val_test_labels[l] for l in test_labels)]

        # with open("{}/generated_data/dataset_data/data_y_{}_{}".format(self.data_dir, self.t, self.overlap), "rb") as f:
        #     data_y = pickle.load(f)
        data_y = self.data_y

        x_data = {}
        y_data = {}

        data_slices = ["normal"]
        if self.vtlp:
            data_slices.append("additional")
        if self.blend_sample_enabled:
            data_slices.append("bsaug")

        train_y = self.pack_data(data_y, train_labels, data_slices, self.train_pack_type, train_or_val="train")
        if len(val_labels) != 0:
            val_y = self.pack_data(data_y, val_labels, ["normal"], self.train_pack_type, train_or_val="val")
        if len(test_labels) != 0:
            test_y = self.pack_data(data_y, test_labels, ["normal"], self.PackType.BySample, train_or_val="val")

        max_length = 0
        for n in data_y.keys():
            for data_slice in ["normal", "additional", "bsaug"]:
                if not self.vtlp and data_slice == "additional":
                    continue
                if data_slice == "bsaug" and data_slice not in data_y[n].keys():
                    continue
                length = len(data_y[n][data_slice])
                if length > max_length:
                    max_length = length
        print("Max length (estimation) is: {}".format(max_length))

        for feature in self.features:
            # with open("{}/generated_data/dataset_features/data_x_{}_{}_feature_{}".format(self.data_dir, self.t, self.overlap, feature), "rb") as f:
            #     data_feature = pickle.load(f)
            data_feature = self.data_feature

            train_feature = self.pack_data(data_feature, train_labels, data_slices, self.train_pack_type, add_dim=True, train_or_val="train")
            if len(val_labels) != 0:
                val_feature = self.pack_data(data_feature, val_labels, ["normal"], self.train_pack_type, add_dim=True, train_or_val="val")
            if len(test_labels) != 0:
                test_feature = self.pack_data(data_feature, test_labels, ["normal"], self.PackType.BySample, train_or_val="val")

            if "train" not in x_data.keys():
                x_data["train"] = train_feature
                y_data["train"] = train_y
            else:
                if self.train_pack_type == self.PackType.AllInOne:
                    x_data["train"] = torch.cat([x_data["train"], train_feature], 0)
                    y_data["train"] = torch.cat([y_data["train"], train_y], 0)
                elif self.train_pack_type == self.PackType.BySample:
                    x_data["train"].extend(train_feature)
                    y_data["train"].extend(train_y)

            if len(val_labels) != 0:
                if "val" not in x_data.keys():
                    x_data["val"] = val_feature
                    y_data["val"] = val_y
                else:
                    x_data["val"].extend(val_feature)
                    y_data["val"].extend(val_y)

            if len(test_labels) != 0:
                if "test" not in x_data.keys():
                    x_data["test"] = test_feature
                    y_data["test"] = test_y
                else:
                    x_data["test"].extend(test_feature)
                    y_data["test"].extend(test_y)

        labels = [0, 0, 0, 0]
        for y in y_data["train"]:
            labels[y] += 1
        print("DEBUG: Train label spread: {}".format(labels))

        if stage in (None, "fit"):
            if self.train_pack_type == self.PackType.AllInOne:
                self.train_dataset = TensorDataset(x_data["train"], y_data["train"].long())
            elif self.train_pack_type == self.PackType.BySample:
                self.train_dataset = IEMOCAPModule.ListDataset(x_data["train"], y_data["train"])
            if self.val_rate != 0:
                if self.train_pack_type == self.PackType.AllInOne:
                    self.val_dataset = TensorDataset(x_data["val"], y_data["val"].long())
                elif self.train_pack_type == self.PackType.BySample:
                    self.val_dataset = IEMOCAPModule.ListDataset(x_data["val"], y_data["val"])

        if stage in (None, "test"):
            self.test_dataset = IEMOCAPModule.ListDataset(x_data["test"], y_data["test"])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8,
                          persistent_workers=True, collate_fn=self.collate_fn())

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, num_workers=8, persistent_workers=True, collate_fn=self.collate_fn())

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=1, num_workers=8, persistent_workers=True)

    @staticmethod
    def simple_collate(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        return [data, target]

    def collate_fn(self):
        if self.train_pack_type == IEMOCAPModule.PackType.BySample:
            return self.simple_collate
        else:
            return None
