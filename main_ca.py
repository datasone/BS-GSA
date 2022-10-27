import random
from typing import List, Optional

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything

from config import Config, ConfigParseResult
from iemocap_data.data_module import IEMOCAPSubset, IEMOCAPModule
from iemocap_data.features import FeatureType
from gsa.model_torch import CATP

def train_test(seed, data_module: Optional[IEMOCAPModule], partition: Optional[int], feature_file: Optional[str]) -> tuple[list[dict[str, float]], IEMOCAPModule, int]:
    if seed is None:
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)  # seed larger than i32::MAX is unavailable

    seed_everything(seed)

    if config.frame_pooling:
        pack_type = IEMOCAPModule.PackType.BySample
    else:
        pack_type = IEMOCAPModule.PackType.AllInOne

    if data_module is None:
        data_module = IEMOCAPModule(config.data_dir, config.features, t=config.t, overlap=config.overlap, batch_size=config.batch_size, train_rate=config.train_rate, val_rate=config.val_rate, test_rate=config.test_rate, val_is_test=config.val_is_test, train_pack_type=pack_type, blend_sample_enabled=config.blend_sample, ee_only=config.ee_only, k_fold=config.k_fold, vtlp=config.vtlp, feature_file=config.feature_file)

    if partition is not None:
        data_module.select_partition(partition)

    model_name = 'BS-GSA-seed={}'.format(seed)

    checkpoint_callback = ModelCheckpoint(
        monitor="sum_acc",
        dirpath="models/",
        filename=model_name + "-{epoch:02d}-{wa:.4f}-{ua:.4f}-{sum_acc:.4f}",
        save_top_k=3,
        mode="max",
    )

    model = CATP(out_types=4, lr=config.lr, dropout_rate=config.dropout_rate, t=config.t, aacnn_mode=config.aacnn_mode(), frame_pooling_enabled=config.frame_pooling, target_dim=30, max_length=30 * 80)

    trainer = Trainer(callbacks=[checkpoint_callback, EarlyStopping(monitor='wa', mode='max', stopping_threshold=0.9999, patience=20)], gpus=config.gpu_num_or_list(), auto_select_gpus=True,
                      max_epochs=config.epochs, auto_scale_batch_size=True, precision=16)
    trainer.fit(model=model, datamodule=data_module)

    results = trainer.test(model=model, datamodule=data_module)
    #with open("result.txt", "w") as f:
    #    f.write(str(results))
    return results, data_module, seed

if __name__ == "__main__":
    config = Config()

    print(config.__dict__)

    match config.validate():
        case ConfigParseResult.SPLIT_RATE_ERR_NO_TEST:
            raise Exception("Train split and valid split should sum up to 1")
        case ConfigParseResult.SPLIT_RATE_ERR:
            raise Exception("Train, valid and test split should sum up to 1")

    if config.seeds is None:
        seeds = [None]
    else:
        seeds = config.seeds

    for seed in seeds:
        if config.k_fold is not None:
            data_module = None
            results = []

            for iteration in range(config.k_fold):
                result, module, seed = train_test(seed, partition=iteration, data_module=data_module, feature_file=config.feature_file)
                data_module = module
                results.append(result[0])

            ua = sum(map(lambda x: x['ua'], results)) / config.k_fold
            wa = sum(map(lambda x: x['wa'], results)) / config.k_fold
            print({'ua': ua, 'wa': wa})
            with open('result.txt', 'w') as f:
                f.write(str({'ua': ua, 'wa': wa}))
        else:
            results = train_test(seed, data_module=None, partition=None)
            print(str(results))
