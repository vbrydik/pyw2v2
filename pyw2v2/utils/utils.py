import os
import yaml
import pandas as pd
from easydict import EasyDict
from pathlib import Path
from datasets import Dataset


def load_config(path):
    path = Path(path)
    if os.path.exists(path):
        with open(path) as f:
            config = EasyDict(yaml.safe_load(f))
    else:
        raise FileNotFoundError(f"No such file {path}")
    return config


def load_custom_dataset_commonvoice_format(path, split, path_column='path'):
    # TODO: add support for multiple split to be together. Example: train+validation
    dataset_path = Path(path) / (split + '.tsv')
    df = pd.read_csv(dataset_path, sep='\t')
    df[path_column] = [str((Path(path) / p).absolute()) for _, p in df[path_column].iteritems()]
    return Dataset.from_pandas(df)