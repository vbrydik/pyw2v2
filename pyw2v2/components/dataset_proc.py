import librosa
import torchaudio
import numpy as np
from datasets import Dataset
from easydict import EasyDict
from transformers import Wav2Vec2Processor
# from datasets.utils.logging import set_verbosity_error
# set_verbosity_error()


class DatasetPreprocessor:
    
    def __init__(self, config: EasyDict, processor: Wav2Vec2Processor=None):
        self._text_column = config.text_column
        self._path_column = config.path_column
        self._remove_chars = config.remove_chars
        self._replace_chars = config.replace_chars
        self._sampling_rate = config.sampling_rate
        self._num_proc = config.n_jobs
        self._processor = processor
        
    @property
    def processor(self):
        return self._processor
    
    @processor.setter
    def processor(self, var: Wav2Vec2Processor):
        self._processor = var

    def _to_lower(self, batch):
        batch[self._text_column] = batch[self._text_column].lower() 
        return batch

    def _remove_characters(self, batch):
        for char in self._remove_chars:
            batch[self._text_column] = batch[self._text_column].replace(char, '')
        return batch
    
    def _replace_characters(self, batch):
        for k, v in self._replace_chars.items():
            batch[self._text_column] = batch[self._text_column].replace(k, v)
        return batch
    
    def _speech_file_to_array(self, batch):
        speech_array, sampling_rate = torchaudio.load(batch[self._path_column])
        batch["speech"] = speech_array[0].numpy()
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch[self._text_column]
        return batch
    
    def _resample(self, batch):
        batch["speech"] = librosa.resample(np.asarray(batch["speech"]), batch["sampling_rate"], self._sampling_rate)
        batch["sampling_rate"] = self._sampling_rate
        return batch

    def _prepare_batch(self, batch):
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {self._processor.feature_extractor.sampling_rate}."
        batch["input_values"] = self._processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
        with self._processor.as_target_processor():
            batch["labels"] = self._processor(batch["target_text"]).input_ids
        return batch
    
    @staticmethod
    def dataset_head(dataset: Dataset, n: int) -> Dataset:
        return Dataset.from_dict(dataset[:n])

    def process(self, dataset: Dataset, n_samples: int=None) -> Dataset:
        if n_samples:
            dataset = self.dataset_head(dataset, n_samples)
        dataset = dataset.map(self._remove_characters, num_proc=self._num_proc)
        dataset = dataset.map(self._replace_characters, num_proc=self._num_proc)
        dataset = dataset.map(self._to_lower, num_proc=self._num_proc)
        dataset = dataset.map(self._speech_file_to_array, num_proc=self._num_proc)
        dataset = dataset.map(self._resample, num_proc=self._num_proc)
        dataset = dataset.map(self._prepare_batch, batch_size=8, batched=True, num_proc=self._num_proc)
        return dataset