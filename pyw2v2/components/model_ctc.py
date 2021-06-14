import torch
import numpy as np
from easydict import EasyDict
from datasets import Dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import TrainingArguments, Trainer

from pyw2v2.components.metrics import Metrics
from pyw2v2.components.external.data_collator_ctc import DataCollatorCTCWithPadding
from pyw2v2.components.dataset_proc import DatasetPreprocessor


class ModelCTC:

    def __init__(self, config: EasyDict):
        self._model = None
        self._metrics = None
        self._processor = None
        self._data_collator = None
        self._training_args = None
        self._dataset_preprocessor = None
        
        self._init_processor(config)
        self._init_metrics(config)
        self._init_data_collator()
        self._init_model(config)
        self._init_training_args(config)
        
    @property
    def processor(self):
        return self._processor
    
    @processor.setter
    def processor(self, var: Wav2Vec2Processor):
        self._processor = var

    @property
    def dataset_preprocessor(self):
        return self._dataset_preprocessor
    
    @processor.setter
    def dataset_preprocessor(self, var: DatasetPreprocessor):
        self._dataset_preprocessor = var

    def _init_processor(self, config: EasyDict):
        config.processor.tokenizer.vocab_file = config.common.vocab_file
        tokenizer = Wav2Vec2CTCTokenizer(**config.processor.tokenizer)
        feature_extractor = Wav2Vec2FeatureExtractor(**config.processor.feature_extractor)

        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        processor.save_pretrained(config.common.model_path)
        self._processor = processor

    def _init_metrics(self, config: EasyDict):
        self._metrics = Metrics(*config.common.metrics)

    def _init_data_collator(self):
        self._data_collator = DataCollatorCTCWithPadding(processor=self._processor, padding=True)

    def _init_model(self, config: EasyDict):
        if not config.common.checkpoint_model:
            print(f"Loading pretrained model {config.common.pretrained_model}")
            config.model.pretrained_model_name_or_path = config.common.pretrained_model
            config.model.pad_token_id = self._processor.tokenizer.pad_token_id
            config.model.vocab_size = len(self._processor.tokenizer)
            self._model = Wav2Vec2ForCTC.from_pretrained(**config.model)
        else:
            print(f"Loading from checkpoint {config.common.checkpoint_model}")
            self._model = Wav2Vec2ForCTC.from_pretrained(config.common.checkpoint_model).to("cuda")
    
    def _init_training_args(self, config: EasyDict):
        config.training_args.output_dir = config.common.model_path
        self._training_args = TrainingArguments(**config.training_args)
        
    def _compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self._processor.tokenizer.pad_token_id
        pred_str = self._processor.batch_decode(pred_ids)
        label_str = self._processor.batch_decode(pred.label_ids, group_tokens=False)

        res = self._metrics.compute(predictions=pred_str, references=label_str)
        return res
    
    def train(self, train_set: Dataset, eval_set: Dataset):
        trainer = Trainer(
            model=self._model,
            data_collator=self._data_collator,
            args=self._training_args,
            compute_metrics=self._compute_metrics,
            train_dataset=train_set,
            eval_dataset=eval_set,
            tokenizer=self._processor.feature_extractor)
        trainer.train()

    def decode_sample(self, batch):
        input_dict = self._processor(batch["input_values"], return_tensors="pt", padding=True)
        logits = self._model(input_dict.input_values.to("cuda")).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        batch['decoded'] = self._processor.decode(pred_ids)
        return batch

    def decode(self, dataset: Dataset):
        return dataset.map(self.decode_sample)

    def decode_raw(self, path: str or list):
        if isinstance(path, str):
            sample = self._dataset_preprocessor.process_one(path, audio_only=True, make_labels=False)
            return self.decode_sample(sample)
        elif isinstance(path, list):
            samples = [self._dataset_preprocessor.process_one(p, audio_only=True, make_labels=False) for p in path]
            return [self.decode_sample(s) for s in samples]
        else:
            raise NotImplementedError
