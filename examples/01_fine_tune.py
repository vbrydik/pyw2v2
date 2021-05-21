from pyw2v2 import ModelCTC, DatasetPreprocessor
from pyw2v2.utils import load_config, load_custom_dataset_commonvoice_format

if __name__ == "__main__":
    # Load pretrained model
    model_config = load_config("./configs/ctc/default.yaml")
    model = ModelCTC(model_config)
    
    # Loading dataset in Common Voice format
    train_set = load_custom_dataset_commonvoice_format('./datasets/example', 'train')
    eval_set = load_custom_dataset_commonvoice_format('./datasets/example', 'test')

    # Set up dataset preprocessor
    dataproc_config = load_config("./configs/dataproc/default.yaml").data_proc
    data_processor = DatasetPreprocessor(dataproc_config)
    data_processor.processor = model.processor

    # Process data
    train_set = data_processor.process(train_set, dataproc_config.n_samples_train)
    eval_set = data_processor.process(eval_set, dataproc_config.n_samples_test)
    
    # Train/Fine-tune model
    model.train(train_set, eval_set)