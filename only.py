import argparse
import numpy as np
import torch.nn as nn
from logging import getLogger
from recbole.config import Config
from recbole.data.dataloader import TrainDataLoader
from recbole.utils import init_seed, init_logger
from model.rpqrec import RPQRec
from data.dataset import FederatedDataset
import os
import faiss
import torch
from trainer import RPQRecTrainer
from utils import parse_faiss_index

# Function to load index and convert it to tensor format
def load_index(config, logger, item_num, field2id_token):
    code_dim = config['code_dim']
    code_cap = config['code_cap']
    dataset_name = config['dataset']
    index_suffix = config['index_suffix']
    if config['index_pretrain_dataset'] is not None:
        index_dataset = config['index_pretrain_dataset']
    else:
        index_dataset = config['dataset']
    index_path = os.path.join(
        config['index_path'],
        index_dataset,
        f'{index_dataset}.{index_suffix}'
    )
    logger.info(f'Index path: {index_path}')
    uni_index = faiss.read_index(index_path)
    rpq_codes, centroid_embeds, coarse_embeds, opq_transform = parse_faiss_index(uni_index)
    assert code_dim == rpq_codes.shape[1], rpq_codes.shape
    # uint8 -> int32 to reserve 0 padding
    rpq_codes = rpq_codes.astype(np.int32)
    # 0 for padding
    rpq_codes = rpq_codes + 1
    # flatten pq codes
    base_id = 0
    for i in range(code_dim):
        rpq_codes[:, i] += base_id
        base_id += code_cap + 1

    logger.info('Loading filtered index mapping.')
    filter_id_dct = {}
    with open(
            os.path.join(config['data_path'],
                         f'{dataset_name}.{config["filter_id_suffix"]}'),
            'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            filter_id_name = line.strip()
            filter_id_dct[filter_id_name] = idx

    logger.info('Converting indexes.')
    mapped_codes = np.zeros((item_num, code_dim), dtype=np.int32)
    for i, token in enumerate(field2id_token):
        if token == '[PAD]': continue
        mapped_codes[i] = rpq_codes[filter_id_dct[token]]
    return torch.LongTensor(mapped_codes)

def pretrain(dataset, **kwargs):
    # Configurations initialization
    props = ['props/RPQRec.yaml', 'props/pretrain.yaml']
    print(props)

    # Manually set train_neg_sample_args to None
    kwargs['train_neg_sample_args'] = None

    # Load configuration for a single dataset
    config = Config(model=RPQRec, dataset=dataset, config_file_list=props, config_dict=kwargs)

    # Seed initialization
    init_seed(config['seed'], config['reproducibility'])

    # Logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    logger.info(dataset)

    # Load the dataset
    dataset_A = FederatedDataset(config, rpq_codes=None)
    logger.info(dataset_A)
    pretrain_dataset_A = dataset_A.build()[0]

    # Load item embeddings
    item_num = dataset_A.item_num
    rpq_codes = load_index(config, logger, item_num, dataset_A.field2id_token['item_id']).to(config['device'])
    pretrain_dataset_A.rpq_codes = rpq_codes

    # Create data loader
    pretrain_data_A = TrainDataLoader(config, pretrain_dataset_A, None, shuffle=True)

    # Initialize model
    model_A = RPQRec(config, pretrain_data_A.dataset).to(config['device'])
    model_A.rpq_codes.to(config['device'])
    logger.info(model_A)

    # Initialize embedding
    global_embedding = nn.Embedding(
        config['code_dim'] * (1 + config['code_cap']), config['hidden_size'], padding_idx=0).to(config['device'])
    global_embedding.weight.data.normal_(mean=0.0, std=config['initializer_range'])

    # Load the global embedding state into the model
    model_A.pq_code_embedding.load_state_dict(global_embedding.state_dict())

    # Initialize trainer and start training
    trainer = RPQRecTrainer(config, model_A)
    trainer.fit(pretrain_data_A, show_progress=True)

    return config['model'], config['dataset']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='M', help='dataset name')
    args, unparsed = parser.parse_known_args()
    print(args)

    model, dataset = pretrain(args.d)
