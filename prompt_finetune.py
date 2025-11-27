import argparse
import numpy as np
from logging import getLogger
from recbole.config import Config
from recbole.data.dataloader import TrainDataLoader
from recbole.utils import set_color
from recbole.utils import init_seed, init_logger
from model.rpqrec import RPQRec
from model.full_prompt import id_prompt
from model.light_prompt import seq_prompt
from model.item_prompt import item_prompt
from data.dataset import FederatedDataset
import os
import faiss
import torch
from recbole.data import data_preparation
from trainer import RPQRecTrainer
from utils import parse_faiss_index

def load_index(config, logger, field2id_token, item_num):
    code_dim = config['code_dim']
    code_cap = config['code_cap']
    dataset_name = config['dataset']
    index_suffix = config['index_suffix']
    pq_path = config['pq_data']
    if config['index_pretrain_dataset'] is not None:
        index_dataset = config['index_pretrain_dataset']
    else:
        index_dataset = config['dataset']
    index_path = os.path.join(
        config['index_path'],
        pq_path,
        f'{pq_path}.{index_suffix}'
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

def change_dict(dict, point):
    for k in dict:
        if k == '[PAD]':
            continue
        dict[k] += point

def finetune(model_name, dataset, pretrained_file='', finetune_mode='', **kwargs):
    # configurations initialization
    props = [f'props/{model_name}.yaml', 'props/prompt.yaml']
    print(props)

    kwargs['train_neg_sample_args'] = None

    # configurations initialization
    config = Config(model=RPQRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    config_A = Config(model=RPQRec, dataset='G', config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    init_seed(config_A['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config_A)
    logger = getLogger()
    logger.info(config_A)
    dataset = FederatedDataset(config_A, rpq_codes=None)
    dataset_A = FederatedDataset(config_A, rpq_codes=None)
    logger.info(dataset_A)
    pretrain_dataset_A = dataset_A.build()[0]
    rpq_codes = load_index(config, logger, dataset_A.field2id_token['item_id'], dataset_A.item_num).to(config['device'])
    pretrain_dataset_A.rpq_codes = rpq_codes
    dataset.rpq_codes = rpq_codes
    pretrain_data_A = TrainDataLoader(config_A, pretrain_dataset_A, None, shuffle=True)
    train_data, valid_data, test_data = data_preparation(config_A, dataset)

    model_A = RPQRec(config_A, pretrain_data_A.dataset).to(config['device'])
    model_A.rpq_codes.to(config['device'])
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model_A.load_state_dict(checkpoint['state_dict'], strict=False)
    prompt_model = id_prompt(config_A, pretrain_data_A.dataset, model_A).to(config['device'])
    prompt_model.rpq_codes.to(config['device'])
    for _ in prompt_model.rpqrec.parameters():
        _.requires_grad = False
    for _ in prompt_model.rpqrec.rpq_code_embedding.parameters():
        _.requires_grad = True
    # if finetune_mode == 'fix_enc':
    #     logger.info('[Fine-tune mode] Fix Seq Encoder!')
    #     for _ in model_A.module.position_embedding.parameters():
    #         _.requires_grad = False
    #     for _ in model_A.module.trm_encoder.parameters():
    #         _.requires_grad = False
    #     for _ in model_A.module.rpq_code_embedding.parameters():
    #         _.requires_grad = False
    # elif finetune_mode == 'fix_prompt':
    #     logger.info('[Fine-tune mode] Fix Prompt!')
    #     model_A.module.prompts.requires_grad = False
    #     for _ in model_A.module.attn_layer.parameters():
    #         _.requires_grad = True
    num_params = sum(p.numel() for p in prompt_model.parameters() if p.requires_grad)
    logger.info(model_A)
    logger.info(prompt_model)
    print(f'Number of parameters: {num_params}')
    trainer = RPQRecTrainer(config_A, prompt_model)
    best_valid_score, best_valid_result = trainer.fit(pretrain_data_A, valid_data, show_progress=True)
    # best_valid_result = trainer.evaluate(valid_data, load_best_model=False, show_progress=True)
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')

    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config_A['model'], config_A['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config_A['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='RPQRec', help='model name')
    parser.add_argument('-d', type=str, default='MG', help='dataset name')
    parser.add_argument('-p', type=str, default='save_MG/RPQRec-Nov-07-2024_11-10-05.pth', help='pre-trained model path')
    parser.add_argument('-f', type=str, default='', help='fine-tune mode')
    args, unparsed = parser.parse_known_args()
    print(args)
    finetune(args.m, args.d, pretrained_file=args.p, finetune_mode=args.f)