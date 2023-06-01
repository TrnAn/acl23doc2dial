from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.document_grounded_dialog_rerank_trainer import \
    DocumentGroundedDialogRerankTrainer
from modelscope.utils.constant import DownloadMode
# from modelscope.hub.snapshot_download import snapshot_download
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
import argparse
import json
import pandas as pd
import os
import random

import utils.preprocessing as preprocessing

def split_dataset(dataset:list, seed:int=42, split_ratio:float=.9):
    random.seed(seed)
    random.shuffle(dataset)

    # Calculate the split points
    train_size = int(split_ratio * len(dataset))

    # Split the data into training and testing sets
    train_ds = dataset[:train_size]
    dev_ds = dataset[train_size:]
    return train_ds, dev_ds


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--test-size", help= "Set test split", type= float, default= 0.1)
    parser.add_argument("--lang-token", help= "Add language token <lang> to input", action=argparse.BooleanOptionalAction)
    parser.add_argument("--extended-dataset", help= "Run experiments on English and Chinese dataset", action=argparse.BooleanOptionalAction)
    parser.add_argument("--only-english", help= "Run experiments only on English dataset", type= int, default=0)
    parser.add_argument("--only-chinese", help= "Run experiments only on Chinese dataset", type= int, default=0)
    parser.add_argument("--cache-dir", help= "Specifiy cache dir to save model to", type= str, default= ".")
    args = vars(parser.parse_args())

    args.update({
        'device': 'gpu',
        'tokenizer_name': '',
        'cache_dir': args["cache_dir"],
        'instances_size': 1,
        'output_dir': f'{args["cache_dir"]}/output',
        'max_num_seq_pairs_per_device': 32,
        'full_train_batch_size': 32,
        'gradient_accumulation_steps': 32,
        'per_gpu_train_batch_size': 1,
        'num_train_epochs': 1,
        'train_instances': -1,
        'learning_rate': 2e-5,
        'max_seq_length': 512,
        'num_labels': 2,
        'fold': '',  # IofN
        'doc_match_weight': 0.0,
        'query_length': 195,
        'resume_from': '',  # to resume training from a checkpoint
        'config_name': '',
        'do_lower_case': True,
        'weight_decay': 0.0,  # previous default was 0.01
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'warmup_instances': 0,  # previous default was 0.1 of total
        'warmup_fraction': 0.0,  # only applies if warmup_instances <= 0
        'no_cuda': False,
        'n_gpu': 1,
        'seed': 42,
        'fp16': False,
        'fp16_opt_level': 'O1',  # previous default was O2
        'per_gpu_eval_batch_size': 8,
        'log_on_all_nodes': False,
        'world_size': 1,
        'global_rank': 0,
        'local_rank': -1,
        'tokenizer_resize': True,
        'model_resize': True
    })

    args[
        'gradient_accumulation_steps'] = args['full_train_batch_size'] // (
            args['per_gpu_train_batch_size'] * args['world_size'])
    
    train_dataset_fr = preprocessing.read('DAMO_ConvAI/FrDoc2BotRerank')
    train_dataset_vi = preprocessing.read('DAMO_ConvAI/ViDoc2BotRerank')

    train_dataset_fr["lang"] = "fr"
    train_dataset_vi["lang"] = "vi"
    # train_dataset_fr = MsDataset.load(
    #     'DAMO_ConvAI/FrDoc2BotRerank',
    #     download_mode=DownloadMode.FORCE_REDOWNLOAD,
    #     split='train')
    

    train_dataset_fr, dev_dataset_fr = preprocessing.test_split(train_dataset_fr)
    train_dataset_vi, dev_dataset_vi = preprocessing.test_split(train_dataset_vi)
    # train_dataset_vi = MsDataset.load(
    #     'DAMO_ConvAI/ViDoc2BotRerank',
    #     download_mode=DownloadMode.FORCE_REDOWNLOAD,
    #     split='train')

    # train_dataset_fr, dev_dataset_fr = split_dataset(list(train_dataset_fr))
    # train_dataset_vi, dev_dataset_vi = split_dataset(list(train_dataset_vi))

    print(f"split size: {len(train_dataset_vi)=} - {len(dev_dataset_vi)=}; {len(train_dataset_fr)=} - {len(dev_dataset_fr)=}")
  
    # parent_dir = "all_passages/lang_token" if args["lang_token"] else "all_passages"
    # all_passages = []
    # languages = ['fr', 'vi']

    # if args["extended_dataset"]:
    #     if not bool(args["only_chinese"]):
    #         languages += ['en']
    #     if not bool(args["only_english"]):
    #         languages += ['cn']

    # for file_name in languages:
    #     with open(f'{parent_dir}/{file_name}.json') as f:
    #         all_passages += json.load(f)
    train_dataset = train_dataset_fr + train_dataset_vi
    dev_dataset   = dev_dataset_fr + dev_dataset_vi

    trainer = DocumentGroundedDialogRerankTrainer(
        model=f'DAMO_ConvAI/nlp_convai_ranking_pretrain', 
        train_dataset=train_dataset.to_dict('records'), 
        dev_dataset=dev_dataset.to_dict('records'), 
        args=args
        )
    
    # trainer.train()

    print(f"=== Evaluation Scores ===")
    trainer.evaluate()


if __name__ == '__main__':
    main()
