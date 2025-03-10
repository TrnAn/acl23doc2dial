from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.document_grounded_dialog_rerank_trainer import \
    DocumentGroundedDialogRerankTrainer
from modelscope.utils.constant import DownloadMode
from modelscope.hub.snapshot_download import snapshot_download

import argparse
import json
import pandas as pd
import utils.preprocessing as preprocessing
from utils.seed import set_seed
from utils.preprocessing import get_args, add_translation2trainset
import utils.data_exploration as exploration
set_seed()


def main(**kwargs):
    kwargs.update({
        'device': 'gpu',
        'tokenizer_name': '',
        'cache_dir': kwargs["cache_dir"],
        'instances_size': 1,
        'output_dir': f'{kwargs["cache_dir"]}/output',
        'max_num_seq_pairs_per_device': 32,
        'full_train_batch_size': 32,
        'gradient_accumulation_steps': 32,
        'per_gpu_train_batch_size': 1,
        'num_train_epochs': 10,
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

    kwargs[
        'gradient_accumulation_steps'] = kwargs['full_train_batch_size'] // (
            kwargs['per_gpu_train_batch_size'] * kwargs['world_size'])
    
    langs = set(kwargs["target_langs"]) if kwargs["translate_mode"] == "test" else set(item for sublist in kwargs["eval_lang"] for item in sublist) 
    train_dataset_fr,  train_dataset_vi, train_dataset_en, train_dataset_cn = None, None, None, None

    if "en" in langs:
        train_dataset_en = pd.read_json(f"{kwargs['cache_dir']}/DAMO_ConvAI/nlp_convai_retrieval_pretrain/en_{kwargs['extended_rerank_dataset_fname']}")
        train_dataset_en['output']      = train_dataset_en['output'].apply(eval)

    if "cn" in langs:
        train_dataset_cn = pd.read_json(f"{kwargs['cache_dir']}/DAMO_ConvAI/nlp_convai_retrieval_pretrain/cn_{kwargs['extended_rerank_dataset_fname']}")
        print(train_dataset_cn.columns)
        print(train_dataset_cn.head(1))
        train_dataset_cn['output']      = train_dataset_cn['output'].apply(eval)

    if "fr" in langs:
        train_dataset_fr = MsDataset.load(
            'DAMO_ConvAI/FrDoc2BotRerank',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
            split='train')
        train_dataset_fr = pd.DataFrame(list(train_dataset_fr))
        train_dataset_fr["lang"] = "fr"

   
    if "vi" in langs:
        train_dataset_vi = MsDataset.load(
            'DAMO_ConvAI/ViDoc2BotRerank',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
            split='train')
        train_dataset_vi = pd.DataFrame(list(train_dataset_vi))

        train_dataset_vi["lang"] = "vi"


    train_dataset_fr, dev_dataset_fr = preprocessing.test_split(train_dataset_fr)
    train_dataset_vi, dev_dataset_vi = preprocessing.test_split(train_dataset_vi)
    train_dataset_en, dev_dataset_en = preprocessing.test_split(train_dataset_en)
    train_dataset_cn, dev_dataset_cn = preprocessing.test_split(train_dataset_cn)

     # add machine translated en -> fr, vi queries to train set
    if kwargs["translate_mode"] == "train": 
        pipeline_step = 'rerank'
        train_dataset_vi = add_translation2trainset(train_df=train_dataset_vi, lang='vi', pipeline_step=pipeline_step, dir=kwargs["cache_dir"])
        train_dataset_fr = add_translation2trainset(train_df=train_dataset_fr, lang='fr', pipeline_step=pipeline_step, dir=kwargs["cache_dir"])
    

    if kwargs["lang_token"]:
        if "vi" in langs:
            train_dataset_vi = preprocessing.add_lang_token(train_dataset_vi, "vi", ["input"]) 
            train_dataset_vi['passages'] = train_dataset_vi['passages'].apply(lambda x: json.dumps([{'pid': d['pid'], 'text': f"{preprocessing.LANG_TOKENS_DD['vi']} {d['text']}"} for d in eval(x)], ensure_ascii=False))
            dev_dataset_vi = preprocessing.add_lang_token(dev_dataset_vi, "vi", ["input"]) 
            dev_dataset_vi['passages'] = dev_dataset_vi['passages'].apply(lambda x: json.dumps([{'pid': d['pid'], 'text': f"{preprocessing.LANG_TOKENS_DD['vi']} {d['text']}"} for d in eval(x)], ensure_ascii=False))
        if "fr" in langs:
            train_dataset_fr = preprocessing.add_lang_token(train_dataset_fr, "fr", ["input"]) 
            train_dataset_fr['passages'] = train_dataset_fr['passages'].apply(lambda x: json.dumps([{'pid': d['pid'], 'text': f"{preprocessing.LANG_TOKENS_DD['fr']} {d['text']}"} for d in eval(x)], ensure_ascii=False))
            dev_dataset_fr = preprocessing.add_lang_token(dev_dataset_fr, "fr", ["input"]) 
            dev_dataset_fr['passages'] = dev_dataset_fr['passages'].apply(lambda x: json.dumps([{'pid': d['pid'], 'text': f"{preprocessing.LANG_TOKENS_DD['fr']} {d['text']}"} for d in eval(x)], ensure_ascii=False))
    

    lang_dd = {
        "fr": (train_dataset_fr, dev_dataset_fr),
        "vi": (train_dataset_vi, dev_dataset_vi),
        "en": (train_dataset_en, dev_dataset_en),
        "cn": (train_dataset_cn, dev_dataset_cn)
    }

    train_dataset = [] 
    dev_dataset = []
   
    for lang in langs:
        train, dev = lang_dd[lang]
        train_dataset.append(train)
        dev_dataset.append(dev)
    
    train_dataset   = pd.concat(train_dataset) 
    dev_dataset     = pd.concat(dev_dataset)
    tmp_df = pd.concat([train_dataset, dev_dataset])
    tmp_df = tmp_df.reset_index(drop=True)
    tmp_df["id"] = tmp_df.index.astype(str)

    train_dataset   = tmp_df[:len(train_dataset)]
    dev_dataset     = tmp_df[len(train_dataset):]

    # ensure equal dataset sizes
    if kwargs["equal_dataset_size"]:
        train_dataset    = preprocessing.get_equal_dataset_size_by_lang(train_dataset)
        dev_dataset      = preprocessing.get_equal_dataset_size_by_lang(dev_dataset)
    
    freq_df = exploration.get_freq_df(train_dataset, dev_dataset)
    exploration.plot_freq(freq_df, plot_dir=f'{kwargs["cache_dir"]}/plot', fname="freq_dist_rerank.png")

    if kwargs["translate_mode"] == "test":
        kwargs["eval_lang"] = [kwargs["target_langs"]]
        
    trainer = DocumentGroundedDialogRerankTrainer(
        model='DAMO_ConvAI/nlp_convai_ranking_pretrain', 
        train_dataset   = train_dataset.to_dict('records'),
        eval_dataset    = dev_dataset.to_dict('records'),
        args=kwargs
        )
    
    trainer.train()
    trainer.evaluate(eval_lang=kwargs["eval_lang"])

    extended_lang = langs - set(["fr", "vi"])
    if len(extended_lang) > 0:
        for lang_tag in extended_lang:
            combined_df = pd.concat([lang_dd[lang_tag][0], lang_dd[lang_tag][1]])
            trainer.save_dataset(dataset=combined_df.to_dict('records'), fname=f"{lang_tag}_{kwargs['extended_generation_dataset_fname']}")


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)
