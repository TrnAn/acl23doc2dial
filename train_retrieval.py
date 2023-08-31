import os
import json
import pandas as pd
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
import utils.preprocessing as preprocessing
from utils.seed import set_seed
from utils.preprocessing import get_args, add_translation2trainset, get_unique_passages, add_hard_negatives
import utils.data_exploration as exploration
set_seed()
from tqdm import tqdm
tqdm.pandas()

def main(**kwargs):
    fr_train_dataset, vi_train_dataset, en_train_dataset, cn_train_dataset = None, None, None, None
    langs = set(kwargs["source_langs"] + kwargs["target_langs"]) if kwargs["translate_mode"] == "test" else set(item for sublist in kwargs["eval_lang"] for item in sublist) 

    if "fr" in langs:
        fr_train_dataset = preprocessing.read('DAMO_ConvAI/FrDoc2BotRetrieval')
        fr_train_dataset["lang"] = "fr"

    if "vi" in langs:
        vi_train_dataset = preprocessing.read('DAMO_ConvAI/ViDoc2BotRetrieval')
        vi_train_dataset["lang"] = "vi" 

    if "en" in langs:
        en_train_dataset = pd.read_json("en_train_dataset_retrieval_generation_hn.json", lines=True)
        en_train_dataset["lang"] = "en"

    if "cn" in langs:
        cn_train_dataset = pd.read_json("cn_train_dataset_in_domain.json", lines=True)
        cn_train_dataset["lang"] = "cn"


    train_dataset_fr, dev_dataset_fr = preprocessing.test_split(fr_train_dataset)
    train_dataset_vi, dev_dataset_vi = preprocessing.test_split(vi_train_dataset)
    train_dataset_en, dev_dataset_en = preprocessing.test_split(en_train_dataset)
    train_dataset_cn, dev_dataset_cn = preprocessing.test_split(cn_train_dataset)

    # add machine translated en -> fr, vi queries to train set
    if kwargs["translate_mode"] == "train":
        pipeline_step= 'retrieval'
        train_dataset_vi = add_translation2trainset(train_df=train_dataset_vi, lang='vi', pipeline_step=pipeline_step, dir=kwargs["cache_dir"])
        train_dataset_fr = add_translation2trainset(train_df=train_dataset_fr, lang='fr', pipeline_step=pipeline_step, dir=kwargs["cache_dir"])    

    if kwargs["lang_token"]:
        train_dataset_fr    = preprocessing.add_lang_token(train_dataset_fr, "fr", colnames=["query", "positive", "negative"]) 
        train_dataset_vi    = preprocessing.add_lang_token(train_dataset_vi, "vi", colnames=["query", "positive", "negative"])
        train_dataset_en    = preprocessing.add_lang_token(train_dataset_en, "en", colnames=["query", "positive", "negative"])
        train_dataset_cn    = preprocessing.add_lang_token(train_dataset_cn, "cn", colnames=["query", "positive", "negative"])

        dev_dataset_fr = preprocessing.add_lang_token(dev_dataset_fr, "fr", colnames=["query", "positive", "negative"]) 
        dev_dataset_vi = preprocessing.add_lang_token(dev_dataset_vi, "vi", colnames=["query", "positive", "negative"]) 
        dev_dataset_en = preprocessing.add_lang_token(dev_dataset_en, "en", colnames=["query", "positive", "negative"])  
        dev_dataset_cn = preprocessing.add_lang_token(dev_dataset_cn, "cn", colnames=["query", "positive", "negative"])   

    lang_dd = {
        "fr": (train_dataset_fr, dev_dataset_fr),
        "vi": (train_dataset_vi, dev_dataset_vi),
        "en": (train_dataset_en, dev_dataset_en),
        "cn": (train_dataset_cn, dev_dataset_cn)
    }

    train_dataset, dev_dataset = [], []
    if kwargs["translate_mode"] == "test":
        langs = set(kwargs["target_langs"])
        kwargs["eval_lang"] = [list(langs)]
    
    if kwargs["translate_mode"] == "train":
        langs = set(list(langs) + kwargs["source_langs"])

    for lang in langs:
        train, dev = lang_dd[lang]
        train_dataset.append(train)
        dev_dataset.append(dev)
        
    train_dataset   = pd.concat(train_dataset) 
    dev_dataset     = pd.concat(dev_dataset)

    if kwargs["translate_mode"] == "test":

        save_dev_dataset = []
        for lang in kwargs["source_langs"] + kwargs["target_langs"]:
            _, dev = lang_dd[lang]
            save_dev_dataset.append(dev)
        save_dev_dataset = pd.concat(save_dev_dataset)

    else:
        save_dev_dataset = dev_dataset

    if kwargs["equal_dataset_size"]:
        train_dataset       = preprocessing.get_equal_dataset_size_by_lang(train_dataset)
        dev_dataset         = preprocessing.get_equal_dataset_size_by_lang(dev_dataset)
        save_dev_dataset    = preprocessing.get_equal_dataset_size_by_lang(save_dev_dataset)

    # preprocessing.save_to_json(save_dev_dataset, save_dev_dataset.columns, fname=kwargs["eval_input_file"], pdir=kwargs["cache_dir"])

    parent_dir = "all_passages/lang_token" if kwargs["lang_token"] else "all_passages"
    all_passages = []
    translated_passages = []
    for file_name in langs:
        with open(f'{parent_dir}/{file_name}.json') as f:
            all_passages += json.load(f)

            if kwargs["translate_mode"] == "train":
                translated_passages += get_unique_passages(locals()[f"train_dataset_{lang}"], lang=lang) if kwargs["lang_token"] else get_unique_passages(locals()[f"train_dataset_{lang}"])
                
    all_passages_w_translations =  list(set(all_passages) | set(translated_passages))
    
    freq_df = exploration.get_freq_df(train_dataset, dev_dataset)
    exploration.plot_freq(freq_df, plot_dir=f'{kwargs["cache_dir"]}/plot', fname="freq_dist_retrieval.png")
    
    print(f"{kwargs['add_n_hard_negatives']=}")
    train_dataset["negative"] = train_dataset.progress_apply(lambda x:  add_hard_negatives(
            query=x["query"], 
            positive=x["positive"], 
            negative=x["negative"], 
            corpus=all_passages_w_translations, 
            n=kwargs["add_n_hard_negatives"]), 
        axis=1)
    
    train_dataset["response"]   = None
    dev_dataset["response"]     = None
    print(f"{dev_dataset.head(2)=}")
    cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_retrieval_pretrain', cache_dir=kwargs["cache_dir"])
    trainer = DocumentGroundedDialogRetrievalTrainer(
        model=cache_path,
        train_dataset   = train_dataset.to_dict('records'),
        eval_dataset    = dev_dataset.to_dict('records'),
        all_passages=all_passages_w_translations,
        eval_passages=all_passages,
        lang_token  = kwargs["lang_token"],
        eval_lang   = kwargs["eval_lang"],
        save_output = kwargs["save_output"]
    )

    trainer.train(
        batch_size=128,
        total_epoches=50-2,
        accumulation_steps=kwargs["gradient_accumulation_steps"],
        loss_log_freq=1
        # per_gpu_batch_size=args.per_gpu_batch_size,
    )
    
    trainer.evaluate()
    
    # TODO add en-vi en-fr in case of translate-train
    extended_lang = langs - set(["fr", "vi"])
    if len(extended_lang) > 0:
        for lang_tag in extended_lang:
            combined_df = pd.concat([locals()[f"train_dataset_{lang_tag}"], locals()[f"dev_dataset_{lang_tag}"]]).reset_index()
            print(f'{combined_df.head(1)["response"]}=')
            trainer.save_dataset(dataset=combined_df.to_dict('records'), fname=f"{lang_tag}_{kwargs['extended_rerank_dataset_fname']}")


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)