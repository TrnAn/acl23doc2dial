import os
import json
import pandas as pd
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
import utils.preprocessing as preprocessing
from utils.seed import set_seed
from utils.preprocessing import get_args, add_translation2trainset, get_unique_passages
set_seed()


def main(**kwargs):
    fr_train_dataset, vi_train_dataset, en_train_dataset, cn_train_dataset = None, None, None, None
    langs = set(item for sublist in kwargs["eval_lang"] for item in sublist)

    if "fr" in langs:
        fr_train_dataset = preprocessing.read('DAMO_ConvAI/FrDoc2BotRetrieval')
        fr_train_dataset["lang"] = "fr"
    if "vi" in langs:
        vi_train_dataset = preprocessing.read('DAMO_ConvAI/ViDoc2BotRetrieval')
        vi_train_dataset["lang"] = "vi" 
    if "en" in langs:
        en_train_dataset = pd.read_json("en_train_dataset_retrieval_generation_in_domain.json", lines=True)
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

        # ttrain_vi = pd.read_json(f"ttrain_retrieval_vi.json", lines=True)
        # ttrain_vi["lang"] = "vi" 
        # old_size = len(train_dataset_vi)
        # train_dataset_vi = pd.concat([train_dataset_vi, ttrain_vi])
        # train_dataset_vi = train_dataset_vi.reset_index(drop=True)
        # new_size = len(train_dataset_vi)
        # print(f"added {new_size - old_size} new datapoints to vi train dataset...")

        # ttrain_fr = pd.read_json(f"ttrain_retrieval_fr.json", lines=True)
        # ttrain_fr["lang"] = "fr" 
        # old_size = len(train_dataset_fr)
        # train_dataset_fr = pd.concat([train_dataset_fr, ttrain_fr])
        # train_dataset_fr = train_dataset_fr.reset_index(drop=True)
        # new_size = len(train_dataset_fr)
        # print(f"added {new_size - old_size} new datapoints to fr train dataset...")
    

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
    for lang in langs:
        train, dev = lang_dd[lang]
        train_dataset.append(train)
        dev_dataset.append(dev)
        
    train_dataset   = pd.concat(train_dataset) 
    dev_dataset     = pd.concat(dev_dataset)

    preprocessing.save_to_json(dev_dataset, dev_dataset.columns, fname=kwargs["eval_input_file"], pdir=kwargs["cache_dir"])

    parent_dir = "all_passages/lang_token" if kwargs["lang_token"] else "all_passages"
    all_passages = []
    for file_name in langs:
        with open(f'{parent_dir}/{file_name}.json') as f:
            all_passages += json.load(f)

            if kwargs["translate_mode"] == "train":
                unique_passages = get_unique_passages(locals()[f"train_dataset_{lang}"], lang=lang) if kwargs["lang_token"] else get_unique_passages(locals()[f"train_dataset_{lang}"])
                all_passages =  list(set(all_passages) | set(unique_passages))


    if kwargs["batch_accumulation"]:
        kwargs["gradient_accumulation_steps"] = kwargs["batch_size"] // (kwargs["num_devices"] * kwargs["per_gpu_batch_size"])

    print(f"BATCH SIZE: {kwargs['per_gpu_batch_size']}")
    cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_retrieval_pretrain', cache_dir=kwargs["cache_dir"])
    trainer = DocumentGroundedDialogRetrievalTrainer(
        model=cache_path,
        train_dataset=train_dataset.to_dict('records'),
        eval_dataset=dev_dataset.to_dict('records'),
        all_passages=all_passages,
        lang_token  = kwargs["lang_token"],
        eval_lang   = kwargs["eval_lang"],
        save_output = kwargs["save_output"]
    )

    trainer.train(
        batch_size=128,
        total_epoches=1,
        accumulation_steps=kwargs["gradient_accumulation_steps"],
        loss_log_freq=1
        # per_gpu_batch_size=args.per_gpu_batch_size,
    )
    trainer.evaluate(
        checkpoint_path=os.path.join(trainer.model.model_dir,
                                    'finetuned_model.bin'))
    
    extended_lang = langs - set(["fr", "vi"])
    if len(extended_lang) > 0:
        for lang_tag in extended_lang:
            combined_df = pd.concat([train_dataset_en, dev_dataset_en]).reset_index()
            trainer.save_dataset(dataset=combined_df.to_dict('records'), fname=f"{lang_tag}_{kwargs['extended_rerank_dataset_fname']}")


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)