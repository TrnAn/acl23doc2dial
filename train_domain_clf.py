from models.document_grounded_dialog_domain_clf_trainer import DocumentGroundedDialogDomainClfTrainer, XLMRobertaDomainClfHead
from utils import preprocessing
from utils.preprocessing import get_args
from transformers import AutoConfig
import pandas as pd
from modelscope.models import Model
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
import os
import json
import copy
import utils.preprocessing as preprocessing
from utils.seed import set_seed
from utils.preprocessing import get_args
import utils.data_exploration as exploration
set_seed()
SEED = 42

def main(**kwargs):
    print("start domain classification in retrieval step...")
    fr_train_dataset, vi_train_dataset = None, None

    langs = set(kwargs["source_langs"] + kwargs["target_langs"]) if kwargs["translate_mode"] == "test" else set(item for sublist in kwargs["eval_lang"] for item in sublist) 

    if "fr" in langs:
        fr_train_dataset = preprocessing.read('DAMO_ConvAI/FrDoc2BotRetrieval')
        fr_train_dataset["lang"] = "fr"

    if "vi" in langs:
        vi_train_dataset = preprocessing.read('DAMO_ConvAI/ViDoc2BotRetrieval')
        vi_train_dataset["lang"] = "vi" 


    train_dataset_fr, dev_dataset_fr = preprocessing.test_split(fr_train_dataset)
    train_dataset_vi, dev_dataset_vi = preprocessing.test_split(vi_train_dataset)
    
    if kwargs["lang_token"]:
        train_dataset_fr    = preprocessing.add_lang_token(train_dataset_fr, "fr", colnames=["query", "positive", "negative"]) 
        train_dataset_vi    = preprocessing.add_lang_token(train_dataset_vi, "vi", colnames=["query", "positive", "negative"])

        dev_dataset_fr = preprocessing.add_lang_token(dev_dataset_fr, "fr", colnames=["query", "positive", "negative"]) 
        dev_dataset_vi = preprocessing.add_lang_token(dev_dataset_vi, "vi", colnames=["query", "positive", "negative"]) 

    
    train_dataset   = pd.concat([train_dataset_fr, train_dataset_vi]) 
    dev_dataset     = pd.concat([dev_dataset_fr, dev_dataset_vi]) 

    train_dataset["domain"] = train_dataset["positive"].str.extract(r"//(?!.*//).*?-(.*)$")
    dev_dataset["domain"]   = dev_dataset["positive"].str.extract(r"//(?!.*//).*?-(.*)$")

    model_dir = 'DAMO_ConvAI/nlp_convai_retrieval_pretrain'
    cache_path = snapshot_download(model_dir, cache_dir=kwargs["cache_dir"])

    parent_dir = "all_passages/lang_token" if kwargs["lang_token"] else "all_passages"
    all_passages = []
    
    if kwargs["translate_mode"] == "train":
        langs = set(list(langs) + kwargs["source_langs"])

    for file_name in langs:
        with open(f'{parent_dir}/{file_name}.json') as f:
            all_passages += json.load(f)

    freq_df = exploration.get_freq_df(train_dataset, dev_dataset)
    exploration.plot_freq(freq_df, plot_dir=f'{kwargs["cache_dir"]}/plot', fname="freq_dist_retrieval.png")

    train_dataset["response"]   = None
    dev_dataset["response"]     = None
    retrieval_trainer = DocumentGroundedDialogRetrievalTrainer(
        model           = cache_path,
        train_dataset   = train_dataset.to_dict('records'),
        eval_dataset    = dev_dataset.to_dict('records'),
        all_passages    = all_passages,
        eval_passages   = all_passages,
        checkpoint_path = os.path.join(kwargs["cache_dir"], 
                                       model_dir,
                                       'finetuned_model.bin'),
        lang_token      = kwargs["lang_token"],
        eval_lang       = kwargs["eval_lang"],
        save_output     = kwargs["save_output"]
    )
    
    retrieval_trainer.train(
        batch_size=128,
        total_epoches=50,
        accumulation_steps=kwargs["gradient_accumulation_steps"],
        loss_log_freq=1
    )

    retrieval_trainer_copy = copy.deepcopy(retrieval_trainer)
    tokenizer_dir   = retrieval_trainer_copy.model.model_dir
    new_model       = retrieval_trainer_copy.model.model.qry_encoder

    config = AutoConfig.from_pretrained(os.path.join(tokenizer_dir, "qry_encoder"))
    config.add_pooling_layer = False

    labels = sorted(pd.concat([train_dataset, dev_dataset])["domain"].unique())
    print(f"{labels=}")

    adapt_args = {
        "num_classes": len(labels), 
        "labels": labels
        }

    clf_model = XLMRobertaDomainClfHead(config=config, adapt_args=adapt_args, model=new_model)
    
    trainer = DocumentGroundedDialogDomainClfTrainer(
        model           = copy.deepcopy(clf_model), 
        tokenizer_dir   = tokenizer_dir,
        checkpoint_path = os.path.join(tokenizer_dir, 'nlp_convai_domain_clf'), 
        train_dataset   = train_dataset.to_dict('records'), 
        eval_dataset    = dev_dataset.to_dict('records'),
        num_classes     = adapt_args["num_classes"],
        lang_token      = kwargs["lang_token"],
        eval_lang       = kwargs["eval_lang"]
        )
    
    
    trainer.train(total_epoches=10)
    trainer.evaluate()

    retrieval_trainer.evaluate()
    retrieval_trainer.evaluate_by_domain(trainer=trainer)


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)

