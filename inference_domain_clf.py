from models.document_grounded_dialog_domain_clf_trainer import DocumentGroundedDialogDomainClfTrainer, XLMRobertaDomainClfHead
from utils.preprocessing import get_args
from transformers import AutoConfig
import json
from modelscope.models import Model
from modelscope.hub.snapshot_download import snapshot_download
import os
import re
import copy
import pandas as pd
from utils.preprocessing import get_unique_langs
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer


def main(**kwargs):
    langs = get_unique_langs(kwargs["eval_lang"])


    if kwargs["translate_mode"] == "test":
        retrieval_fnames = [f"ttest_{src_lang}2{kwargs['target_langs'][0]}.json" for src_lang in kwargs['source_langs']]
        print(retrieval_fnames)
        retrieval_paths  = [os.path.join(kwargs["cache_dir"], retrieval_fname) for retrieval_fname in retrieval_fnames]
    else:
        retrieval_paths  = [f'{kwargs["cache_dir"]}/{kwargs["eval_input_file"]}']

    for idx, retrieval_path in enumerate(retrieval_paths):
        filemode = "w" if idx == 0 else "a"
        with open(retrieval_path) as f_in:
            with open(f'{kwargs["cache_dir"]}/input.jsonl', filemode) as f_out:
                for line in f_in.readlines():
                    sample = json.loads(line)
                    f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

    with open(f'{kwargs["cache_dir"]}/input.jsonl') as f:
        eval_dataset = [json.loads(line) for line in f.readlines()]


    pattern = r"//(?!.*//).*?-(.*)$"
    eval_dataset  = [{"query": item["query"], 
                      "positive": item["positive"],
                      "negative": None,
                      "response": item["response"],
                      "domain": re.search(pattern, item["positive"]).group(1),
                      "lang": item["lang"]} for item in eval_dataset if re.search(pattern, item["positive"])]


    all_passages = []
    parent_dir = "all_passages/lang_token" if kwargs["lang_token"] else "all_passages"    
    for file_name in langs:
        with open(f'{parent_dir}/{file_name}.json') as f:
            all_passages += json.load(f)


    model_dir = 'DAMO_ConvAI/nlp_convai_retrieval_pretrain'
    cache_path = snapshot_download(model_dir, cache_dir=kwargs["cache_dir"])

    retrieval_trainer = DocumentGroundedDialogRetrievalTrainer(
        model           = cache_path,
        train_dataset   = None,
        eval_dataset    = eval_dataset,
        all_passages    = None,
        eval_passages   = all_passages,
        checkpoint_path = os.path.join(kwargs["cache_dir"], 
                                       model_dir,
                                       'finetuned_model.bin'),
        lang_token      = kwargs["lang_token"],
        eval_lang       = kwargs["eval_lang"],
        save_output     = kwargs["save_output"]
    )

    retrieval_trainer_copy = copy.deepcopy(retrieval_trainer)
    tokenizer_dir   = retrieval_trainer_copy.model.model_dir
    new_model       = retrieval_trainer_copy.model.model.qry_encoder

    config = AutoConfig.from_pretrained(os.path.join(tokenizer_dir, "qry_encoder"))
    config.add_pooling_layer = False


    labels = sorted(pd.DataFrame(eval_dataset)["domain"].unique())
    print(f"{labels=}")
    adapt_args = {
        "num_classes": len(labels), 
        "labels": labels
        }
    

    clf_model = XLMRobertaDomainClfHead(config=config, adapt_args=adapt_args, model=new_model)
    dev_dataset = pd.DataFrame(eval_dataset)
    # dev_dataset["negative"] = None
    print(dev_dataset.columns)
    trainer = DocumentGroundedDialogDomainClfTrainer(
        model           = copy.deepcopy(clf_model), 
        tokenizer_dir   = tokenizer_dir,
        checkpoint_path = os.path.join(tokenizer_dir, 'nlp_convai_domain_clf'), 
        train_dataset   = None, 
        eval_dataset    = dev_dataset.to_dict('records'),
        num_classes     = adapt_args["num_classes"],
        lang_token      = kwargs["lang_token"],
        eval_lang       = kwargs["eval_lang"]
        )
    
    trainer.evaluate()

    retrieval_trainer.evaluate()
    retrieval_trainer.evaluate_by_domain(trainer=trainer)


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)

