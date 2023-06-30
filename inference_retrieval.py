import os
import json
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
import datetime
import argparse
from utils.seed import set_seed
from utils.preprocessing import get_args, get_unique_langs, get_unique_passages
set_seed()


def main(**kwargs):
    langs = get_unique_langs(kwargs["eval_lang"])

    #TODO 
    # 1. also read from other file if translate -test 
    # 2. and concat json with prev json data to save input.jsonl

    if kwargs["translate_mode"] == "test":
        retrieval_fname = f"ttest_{'_'.join(kwargs['source_langs'])}2{kwargs['target_langs'][0]}.json"
        retrieval_path  = os.path.join(kwargs["cache_dir"], retrieval_fname)
    else:
        retrieval_path  = f'{kwargs["cache_dir"]}/{kwargs["eval_input_file"]}'

    with open(retrieval_path) as f_in:
        with open(f'{kwargs["cache_dir"]}/input.jsonl', 'w') as f_out:
            for line in f_in.readlines():
                sample = json.loads(line)
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

    with open(f'{kwargs["cache_dir"]}/input.jsonl') as f:
        eval_dataset = [json.loads(line) for line in f.readlines()]

    # TODO: add translated vi -> en fr -> en to all_passages
    # replace native lang wth translations vi -> vi2en; fr -> fr2en
    # all_passages = []
    # for file_name in langs:
    #     with open(f'{parent_dir}/{file_name}.json') as f:
    #         all_passages += json.load(f)
    parent_dir = "all_passages/lang_token" if kwargs["lang_token"] else "all_passages"

    # if kwargs["translate_mode"] == "test":
    #     langs += [f"{src_lang}2{kwargs['target_langs'][0]}" for src_lang in kwargs["source_langs"]]

    all_passages = []
    translated_passages = []
    
    for lang in langs:
        with open(f'{parent_dir}/{lang}.json', encoding='utf-8-sig') as f:
            all_passages += json.load(f)

            # if kwargs["translate_mode"] == "train":
            #     translated_passages += get_unique_passages(locals()[f"train_dataset_{lang}"], lang=lang) if kwargs["lang_token"] else get_unique_passages(locals()[f"train_dataset_{lang}"])

    # add source lang -> target_lang translations (monolingual setting)
    if kwargs["translate_mode"] == "test":
        for src_lang in kwargs["source_langs"]:
            with open(f'{kwargs["cache_dir"]}/{parent_dir}/{src_lang}2{kwargs["target_langs"][0]}.json', encoding='utf-8-sig') as f:
                all_passages += json.load(f)    

    # all_passages_w_translations =  list(set(all_passages) | set(translated_passages))

    # print(f"{all_passages_w_translations=}")
    cache_path = f'{kwargs["cache_dir"]}/DAMO_ConvAI/nlp_convai_retrieval_pretrain'
    trainer = DocumentGroundedDialogRetrievalTrainer(
        model=cache_path,
        train_dataset=None,
        eval_dataset=eval_dataset,
        all_passages=None,
        eval_passages=all_passages,
        lang_token=kwargs["lang_token"],
        eval_lang = kwargs["eval_lang"],
        save_output=kwargs["save_output"]
    )

    trainer.evaluate(
        checkpoint_path=os.path.join(trainer.model.model_dir,
                                    f'finetuned_model.bin'))


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)