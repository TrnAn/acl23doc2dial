import os
import json
from modelscope.trainers.nlp.document_grounded_dialog_generate_trainer import \
    DocumentGroundedDialogGenerateTrainer
from train_generation import evaluate
import datetime
import argparse
import pandas as pd
from utils.seed import set_seed
from utils.preprocessing import get_args, get_unique_langs
set_seed()


def main(**kwargs):
    langs = get_unique_langs(kwargs["eval_lang"])
    # parent_dir = "all_passages/lang_token" if kwargs["lang_token"] else "all_passages"
    # with open(f'{parent_dir}/id_to_passage.json', encoding="utf-8") as f:
    #     id_to_passage = json.load(f)

    # new_key = max(map(int, id_to_passage.keys())) + 1

    # if kwargs["translate_mode"] == "train":
    id_to_passage = {}
    with open(f'{kwargs["cache_dir"]}/rerank_output.jsonl', encoding="utf-8-sig") as f_in:
        for line in f_in.readlines():
            sample = json.loads(line)
       
            # add translated passages to id_to_passage mapping
            # if sample["lang"] in kwargs["target_langs"]:
            for passage_dict in sample["passages"]:
                id_to_passage[passage_dict["pid"]] = passage_dict["text"]


    eval_dataset = []
    with open(f'{kwargs["cache_dir"]}/rerank_output.jsonl', encoding="utf-8-sig") as f_in:
        for line in f_in.readlines():
            # 
            sample = json.loads(line)
            if sample["langs"] not in langs:
                continue
            eval_dataset.append({
                'query': sample['input'],
                'rerank': eval(json.dumps([id_to_passage[x['wikipedia_id']] for x in sample['output'][0]['provenance']],
                                    ensure_ascii=False)),
                'response': sample["responses"], # sample['output'][0]['answer'], #'<response> @'
                'lang': sample["langs"]
            })
    
    kwargs["is_inference"] = True
    cache_path = f'{kwargs["cache_dir"]}/DAMO_ConvAI/nlp_convai_generation_pretrain'
    trainer = DocumentGroundedDialogGenerateTrainer(
        model=cache_path,
        train_dataset=None,
        eval_dataset=eval_dataset,
        lang_token=kwargs["lang_token"],
        eval_lang=kwargs["eval_lang"],    
        translate_mode=kwargs["translate_mode"],
        is_inference = kwargs["is_inference"]
    )

    evaluate(trainer, checkpoint_path=os.path.join(trainer.model.model_dir,
                                                f'finetuned_model.bin'), eval_lang=kwargs["eval_lang"])
    eval_langs = "_".join(langs)  
    fname= f"{eval_langs}_evaluate_result.json"
    with open(f'{cache_path}/{fname}') as f:
        predictions = json.load(f)['outputs']

      
    with open(f'{kwargs["cache_dir"]}/outputStandardFileBaseline_{eval_langs}.json', 'w') as f:
        for query, prediction in zip(eval_dataset, predictions):
            f.write(json.dumps({
                'query': query['query'],
                'gold_response': query['response'],
                'response': prediction, #.replace('<response>','').strip()
                'lang': query['lang']
            }, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)