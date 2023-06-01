import os
import json
from modelscope.trainers.nlp.document_grounded_dialog_generate_trainer import \
    DocumentGroundedDialogGenerateTrainer
from train_generation import evaluate
import datetime
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--cache-dir", help= "Specifiy cache dir to save model to", type= str, default= ".")
    parser.add_argument("--lang-token", help= "Add language token <lang> to input", action=argparse.BooleanOptionalAction)
    parser.add_argument("--extended-dataset", help= "Run experiments on English and Chinese dataset", action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval-input-file", help= "File to read eval dataset (query, rerank, response) from", type=str, default=None)
    parser.add_argument("--only-english", help= "Run experiments only on English dataset", type= int, default=0)
    parser.add_argument("--only-chinese", help= "Run experiments only on Chinese dataset", type= int, default=0)
    parser.add_argument("--eval-lang", help= "Specify languages to evaluate results on",  action='store',type=str, nargs="+")
    args = parser.parse_args()

    parent_dir = "all_passages/lang_token" if args.lang_token else "all_passages"
    with open(f'{parent_dir}/id_to_passage.json') as f:
        id_to_passage = json.load(f)

    eval_dataset = []
    with open(f'{args.cache_dir}/rerank_output.jsonl') as f:
        for line in f.readlines():
            sample = json.loads(line)
            if sample["lang"] not in args.eval_lang:
                continue
            eval_dataset.append({
                'query': sample['input'],
                'rerank': json.dumps([id_to_passage[x['wikipedia_id']] for x in sample['output'][0]['provenance']],
                                    ensure_ascii=False),
                'response': sample['output'][0]['answer'], #'<response> @'
                'lang': sample["lang"]
            })

    cache_path = f'{args.cache_dir}/DAMO_ConvAI/nlp_convai_generation_pretrain'
    trainer = DocumentGroundedDialogGenerateTrainer(
        model=cache_path,
        train_dataset=None,
        eval_dataset=eval_dataset,
        lang_token=args.lang_token
    )

    evaluate(trainer, checkpoint_path=os.path.join(trainer.model.model_dir,
                                                f'finetuned_model.bin'))
    with open(f'{cache_path}/evaluate_result.json') as f:
        predictions = json.load(f)['outputs']

    eval_langs = "_".join(args.eval_lang)    
    with open(f'{args.cache_dir}/outputStandardFileBaseline_{eval_langs}.json', 'w') as f:
        for query, prediction in zip(eval_dataset, predictions):
            f.write(json.dumps({
                'query': query['query'],
                'gold_response': query['response'],
                'response': prediction, #.replace('<response>','').strip()
                'lang': query['lang']
            }, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()