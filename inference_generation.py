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
    parser.add_argument("--eval-lang", help= "Specify languages to evaluate results on", action='append', ngargs='+')
    args = parser.parse_args()


    with open('all_passages/id_to_passage.json') as f:
        id_to_passage = json.load(f)
    
    
    if args.extended_dataset:
        if args.eval_input_file is None:
            raise Exception("Please specify arg --eval-input-file to read eval dataset from")
        eval_dataset = pd.read_json(args.eval_input_file, lines=True)
        eval_dataset = eval_dataset[eval_dataset.lang.isin(args.eval_lang)]
        eval_dataset = eval_dataset.to_dict('records')
    else:
        eval_dataset = []
        with open(f'{args.cache_dir}/rerank_output.jsonl') as f:
            for line in f.readlines():
                sample = json.loads(line)
                eval_dataset.append({
                    'query': sample['input'],
                    'rerank': json.dumps([id_to_passage[x['wikipedia_id']] for x in sample['output'][0]['provenance']],
                                        ensure_ascii=False),
                    'response': sample['output'][0]['answer'] #'<response> @'
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
                'response': prediction.replace('<response>','').strip()
            }, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()