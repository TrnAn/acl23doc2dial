import os
import json
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--cache-dir", help= "Specifiy cache dir to save model to", type= str, default= ".")
    parser.add_argument("--lang-token", help= "Add language token <lang> to input", action=argparse.BooleanOptionalAction)
    parser.add_argument("--extended-dataset", help= "Run experiments on English and Chinese dataset", action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval-input-file", help= "File to read eval dataset (query, rerank, response) from", type=str, default=None)    
    parser.add_argument("--eval-lang", help= "Specify languages to evaluate results on", action='append', ngargs='+')
    args = parser.parse_args()

    with open('dev_no_lang_token.json') as f_in:
        with open(f'{args.cache_dir}/input.jsonl', 'w') as f_out:
            for line in f_in.readlines():
                sample = json.loads(line)
                # sample['positive'] = ''
                # sample['negative'] = ''
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

    with open(f'{args.cache_dir}/input.jsonl') as f:
        eval_dataset = [json.loads(line) for line in f.readlines()]

    all_passages = []
    for file_name in ['fr', 'vi']:
        with open(f'all_passages/{file_name}.json') as f:
            tmp = json.load(f)
            
            if args.lang_token:
                tmp = [f'<{file_name}> {s}' for s in tmp]
            
            all_passages += tmp

    cache_path = f'{args.cache_dir}/DAMO_ConvAI/nlp_convai_retrieval_pretrain'
    trainer = DocumentGroundedDialogRetrievalTrainer(
        model=cache_path,
        train_dataset=None,
        eval_dataset=eval_dataset,
        all_passages=all_passages,
        lang_token=args.lang_token
    )

    trainer.evaluate(
        checkpoint_path=os.path.join(trainer.model.model_dir,
                                    f'finetuned_model.bin'))


if __name__ == '__main__':
    main()