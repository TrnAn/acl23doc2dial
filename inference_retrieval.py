import os
import json
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
import datetime
import argparse
from utils.seed import set_seed
set_seed()


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--cache-dir", help= "Specifiy cache dir to save model to", type= str, default= ".")
    parser.add_argument("--lang-token", help= "Add language token <lang> to input", action=argparse.BooleanOptionalAction)
    parser.add_argument("--only-english", help= "Run experiments only on English dataset", type= int, default=0)
    parser.add_argument("--only-chinese", help= "Run experiments only on Chinese dataset", type= int, default=0)
    parser.add_argument("--extended-dataset", help= "Run experiments on English and Chinese dataset", action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval-input-file", help= "File to read eval dataset (query, rerank, response) from", type=str, default=None)    
    parser.add_argument("--eval-lang", help= "Specify languages to evaluate results on", action='store', type=str, nargs="+")
    args = parser.parse_args()

    with open(args.eval_input_file) as f_in:
        with open(f'{args.cache_dir}/input.jsonl', 'w') as f_out:
            for line in f_in.readlines():
                sample = json.loads(line)
                # sample['positive'] = ''
                # sample['negative'] = ''
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

    with open(f'{args.cache_dir}/input.jsonl') as f:
        eval_dataset = [json.loads(line) for line in f.readlines()]

    parent_dir = "all_passages/lang_token" if args.lang_token else "all_passages"
    all_passages = []

    languages= []
    if not bool(args.only_english):
        languages += ['fr', 'vi']

    if args.extended_dataset:
        # if not bool(args.only_chinese):
        languages += ['en']
        # if not bool(args.only_english):
        #     languages += ['cn']

    for file_name in languages:
        with open(f'{parent_dir}/{file_name}.json') as f:
            all_passages += json.load(f)

    cache_path = f'{args.cache_dir}/DAMO_ConvAI/nlp_convai_retrieval_pretrain'
    trainer = DocumentGroundedDialogRetrievalTrainer(
        model=cache_path,
        train_dataset=None,
        eval_dataset=eval_dataset,
        all_passages=all_passages,
        lang_token=args.lang_token,
        eval_lang = [args.eval_lang]
    )

    trainer.evaluate(
        checkpoint_path=os.path.join(trainer.model.model_dir,
                                    f'finetuned_model.bin'))


if __name__ == '__main__':
    main()