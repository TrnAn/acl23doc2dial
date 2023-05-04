import os
import json
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--cache-dir", help= "Specifiy cache dir to save model to", type= str, default= "./")
    args = parser.parse_args()

    with open('dev.json') as f_in:
        with open('input.jsonl', 'w') as f_out:
            for line in f_in.readlines():
                sample = json.loads(line)
                sample['positive'] = ''
                sample['negative'] = ''
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

    with open('input.jsonl') as f:
        eval_dataset = [json.loads(line) for line in f.readlines()]

    all_passages = []
    for file_name in ['fr', 'vi']:
        with open(f'all_passages/{file_name}.json') as f:
            all_passages += json.load(f)

    cache_path = f'{args.cache_dir}./DAMO_ConvAI/nlp_convai_retrieval_pretrain'
    trainer = DocumentGroundedDialogRetrievalTrainer(
        model=cache_path,
        train_dataset=None,
        eval_dataset=eval_dataset,
        all_passages=all_passages
    )

    trainer.evaluate(
        checkpoint_path=os.path.join(trainer.model.model_dir,
                                    f'finetuned_model.bin'))


if __name__ == '__main__':
    main()