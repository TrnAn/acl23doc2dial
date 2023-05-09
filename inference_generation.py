import os
import json
from modelscope.trainers.nlp.document_grounded_dialog_generate_trainer import \
    DocumentGroundedDialogGenerateTrainer
from train_generation import evaluate
import datetime
import argparse


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--cache-dir", help= "Specifiy cache dir to save model to", type= str, default= "./")
    parser.add_argument("--lang-token", help= "Add language token <lang> to input", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    with open('all_passages/id_to_passage.json') as f:
        id_to_passage = json.load(f)

    eval_dataset = []
    with open(f'{args.cache_dir}/rerank_output.jsonl') as f:
        for line in f.readlines():
            sample = json.loads(line)
            print(f"{sample=}")

            eval_dataset.append({
                'query': sample['input'],
                'rerank': json.dumps([id_to_passage[x['wikipedia_id']] for x in sample['output'][0]['provenance']],
                                    ensure_ascii=False),
                'response': sample['output'] #'<response> @'
            })

    cache_path = f'{args.cache_dir}/DAMO_ConvAI/nlp_convai_generation_pretrain'
    trainer = DocumentGroundedDialogGenerateTrainer(
        model=cache_path,
        train_dataset=None,
        eval_dataset=eval_dataset,
    )
    evaluate(trainer, checkpoint_path=os.path.join(trainer.model.model_dir,
                                                f'finetuned_model.bin'))
    with open(f'{cache_path}/evaluate_result.json') as f:
        predictions = json.load(f)['outputs']

    with open(f'{args.cache_dir}/outputStandardFileBaseline.json', 'w') as f:
        for query, prediction in zip(eval_dataset, predictions):
            f.write(json.dumps({
                'query': query['query'],
                'response': prediction.replace('<response>','').strip()
            }, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()