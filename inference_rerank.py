import torch
import os
import json
from tqdm import tqdm
from modelscope.models import Model
from modelscope.models.nlp import DocumentGroundedDialogRerankModel
from modelscope.pipelines.nlp import DocumentGroundedDialogRerankPipeline
from modelscope.preprocessors.nlp import \
    DocumentGroundedDialogRerankPreprocessor
from modelscope.trainers.nlp.document_grounded_dialog_rerank_trainer import \
    DocumentGroundedDialogRerankTrainer
from typing import Union
import argparse
from utils.seed import set_seed
set_seed()


class myDocumentGroundedDialogRerankPipeline(DocumentGroundedDialogRerankPipeline):
    def __init__(self,
                 model: Union[DocumentGroundedDialogRerankModel, str],
                 preprocessor: DocumentGroundedDialogRerankPreprocessor = None,
                 config_file: str = None,
                 device: str = 'cuda',
                 auto_collate=True,
                 seed: int = 88,
                 **kwarg):
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate,
            seed=seed,
            **kwarg
        )

    def save(self, addr):
        file_out = open(addr, 'w')
        for every_dict in self.guess:
            file_out.write(json.dumps(every_dict) + '\n')


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--cache-dir", help= "Specifiy cache dir to save model to", type= str, default= ".")
    parser.add_argument("--only-english", help= "Run experiments only on English dataset", type= int, default=0)
    parser.add_argument("--only-chinese", help= "Run experiments only on Chinese dataset", type= int, default=0)
    parser.add_argument("--lang-token", help= "Add language token <lang> to input", action=argparse.BooleanOptionalAction)
    parser.add_argument("--extended-dataset", help= "Run experiments on English and Chinese dataset", action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval-input-file", help= "File to read eval dataset (query, rerank, response) from", type=str, default=None)
    parser.add_argument("--eval-lang", help= "Specify languages to evaluate results on",  action='store',type=str, nargs="+")
    args = vars(parser.parse_args())

    model_dir = f'./{args["cache_dir"]}/output'
    model_configuration = {
        "framework": "pytorch",
        "task": "document-grounded-dialog-rerank",
        "model": {
            "type": "doc2bot"
        },
        "pipeline": {
            "type": "document-grounded-dialog-rerank"
        },
        "preprocessor": {
            "type": "document-grounded-dialog-rerank"
        }
    }
    file_out = open(f'{model_dir}/configuration.json', 'w')
    json.dump(model_configuration, file_out, indent=4)
    file_out.close()
    args.update({
        'output': model_dir,
        'max_batch_size': 64,
        'exclude_instances': '',
        'include_passages': False,
        'do_lower_case': True,
        'max_seq_length': 512,
        'query_length': 195,
        'tokenizer_resize': True,
        'model_resize': True,
        'kilt_data': True,
        'cache_dir': args["cache_dir"],
        'seed': 42,
        'tokenizer_name': '',
        'instances_size': 1,
        'output_dir': f'{args["cache_dir"]}/output',
        'max_num_seq_pairs_per_device': 32,
        'full_train_batch_size': 32,
        'gradient_accumulation_steps': 32,
        'per_gpu_train_batch_size': 1,
        'num_train_epochs': 1, # 10,
        'train_instances': -1,
        'learning_rate': 2e-5,
        'max_seq_length': 512,
        'num_labels': 2,
        'fold': '',  # IofN
        'doc_match_weight': 0.0,
        'query_length': 195,
        'resume_from': '',  # to resume training from a checkpoint
        'config_name': '',
        'do_lower_case': True,
        'weight_decay': 0.0,  # previous default was 0.01
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'warmup_instances': 0,  # previous default was 0.1 of total
        'warmup_fraction': 0.0,  # only applies if warmup_instances <= 0
        'no_cuda': False,
        'n_gpu': 1,
        'seed': 42,
        'fp16': False,
        'fp16_opt_level': 'O1',  # previous default was O2
        'per_gpu_eval_batch_size': 8,
        'log_on_all_nodes': False,
        'world_size': 1,
        'global_rank': 0,
        'local_rank': -1,
        'tokenizer_resize': True,
        'model_resize': True
    })
    model = Model.from_pretrained(model_dir, **args)
    mypreprocessor = DocumentGroundedDialogRerankPreprocessor(
        model.model_dir, **args)

    pipeline_ins = myDocumentGroundedDialogRerankPipeline(
        model=model, preprocessor=mypreprocessor, **args)

    file_in = open(f'./{args["cache_dir"]}/input.jsonl', 'r')
    all_querys = []
    for every_query in file_in:
        all_querys.append(json.loads(every_query))
        
    # passage_to_id = {}
    # ptr = -1
    # languages = ['fr', 'vi', 'en', 'cn'] if args["extended_dataset"] else ['fr', 'vi']
    # for file_name in languages:
    #     with open(f'./all_passages/{file_name}.json') as f:
    #         all_passages = json.load(f)
    #         if args["lang_token"]:
    #             all_passages = [f"<{file_name}> " + passage for passage in all_passages]
    #         for every_passage in all_passages:
    #             ptr += 1
    #             passage_to_id[every_passage] = str(ptr)

    passage_to_id = {}
    ptr = -1
    parent_dir = "all_passages/lang_token" if args["lang_token"] else "all_passages"
    
    languages = []
    if not bool(args["only_english"]):
        languages += ['fr', 'vi']
        
    if args["extended_dataset"]:
        if not bool(args["only_chinese"]):
            languages += ['en']
        # if not bool(args["only_english"]):
        #     languages += ['cn']

    for file_name in languages:
        with open(f'./{parent_dir}/{file_name}.json') as f:
            all_passages = json.load(f)
            for every_passage in all_passages:
                ptr += 1
                passage_to_id[every_passage] = str(ptr)

    file_in = open(f'{args["cache_dir"]}/DAMO_ConvAI/nlp_convai_retrieval_pretrain/evaluate_result.json', 'r')
    retrieval_file      = json.load(file_in)
    retrieval_result    = retrieval_file['outputs'] # predicted
    retrieval_targets   = retrieval_file['targets']
    input_list = []
    passages_list = []
    lang_list = []
    ids_list = []
    output_list = []
    positive_pids_list = []
    ptr = -1
    for x in tqdm(all_querys):
        ptr += 1
        now_id = str(ptr)
        now_input = x
        now_wikipedia = []
        now_passages = []
 
        all_candidates = retrieval_result[ptr]
        target = retrieval_targets[ptr]
        
        for idx, every_passage in enumerate(all_candidates):
            get_pid = passage_to_id[every_passage]
            get_positive_pid = passage_to_id[target]
            now_wikipedia.append({'wikipedia_id': str(get_pid)})
            now_passages.append({"pid": str(get_pid), "title": "", "text": every_passage})
        now_output = [{'answer': target, 'provenance': now_wikipedia}]

        input_list.append(now_input['query'])
        passages_list.append(str(now_passages))
        ids_list.append(now_id)
        output_list.append(str(now_output))
        lang_list.append(now_input["lang"])
        # positive_pids_list.append(str([]))
        positive_pids_list.append(json.dumps([get_positive_pid]))
    

    evaluate_dataset = {'input': input_list, 'id': ids_list, 'passages': passages_list, 'output': output_list,
                        'positive_pids': positive_pids_list, 'lang': lang_list}


    dev_dataset = [
        {key: value for key, value in zip(evaluate_dataset.keys(), values)}
        for values in zip(*evaluate_dataset.values())
    ]

    args["device"] = "gpu"
    trainer = DocumentGroundedDialogRerankTrainer(
        model='DAMO_ConvAI/nlp_convai_ranking_pretrain', 
        train_dataset=[], 
        dev_dataset=dev_dataset, 
        args=args
        )

    trainer.evaluate(eval_lang=[args["eval_lang"]])
    pipeline_ins(evaluate_dataset)
    pipeline_ins.save(f'./{args["cache_dir"]}/rerank_output.jsonl')


if __name__ == '__main__':
    main()
