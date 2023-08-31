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
from utils.preprocessing import get_args, get_unique_langs
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
            file_out.write(json.dumps(every_dict, ensure_ascii=False) + '\n')


def main(**kwargs):
    model_dir = f'./{kwargs["cache_dir"]}/output'
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
    kwargs.update({
        'output': model_dir,
        'max_batch_size': 64,
        'exclude_instances': '',
        'include_passages': True, #False,
        'do_lower_case': True,
        'max_seq_length': 512,
        'query_length': 195,
        'tokenizer_resize': True,
        'model_resize': True,
        'kilt_data': True
    })
    
    langs = get_unique_langs(kwargs["eval_lang"])
    # kwargs["eval_lang"] = kwargs["eval_lang"][0]

    model = Model.from_pretrained(model_dir, **kwargs)
    mypreprocessor = DocumentGroundedDialogRerankPreprocessor(
        model.model_dir, **kwargs)

    pipeline_ins = myDocumentGroundedDialogRerankPipeline(
        model=model, preprocessor=mypreprocessor, **kwargs)

    # file_in = open(f'./{kwargs["cache_dir"]}/input.jsonl', 'r', encoding="utf-8-sig")
    # all_querys = []
    # for every_query in file_in:
    #     all_querys.append(json.loads(every_query))

    passage_to_id = {}
    ptr = -1
    parent_dir = "all_passages/lang_token" if kwargs["lang_token"] else "all_passages"

    # replace native lang wth translations vi -> vi2en; fr -> fr2en
    if kwargs["translate_mode"] in ["train", "test"]:
        passage_langs =  kwargs["source_langs"] + kwargs["target_langs"]
    else:
        passage_langs = langs
    
    
    passage_langs = ["fr", "vi", "en", "cn"]
    for file_name in passage_langs:
        with open(f'./{parent_dir}/{file_name}.json', encoding="utf-8-sig") as f:
            all_passages = json.load(f)
            for every_passage in all_passages:
                ptr += 1
                passage_to_id[every_passage.strip()] = str(ptr)


    if kwargs["translate_mode"] == "test":
         for src_lang in kwargs["source_langs"]:
            with open(f'{kwargs["cache_dir"]}/{parent_dir}/{src_lang}2{kwargs["target_langs"][0]}.json', encoding="utf-8-sig") as f:
                all_passages = json.load(f)
                for every_passage in all_passages:
                    ptr += 1
                    passage_to_id[every_passage.strip()] = str(ptr)

    # translate evaluate_result
    with open(f'{kwargs["cache_dir"]}/DAMO_ConvAI/nlp_convai_retrieval_pretrain/evaluate_result.json', encoding="utf-8-sig") as file_in:
        retrieval_file = json.load(file_in)

    for lang in kwargs["eval_lang"]:
        retrieval_result    = retrieval_file['outputs'] # predicted
        retrieval_targets   = retrieval_file['targets']
        queries             = retrieval_file['queries']
        languages           = retrieval_file['langs']
        responses           = retrieval_file['response']
        input_list = []
        passages_list = []
        lang_list = []
        ids_list = []
        output_list = []
        positive_pids_list = []
        responses_list = []
        # ptr = -1

        # for ptr, x in tqdm(enumerate(all_querys)):
        for idx, (query, language, all_candidates, target, response) in enumerate(zip(queries, languages, retrieval_result, retrieval_targets, responses)):
            # ptr += 1
            # now_id = str(ptr)
            # now_input = x
            # now_positive = now_input["positive"]
            now_wikipedia = []
            now_passages = []

            if language not in lang:
                continue
            
            # retrieval_idx = retrieval_targets.index(now_positive)
        
            # all_candidates = retrieval_result[ptr]
            # all_candidates = retrieval_result[retrieval_idx]
            # target = retrieval_targets[ptr]
            # target = retrieval_targets[retrieval_idx]
            
            
            for every_passage in all_candidates:
                get_pid = passage_to_id[every_passage.strip()]
                get_positive_pid = passage_to_id[target.strip()]
                now_wikipedia.append({'wikipedia_id': str(get_pid)})
                now_passages.append({"pid": str(get_pid), "title": "", "text": every_passage})
            now_output = [{'answer': target, 'provenance': now_wikipedia}]


            input_list.append(query)
            passages_list.append(str(now_passages))
            ids_list.append(idx)
            output_list.append(str(now_output))
            lang_list.append(language)
            positive_pids_list.append(json.dumps([get_positive_pid]))
            responses_list.append(response)
            # input_list.append(now_input['query'])
            # passages_list.append(str(now_passages))
            # ids_list.append(now_id)
            # output_list.append(str(now_output))
            # lang_list.append(now_input["lang"])
            # positive_pids_list.append(json.dumps([get_positive_pid]))
        

        evaluate_dataset = {'input': input_list, 'id': ids_list, 'passages': passages_list, 'output': output_list,
                            'positive_pids': positive_pids_list, 'langs': lang_list, 'responses': responses_list}

        pipeline_ins(evaluate_dataset)

        if sorted(lang) == sorted(langs):
            print(f"save rerank_output.jsonl for {lang=}...")
            pipeline_ins.save(f'./{kwargs["cache_dir"]}/rerank_output.jsonl')


if __name__ == '__main__':
    kwargs = {}
    kwargs.update(get_args())
    main(**kwargs)