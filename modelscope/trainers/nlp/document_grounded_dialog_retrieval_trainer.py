# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
import faiss
import json
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
import itertools
from modelscope.metainfo import Trainers
from modelscope.models import Model
from modelscope.preprocessors import \
    DocumentGroundedDialogRetrievalPreprocessor
from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import ModeKeys
from modelscope.utils.logger import get_logger
from torchsummary import summary
from utils.preprocessing import save_to_json
import pandas as pd
import numpy as np
from typing import Union, Any, Dict
logger = get_logger()

def collate(batch):
    query = [item['query'] for item in batch]
    positive = [item['positive'] for item in batch]
    # negative = [item['negative'] for item in batch]
    negative = np.array([item['negative'] for item in batch]).ravel().tolist()

    lang = [item['lang'] for item in batch]
    return query, positive, negative, lang


def prepare_optimizer(model, lr, weight_decay, eps):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        weight_decay,
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
        0.0,
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    return optimizer


def prepare_scheduler(optimizer, epochs, steps_per_epoch, warmup_rate):
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    return scheduler


def measure_result(result_dict):
    recall_k = [1, 5, 10, 20]
    meters = {f'R@{k}': [] for k in recall_k}
    
    for output, target in zip(result_dict['outputs'], result_dict['targets']):
        for k in recall_k:
            # print(f"{target=}")
            # print(f"{output[:k]=}")

            if target in output[:k]:
                meters[f'R@{k}'].append(1)
            else:
                meters[f'R@{k}'].append(0)
    for k, v in meters.items():
        meters[k] = sum(v) / len(v)
    return meters


@TRAINERS.register_module(
    module_name=Trainers.document_grounded_dialog_retrieval_trainer)
class DocumentGroundedDialogRetrievalTrainer(EpochBasedTrainer):

    def __init__(self, model: str, revision='v1.0.0', *args, **kwargs):
        self.model = Model.from_pretrained(model, revision=revision)
        
        self.checkpoint_path = self.model.model_dir
 
        print(summary(self.model.model))
        self.preprocessor = DocumentGroundedDialogRetrievalPreprocessor(
            model_dir=self.model.model_dir, lang_token=kwargs["lang_token"])
        self.device = self.preprocessor.device
        self.eval_lang = kwargs["eval_lang"]
        if kwargs["lang_token"] is not None:
            self.model.model.qry_encoder.encoder.resize_token_embeddings(self.preprocessor.token_length)  # resize query encoder of DPR model
            self.model.model.ctx_encoder.encoder.resize_token_embeddings(self.preprocessor.token_length)  # resize context encoder of DPR model

        print(f"{self.checkpoint_path=}")

        model_fdir = os.path.join(self.model.model_dir,
                                          'finetuned_model.bin')

        if os.path.exists(model_fdir): 
            logger.info(f"load model: {model_fdir=}")
            state_dict = torch.load(model_fdir)
            self.model.model.load_state_dict(state_dict)


        self.model.model.to(self.device)
        self.train_dataset  = kwargs['train_dataset']
        self.eval_dataset   = kwargs['eval_dataset']
        self.all_passages   = kwargs['all_passages']
        self.eval_passages  = kwargs["eval_passages"]
        self.save_output    = kwargs['save_output']


    def train(self,
              total_epoches=20,
              batch_size=128,
              per_gpu_batch_size=32,
              accumulation_steps=1,
              learning_rate=2e-5,
              warmup_ratio=0.1,
              weight_decay=0.1,
              eps=1e-06,
              loss_log_freq=40):
        """
        Fine-tuning trainsets
        """
        # obtain train loader
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate)

        optimizer = prepare_optimizer(self.model.model, learning_rate,
                                      weight_decay, eps)
        steps_per_epoch = len(train_loader) // accumulation_steps
        scheduler = prepare_scheduler(optimizer, total_epoches,
                                      steps_per_epoch, warmup_ratio)

        best_score = 0.0
        for epoch in range(total_epoches):
            self.model.model.train()
            losses = []
            for index, payload in enumerate(tqdm.tqdm(train_loader)):
                query, positive, negative, _ = payload

                # print(f"before removing duplicates/FN: {len(negative)=}")
                # negative = list(filter(lambda x: x not in positive, negative))
                # print(f"after removing duplicates/FN: {len(negative)=}")
                processed = self.preprocessor(
                    {
                        'query': query,
                        'positive': positive,
                        'negative': negative
                    },
                    invoke_mode=ModeKeys.TRAIN)
                
                loss, logits = self.model.forward(processed)

                loss = loss / accumulation_steps

                loss.backward()

                if (index + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                losses.append(loss.item())
                if (index + 1) % loss_log_freq == 0:
                    logger.info(
                        f'epoch: {epoch} \t batch: {batch_size * index} \t loss: {sum(losses) / len(losses)}'
                    )
                    losses = []
            if losses:
                logger.info(
                    f'epoch: {epoch} \t batch: last \t loss: {sum(losses) / len(losses)}'
                )

            meters = self.evaluate(per_gpu_batch_size=per_gpu_batch_size)

            # total_score = sum([x for x in meters.values()])
            total_score = sum(sum(meter.values()) for meter in meters.values()) # score on all eval lang combiscancel nations
            logger.info(
                f'obtain max score: {total_score:.4f}')
            
            if total_score >= best_score:
                best_score = total_score
                model_path = os.path.join(self.model.model_dir,
                                          'finetuned_model.bin')
                state_dict = self.model.model.state_dict()
                torch.save(state_dict, model_path)
                logger.info(
                    'epoch %d obtain max score: %.4f, saving model to %s' %
                    (epoch, total_score, model_path))
                

    def evaluate(self, per_gpu_batch_size=32):
        """
        Evaluate testsets
        """

        # state_dict = torch.load(os.path.join(self.model.model_dir, "finetuned_model.bin"))
        # self.model.model.load_state_dict(state_dict)

        self.model.model.eval()
        with torch.no_grad():
            all_ctx_vector  = []
            all_passages    = []
            for mini_batch in tqdm.tqdm(
                    range(0, len(self.eval_passages), per_gpu_batch_size)):
                context = self.eval_passages[mini_batch:mini_batch
                                            + per_gpu_batch_size]
                processed = \
                    self.preprocessor({'context': context},
                                      invoke_mode=ModeKeys.INFERENCE,
                                      input_type='context')
                sub_ctx_vector = self.model.encode_context(
                    processed).detach().cpu().numpy()
                all_ctx_vector.append(sub_ctx_vector)
                all_passages += context

            all_ctx_vector = np.concatenate(all_ctx_vector, axis=0)
            all_ctx_vector = np.array(all_ctx_vector).astype('float32')
            faiss_index = faiss.IndexFlatIP(all_ctx_vector.shape[-1])
            faiss_index.add(all_ctx_vector) # context -> passages

            unique_langs = set(item for sublist in self.eval_lang for item in sublist) 
            logger.info(f"save passage embeddings to: {self.model.model_dir}/passage_embeddings_{'_'.join(unique_langs)}.txt...")
            passage_df = pd.DataFrame({"passage": all_passages, "embedding": all_ctx_vector.tolist()})
            passage_df.to_csv(f'{self.model.model_dir}/passage_embeddings_{"_".join(unique_langs)}.csv', index=False)

            self.retrieval_results = {}
            all_meters = {}
            for idx, lang in enumerate(self.eval_lang):
                results = {'queries': [], 'langs': [], 'outputs': [], 'targets': []}
                valid_loader = DataLoader(
                    dataset=self.eval_dataset,
                    batch_size=per_gpu_batch_size,
                    collate_fn=collate)
                for payload in tqdm.tqdm(valid_loader):
                    query, positive, negative, curr_lang = payload

                    if bool(set(curr_lang) & set(lang)) == 0: # language is not in current batch
                        continue

                    query, positive = zip(*[(q, p) for q, p, l in zip(query, positive, curr_lang) if l in lang])
                    query = list(query)
                    positive = list(positive)

                    processed = self.preprocessor({'query': query},
                                                invoke_mode=ModeKeys.INFERENCE)
                    query_vector = self.model.encode_query(
                        processed).detach().cpu().numpy().astype('float32')

                    D, Index = faiss_index.search(query_vector, 20)
                    results['outputs']  += [[
                        self.eval_passages[x] for x in retrieved_ids
                    ] for retrieved_ids in Index.tolist()]
                    results['targets']  += positive
                    results['queries']  += query
                    results['langs']    += curr_lang

                meters = measure_result(results)

                logger.info(f"{'_'.join(lang)} - {meters}")
                
                #(lang == self.eval_lang[0] and len(self.eval_lang) > 1) or 
                # if self.save_output:
                #     result_path = os.path.join(self.model.model_dir,
                #         f'evaluate_result.json')
                #     logger.info(f"saving evaluate_result.json...")

                #     with open(result_path, 'w') as f:
                #         json.dump(results, f, ensure_ascii=False, indent=4)

                all_meters["_".join(lang)] = meters

            logger.info(f"{all_meters=}")

        return all_meters


    def evaluate_by_domain(self, trainer, per_gpu_batch_size=32):
            
            def collate_retrieval(batch):
                query   = [item['query']    for item in batch]
                positive = [item['positive']  for item in batch]
                labels  = [item['domain']   for item in batch]
                langs   = [item['lang']    for item in batch]

                return query, positive, labels, langs
            
            """
            Evaluate testsets
            """

            trainer.model.eval()
            self.model.model.eval()
            with torch.no_grad():
                all_meters = {}
                for idx, lang in enumerate(self.eval_lang):
                    save_output = True if idx == 0 else False
                    
                    results = {'outputs': [], 'targets': []}
                    
                    for domain in trainer.model.labels:
                        results_domain = {'outputs': [], 'targets': []}
                        logger.info(f"evaluate domain: {domain} for {', '.join(lang)}")
                        
                        domain_queries, domain_passages = trainer.predict(
                            dataset=self.eval_dataset, 
                            filter_domain=domain, 
                            all_passages=self.eval_passages, 
                            per_gpu_batch_size=128)
                        
                        valid_loader = DataLoader(
                        dataset=domain_queries,
                        num_workers=4,
                        pin_memory=True,
                        batch_size=per_gpu_batch_size,
                        collate_fn=collate_retrieval)
        
                        domain_ctx_vector = []
                        for mini_batch in range(0, len(domain_passages), per_gpu_batch_size):
                            context = domain_passages[mini_batch:mini_batch
                                                        + per_gpu_batch_size]
                            processed = \
                                self.preprocessor({'context': context},
                                                invoke_mode=ModeKeys.INFERENCE,
                                                input_type='context')
                            sub_ctx_vector = self.model.encode_context(
                                processed).detach().cpu().numpy()
                            domain_ctx_vector.append(sub_ctx_vector)

                        domain_ctx_vector = np.concatenate(domain_ctx_vector, axis=0)
                        domain_ctx_vector = np.array(domain_ctx_vector).astype('float32')
                        faiss_index = faiss.IndexFlatIP(domain_ctx_vector.shape[-1])
                        faiss_index.add(domain_ctx_vector) # context -> passages

                        for payload in tqdm.tqdm(valid_loader):
                            queries, positives, domains, curr_langs = payload

                            if not bool(set(lang) & set(curr_langs)): # language is not in current batch
                                continue

                            processed = self.preprocessor({'query': queries},
                                                        invoke_mode=ModeKeys.INFERENCE)
                            query_vector = self.model.encode_query(
                                processed).detach().cpu().numpy().astype('float32')
                            D, Index = faiss_index.search(query_vector, 20)
                            
                            results_domain['outputs'] += [[
                                domain_passages[x] for x in retrieved_ids
                            ] for retrieved_ids in Index.tolist()]
                            results_domain['targets'] += positives
                            # print(f"{results_domain['outputs'][-1][0]=} = {positives[-1]=}")


                        if results_domain['outputs']:
                            meters = measure_result(results_domain)
                            logger.info(f"{'_'.join(lang+[domain])} - {meters}")

                            results["outputs"] += results_domain['outputs']
                            results["targets"] += results_domain['targets']

                            # result_path = os.path.join(self.model.model_dir, f'evaluate_result_{domain}.json')
                            # with open(result_path, 'w') as f:
                            #     logger.info(f"saving evaluate_result_{domain}.json...")
                            #     json.dump(results_domain, f, ensure_ascii=False, indent=4)

                    meters = measure_result(results)
                    if save_output:
                        result_path = os.path.join(self.model.model_dir,
                            f'evaluate_result.json')
                        logger.info(f"saving evaluate_result.json...")

                        with open(result_path, 'w') as f:
                            json.dump(results, f, ensure_ascii=False, indent=4)

                    all_meters["_".join(lang)] = meters
                    logger.info(f"final result: {all_meters=}")

            return all_meters


    def save_dataset(self, dataset: Union[list, Dict[str, Any]],  per_gpu_batch_size=32, fname:str="rerank_dataset.json"):
        retrieval_results = []

        with torch.no_grad():
            all_ctx_vector = []
            for mini_batch in tqdm.tqdm(
                    range(0, len(self.all_passages), per_gpu_batch_size)):
                context = self.all_passages[mini_batch:mini_batch
                                            + per_gpu_batch_size]
                processed = \
                    self.preprocessor({'context': context},
                                        invoke_mode=ModeKeys.INFERENCE,
                                        input_type='context')
                sub_ctx_vector = self.model.encode_context(
                    processed).detach().cpu().numpy()
                all_ctx_vector.append(sub_ctx_vector)

            all_ctx_vector = np.concatenate(all_ctx_vector, axis=0)
            all_ctx_vector = np.array(all_ctx_vector).astype('float32')

            faiss_index = faiss.IndexFlatIP(all_ctx_vector.shape[-1])
            faiss_index.add(all_ctx_vector) # context -> passages

            for jobj in tqdm.tqdm(dataset):
                keys = ["id", "input", "lang", "output", "passages", "positive_pids"]
                default_value = []
                tmp = dict.fromkeys(keys, default_value)

                index       = jobj["index"]
                lang        = jobj["lang"]
                query       = jobj["query"]
                positive    = jobj["positive"]
                response    = jobj["response"]
         
                processed       = self.preprocessor({'query': [query]}, invoke_mode=ModeKeys.INFERENCE)
                query_vector    = self.model.encode_query(processed).detach().cpu().numpy().astype('float32')
   
                # get ranked passages top-k
                _, Index = faiss_index.search(query_vector, 20)

                # if ranked passages do not contain positive pid -> add it as top 1
                ranked_indices = Index.tolist()[0]
                if self.all_passages.index(positive) not in ranked_indices:
                    ranked_indices = [self.all_passages.index(positive)] + Index.tolist()[0][:-1]
                
                tmp["id"] = index
                tmp["input"] = query
                tmp["lang"] = lang
                tmp["response"] = response
                tmp["positive_pids"] = json.dumps([str(self.all_passages.index(positive))])
                provenance = [{'wikipedia_id': str(retrieved_ids)}  for retrieved_ids in ranked_indices]
                
                tmp["output"]   = json.dumps([{'answer': '', 'provenance': provenance}])
                tmp["passages"] = json.dumps([{
                    'pid': str(retrieved_ids), 
                    'title':'', 
                    'text': self.all_passages[retrieved_ids]} for retrieved_ids in ranked_indices], ensure_ascii=False)
                retrieval_results.append(tmp)

            # Save the list of dictionaries as JSON with ensure_ascii=False
            path = os.path.join(self.model.model_dir, fname)
            logger.info(f"saving dataset for rerank training to {path}...")
            with open(path, "w") as json_file:
                json.dump(retrieval_results, json_file, ensure_ascii=False, indent=4)