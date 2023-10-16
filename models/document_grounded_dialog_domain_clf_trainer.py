from transformers import XLMRobertaPreTrainedModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from modelscope.trainers import EpochBasedTrainer
from modelscope.trainers.builder import TRAINERS
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import F1Score
from modelscope.preprocessors import \
    DocumentGroundedDialogRetrievalPreprocessor
from pathlib import Path
from transformers import AdamW, get_scheduler
from modelscope.utils.constant import ModeKeys
import tqdm
import os
from collections import defaultdict 
from modelscope.utils.logger import get_logger
logger = get_logger()
import copy

class XLMRobertaDomainClfHead(XLMRobertaPreTrainedModel):
    def __init__(self, config, adapt_args, model):
        super().__init__(config)

        self.num_labels = adapt_args["num_classes"]
        self.labels = adapt_args["labels"]
        self.config     = config
        self.model      = model
        # self.dropout    = nn.Dropout(config['hidden_dropout_prob'])
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout    = nn.Dropout(classifier_dropout)
        self.classifier   = nn.Linear(config.hidden_size, adapt_args["num_classes"])

        # self.classifier = nn.Linear(config.hidden_size, adapt_args["num_classes"])
    
    @staticmethod
    def encode(model, input_ids, attention_mask, gck_segment=32):
        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        pooled_output = []
        for mini_batch in range(0, input_ids.shape[0], gck_segment):
            mini_batch_input_ids = input_ids[mini_batch:mini_batch
                                             + gck_segment]
            mini_batch_attention_mask = attention_mask[mini_batch:mini_batch
                                                       + gck_segment]
            mini_batch_pooled_output = checkpoint(model, mini_batch_input_ids,
                                                  mini_batch_attention_mask,
                                                  dummy_tensor)
            
            pooled_output.append(mini_batch_pooled_output)
        return torch.cat(pooled_output, dim=0)
    

    def forward(self, 
                query_input_ids, 
                query_attention_mask=None, 
                labels=None, 
                gck_segment=32):
        query_vector = self.encode(model=self.model, 
                                   input_ids=query_input_ids,
                                   attention_mask=query_attention_mask, 
                                   gck_segment=gck_segment).to(device=self.device)
        
        query_vector = query_vector
        
        x = self.dropout(query_vector)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        # logits = self.classifier(query_vector)

        outputs = (None, logits) # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + (outputs[0],)

        return outputs  # (loss), logits, (hidden_states), (attentions)


def measure_result(results, num_classes):
    preds = results["outputs"]
    target = results["targets"]
    
    f1_fn = F1Score(task="multiclass", num_classes=num_classes, average="macro")
    f1 = f1_fn(preds, target)

    macro_acc_fn = MulticlassAccuracy(num_classes=num_classes, average="macro")
    macro_acc = macro_acc_fn(preds, target)

    acc_fn = MulticlassAccuracy(num_classes=num_classes, average=None)
    acc = acc_fn(preds, target)

    return f1, macro_acc, acc


def collate(batch):
    query   = [item['query']    for item in batch]
    labels  = [item['domain']   for item in batch]
    langs   = [item['lang']    for item in batch]
    return query, labels, langs

            
def collate_retrieval(batch):
    query   =   [item['query']    for item in batch]
    response =  [item['response']  for item in batch]
    positive =  [item['positive']  for item in batch]
    labels  =   [item['domain']   for item in batch]
    langs   =   [item['lang']    for item in batch]

    return query, response, positive, labels, langs

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

@TRAINERS.register_module(
    module_name="document_grounded_dialog_domain_clf_trainer", force=True)
class DocumentGroundedDialogDomainClfTrainer(EpochBasedTrainer):

    def __init__(self, model, tokenizer_dir, train_dataset, eval_dataset, eval_lang, checkpoint_path=None, **kwargs):
        self.model = model
        self.train_dataset  = train_dataset
        self.eval_dataset   = eval_dataset
        self.eval_lang = eval_lang
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"         
                
        self.preprocessor = DocumentGroundedDialogRetrievalPreprocessor(            
            model_dir=tokenizer_dir, lang_token=kwargs["lang_token"])
        
        self.device = self.preprocessor.device

        os.makedirs(checkpoint_path, exist_ok=True)
        self.checkpoint_path = checkpoint_path
        print(f"{self.checkpoint_path=}")
        if checkpoint_path is not None:
            model_fdir = os.path.join(checkpoint_path, "finetuned_model.bin")

            if os.path.exists(model_fdir):
                logger.info(f"load domain clf model: {model_fdir=}")

                state_dict = torch.load(model_fdir)
                self.model.load_state_dict(state_dict)  

        if kwargs["lang_token"] is not None:
            self.model.qry_encoder.encoder.resize_token_embeddings(self.preprocessor.token_length)  # resize query encoder of DPR model


        self.model.to(self.device)

        self.label2id   = defaultdict(int)
        self.id2label   = defaultdict(str)
        # labels  = {item['domain'] for item in train_dataset + eval_dataset}

        for idx, label in enumerate(self.model.labels):
            self.label2id[label] = idx
            self.id2label[idx] = label


    def train(self, 
              batch_size=16,
              total_epoches=10,
              per_gpu_batch_size=32,
              learning_rate=2e-5,
              warmup_ratio=0.1,
              accumulation_steps=1,
              weight_decay=0.1, 
              eps=1e-06,
              loss_log_freq=1
              ):
        
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate)

        optimizer = prepare_optimizer(self.model, learning_rate,
                                      weight_decay, eps)
        steps_per_epoch = len(train_loader) // accumulation_steps
        scheduler = prepare_scheduler(optimizer, total_epoches,
                                      steps_per_epoch, warmup_ratio)
        best_score = 0.0
        for epoch in range(total_epoches):
            losses = []
            for index, payload in enumerate(tqdm.tqdm(train_loader)):
                # Every data instance is an input + label pair
                queries, labels, _ = payload
                processed = self.preprocessor({'query': queries}, invoke_mode=ModeKeys.INFERENCE)
                processed["labels"] = torch.tensor([self.label2id[label] for label in labels], dtype=torch.long).to(device=self.device)
                # Zero your gradients for every batch!
                optimizer.zero_grad()
                loss, logits = self.model.forward(**processed)
                
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
            total_score = 0
            for tensor in meters.values():
                f1, macro_acc, _ = tensor
                total_score += sum([f1, macro_acc])

            logger.info(
                f'obtain max score: {total_score:.4f}') # (best score: {best_score:.4f})
            
            # if total_score >= best_score:
                # best_score = total_score
            os.makedirs(self.checkpoint_path, exist_ok=True)
            model_path = os.path.join(self.checkpoint_path,
                                        'finetuned_model.bin')
            state_dict = self.model.state_dict()
            torch.save(state_dict, model_path)
            logger.info(
                'epoch %d obtain max score: %.4f, saving model to %s' %
                (epoch, total_score, model_path))
                

    def evaluate(self, per_gpu_batch_size=32):
        """
        Evaluate testsets
        """
        # if checkpoint_path is None and os.path.isdir(self.save_dir):
        #     checkpoint_path = os.path.join(self.save_dir, "finetuned_model.bin")
        #     logger.info(f"{checkpoint_path=}")

        # state_dict = torch.load(checkpoint_path)
        # self.model.load_state_dict(state_dict)

        self.model.eval()
        with torch.no_grad():
            all_meters = {}

            # @TODO change to kwargs["eval_lang"]
            for lang in self.eval_lang:
                results = {'outputs': torch.Tensor([]), 'targets': torch.Tensor([])}
                valid_loader = DataLoader(
                    dataset=self.eval_dataset,
                    batch_size=per_gpu_batch_size,
                    collate_fn=collate)
                for payload in tqdm.tqdm(valid_loader):
                    query, label, curr_lang = payload

                    if bool(set(curr_lang) & set(lang)) == 0: # language is not in current batch
                        continue

                    query, label, curr_lang = zip(*[(q, ll, lg) for q, ll, lg in zip(query, label, curr_lang) if lg in lang])
                    query       = list(query)
                    label       = list(label)
                    curr_lang   = list(curr_lang)

                    processed = self.preprocessor({'query': query},
                                                invoke_mode=ModeKeys.INFERENCE)

                    _, logits = self.model.forward(**processed)

                    pred_labels = logits.detach().cpu()
                    targets = torch.tensor([self.label2id[l] for l in label])

                    results['outputs'] = torch.cat((results['outputs'], pred_labels))
                    results['targets'] = torch.cat((results['targets'], targets))

                meters = measure_result(results, num_classes=self.model.num_labels)

                all_meters["_".join(lang)] = meters
                logger.info(f"f1-score (macro): {meters[0]:.3f}; accuracy (macro): {meters[1]:.3f}; accuracy for each class: {list(zip(self.id2label.values(), meters[2].numpy()))}")

        return all_meters


    def predict(self, dataset:list, filter_domain:str, all_passages:list, per_gpu_batch_size:int=32):
        """
        Predict domain given query
        """
        def filter_queries_by_domain(payload, filter_domain):
     
            filtered_list = [{"query":query, 
                              "response": response, 
                              "positive": positive, 
                              "domain": domain, 
                              "lang":lang} for query, response, positive, domain, lang, pred_labels in payload if pred_labels == filter_domain]
            return filtered_list

        # if checkpoint_path is None and os.path.isdir(self.save_dir):
        #     checkpoint_path = os.path.join(self.save_dir, "finetuned_model.bin")
        #     logger.info(f"{checkpoint_path=}")

        # state_dict = torch.load(checkpoint_path)
        # self.model.load_state_dict(state_dict)

        self.model.eval()
        with torch.no_grad():
            data_loader = DataLoader(
                dataset=dataset,
                num_workers=4, 
                pin_memory=True,
                batch_size=per_gpu_batch_size,
                collate_fn=collate_retrieval)
            
            queries_of_domain = []

            for payload in tqdm.tqdm(data_loader):
                query, response,  positive, domain, lang = payload
                processed = self.preprocessor({'query': query},
                                            invoke_mode=ModeKeys.INFERENCE)

                _, logits = self.model.forward(**processed)

                pred_labels = torch.argmax(logits, dim=1).detach().cpu().numpy()
                pred_labels = [self.id2label[label] for label in pred_labels]
                
                tmp = zip(query, response, positive, domain, lang, pred_labels)
                queries_of_domain += filter_queries_by_domain(tmp, filter_domain)

                # print(list(zip(domain, pred_labels)))

            all_passages = [p for p in all_passages if p.endswith(filter_domain)]
            # print(f"{filter_domain} = {all_passages=}")
            # print(f"{queries_of_domain[:2]=}")
        return queries_of_domain, all_passages

