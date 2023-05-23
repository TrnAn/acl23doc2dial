import os
import re
import string
from collections import Counter
import gc
import json
import sacrebleu
import torch
import tqdm
import pandas as pd
from ast import literal_eval
from rouge import Rouge
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
import transformers
transformers.logging.set_verbosity_error()

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.document_grounded_dialog_generate_trainer import \
    DocumentGroundedDialogGenerateTrainer
from modelscope.utils.constant import DownloadMode
from modelscope.utils.logger import get_logger
from sklearn.model_selection import train_test_split
import argparse, sys
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import utils.preprocessing as preprocessing
import utils.data_exploration as exploration
sns.set(style='whitegrid')
sns.set_palette('pastel')

logger = get_logger()
import sys
sys.path.insert(0, '/path/to/your/local/folder')

# TODO Fix bug for cn/en dataset: missing rerank column error - breaks at 'context' list comprehension
def collate(batch):
    query = [item['query'] for item in batch]
    context = [item['rerank'] for item in batch]
    label = [item['response'] for item in batch]
    return query, context, label


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


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def matching_evaluate(references, predictions):
    f1 = em = total = 0
    for ref_text, prediction in zip(references, predictions):
        total += 1
        ground_truths = [ref_text]
        f1 += metric_max_over_ground_truths(f1_score, prediction,
                                            ground_truths)
        em += metric_max_over_ground_truths(exact_match_score, prediction,
                                            ground_truths)
    f1 = 100.0 * f1 / total
    em = 100.0 * em / total

    return f1, em


def measure_result(result_dict):
    meters = dict()

    hypothesis_list = [
        x.replace('<extra_id_0>', '') for x in result_dict['outputs']
    ]
    hypothesis_list = [x if len(x) > 10 else 'placeholder' for x in hypothesis_list]
    reference_list = [
        x.replace('<response>', '') for x in result_dict['targets']
    ]
    instance_num = len(reference_list)

    # F1
    f1, em = matching_evaluate(reference_list, hypothesis_list)
    meters['f1'] = f1

    # SacreBleu
    bleu_score = [
        sacrebleu.sentence_bleu(hypothesis, [reference]).score
        for hypothesis, reference in zip(hypothesis_list, reference_list)
    ]
    bleu_score = sum(bleu_score) / instance_num
    meters['bleu'] = bleu_score

    # Rouge-L
    rouge_func = Rouge()
    rouge_score = [
        x['rouge-l']['f']
        for x in rouge_func.get_scores(hypothesis_list, reference_list)
    ]
    rouge_score = (sum(rouge_score) / instance_num) * 100
    meters['rouge'] = rouge_score

    return meters


def train(trainer,
          total_epoches=1, #10,
          batch_size=16,
          accumulation_steps=1,
          learning_rate=1e-4,
          warmup_ratio=0.1,
          weight_decay=0.1,
          eps=1e-06,
          loss_log_freq=40,
          clip_grad_norm=1.0):
    model = trainer.model.model.generator.generator
    tokenizer = trainer.preprocessor.generation_tokenizer
    device = trainer.preprocessor.device

    train_loader = DataLoader(
        dataset=trainer.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True)

    optimizer = prepare_optimizer(trainer.model.model, learning_rate,
                                  weight_decay, eps)
    steps_per_epoch = len(train_loader) // accumulation_steps
    scheduler = prepare_scheduler(optimizer, total_epoches,
                                  steps_per_epoch, warmup_ratio)
    best_score = 0.0
    for epoch in range(total_epoches):
        trainer.model.model.train()
        losses = []
        for index, payload in enumerate(tqdm.tqdm(train_loader)):
            query, context, label = payload
            query = [
                tokenizer.decode(
                    tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0][:128])
                for x in query
            ]
            # print(f"query - number of tokens: {len(query)}")

            generator_inputs = [
                ' '.join([query[i], '<passage>', context[i][0]])
                for i in range(len(query))
            ]
        
            # for i in range(len(query)):
            #     print(f"context= {context[i]}")
            #     print(f"context at i= {context[i][0]}")
            #     print(f"context length: {len(context[i][0])}") 

            input_ids = tokenizer.batch_encode_plus(
                list(generator_inputs), padding=True, return_tensors='pt').input_ids.to(device)
            label_ids = tokenizer.batch_encode_plus(
                list(label), padding=True, return_tensors='pt').input_ids.to(device)

            loss = model(input_ids=input_ids, labels=label_ids)[0]

            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            loss.backward()

            if (index + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            losses.append(loss.item())
            if (index + 1) % loss_log_freq == 0:
                logger.info(
                    f'epoch: {epoch} \t batch: {batch_size * index} \t loss: {sum(losses) / len(losses)}'
                )
                losses = []

            print(f"Memory cached: {torch.cuda.memory_allocated()/1024**2}MB")

        if losses:
            logger.info(
                f'epoch: {epoch} \t batch: last \t loss: {sum(losses) / len(losses)}'
            )

        meters = evaluate(trainer, batch_size=batch_size)
        total_score = sum([x for x in meters.values()])
        if total_score >= best_score:
            best_score = total_score
            model_path = os.path.join(trainer.model.model_dir,
                                      'finetuned_model.bin')
            state_dict = trainer.model.model.state_dict()
            torch.save(state_dict, model_path)
            logger.info(
                'epoch %d obtain max score: %.4f, saving model to %s' %
                (epoch, total_score, model_path))


def evaluate(trainer, batch_size=16, checkpoint_path=None):
    model = trainer.model.model.generator.generator
    tokenizer = trainer.preprocessor.generation_tokenizer
    device = trainer.preprocessor.device

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)
        trainer.model.model.load_state_dict(state_dict)

    valid_loader = DataLoader(
        dataset=trainer.eval_dataset,
        batch_size=batch_size,
        collate_fn=collate, 
        pin_memory=True)
    trainer.model.model.eval()
    with torch.no_grad():
        results = {'outputs': [], 'targets': []}
        for index, payload in enumerate(tqdm.tqdm(valid_loader)):
            query, context, label = payload
            query = [
                tokenizer.decode(
                    tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0][:128])
                for x in query
            ]
            generator_inputs = [
                ' '.join([query[i], '<passage>', context[i][0]])
                for i in range(len(query))
            ]
            input_ids = tokenizer.batch_encode_plus(
                list(generator_inputs), padding=True, return_tensors='pt').input_ids.to(device)

            outputs = model.generate(input_ids, num_beams=3, max_length=128, early_stopping=True,
                                     no_repeat_ngram_size=3)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            label = trainer.preprocessor.generation_tokenizer.batch_decode(
                trainer.preprocessor.generation_tokenizer.batch_encode_plus(
                    label, add_special_tokens=False).input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)

            results['outputs'] += predictions
            results['targets'] += label
        meters = measure_result(results)
        result_path = os.path.join(trainer.model.model_dir,
                                   'evaluate_result.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info(meters)
    return meters


def main():
    gc.collect()

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--gradient-accumulation-steps", help= "Specifiy cache dir to save model to", type= int, default= 1)
    parser.add_argument("--num-devices", help= "Specifiy number of devices available", type= int, default= 1)
    parser.add_argument("--batch-size", help= "Specifiy batch size", type= int, default= 16)
    parser.add_argument("--per-gpu-batch-size", help= "Specifiy batch size", type= int, default= 16)
    parser.add_argument("--extended-dataset", help= "Run experiments on English and Chinese dataset", action=argparse.BooleanOptionalAction)
    parser.add_argument("--only-english", help= "Run experiments only on English dataset", type= int, default=0)
    parser.add_argument("--only-chinese", help= "Run experiments only on Chinese dataset", type= int, default=0)
    parser.add_argument("--eval-input-file", help= "File to read eval dataset (query, rerank, response) from", type=str, default=None)
    parser.add_argument("--test-size", help= "Set test split", type= float, default= 0.1)
    parser.add_argument("--lang-token", help= "Add language token <lang> to input", action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch-accumulation", help= "Use batch accumulation to maintain baseline results", action=argparse.BooleanOptionalAction)
    parser.add_argument("--cache-dir", help= "Specifiy cache dir to save model to", type= str, default= ".")
    args = parser.parse_args()
    
    # read in English + Chinese dataset
    en_train_dataset, cn_train_dataset = None, None
    if args.extended_dataset:
        if not bool(args.only_chinese):
            en_train_dataset = pd.read_json("en_train_dataset_retrieval_generation_in_domain.json", lines=True)
            # n = int(len(en_train_dataset) * 0.7)  # Number of samples to select, 75% of the DataFrame
            # en_train_dataset = en_train_dataset.sample(n=n, random_state = 42)
            en_train_dataset["lang"] = "en"
            en_train_dataset = en_train_dataset.rename({"passages": "rerank"},  axis='columns')
  
        if not bool(args.only_english):
            cn_train_dataset = pd.read_json("cn_train_dataset_in_domain.json", lines=True)
            cn_train_dataset["lang"] = "cn"
            cn_train_dataset = cn_train_dataset.rename({"passages": "rerank"},  axis='columns')


        # cn_train_dataset = preprocessing.read('DAMO_ConvAI/ZhDoc2BotDialogue')
        # en_train_dataset = preprocessing.read('DAMO_ConvAI/EnDoc2BotDialogue')
    
    # read in Vietnamese + French dataset
    fr_train_dataset = preprocessing.read('DAMO_ConvAI/FrDoc2BotGeneration')
    vn_train_dataset = preprocessing.read('DAMO_ConvAI/ViDoc2BotGeneration')

    fr_train_dataset["lang"] = "fr"
    vn_train_dataset["lang"] = "vi"

    seed = 42
    train_dataset_vn, dev_dataset_vn = preprocessing.test_split(vn_train_dataset, random_state=seed)
    train_dataset_fr, dev_dataset_fr = preprocessing.test_split(fr_train_dataset, random_state=seed)

    train_dataset_en, dev_dataset_en = preprocessing.test_split(en_train_dataset, random_state=seed)
    train_dataset_cn, dev_dataset_cn = preprocessing.test_split(cn_train_dataset, random_state=seed)

    if args.lang_token:
        train_dataset_fr      = preprocessing.add_lang_token(train_dataset_fr, "fr", ["query", "rerank"]) 
        train_dataset_vn      = preprocessing.add_lang_token(train_dataset_vn, "vi", ["query", "rerank"]) 
        train_dataset_en      = preprocessing.add_lang_token(train_dataset_en, "en", ["query", "rerank"]) 
        train_dataset_cn      = preprocessing.add_lang_token(train_dataset_cn, "cn", ["query", "rerank"]) 

        dev_dataset_fr = preprocessing.add_lang_token(dev_dataset_fr, "fr", ["query", "rerank"]) 
        dev_dataset_vn = preprocessing.add_lang_token(dev_dataset_vn, "vi", ["query", "rerank"]) 
        dev_dataset_en = preprocessing.add_lang_token(dev_dataset_en, "en", ["query", "rerank"]) 
        dev_dataset_cn = preprocessing.add_lang_token(dev_dataset_cn, "cn", ["query", "rerank"]) 

    train_df    = pd.concat([train_dataset_fr, train_dataset_vn, train_dataset_en, train_dataset_cn])
    dev_df      = pd.concat([dev_dataset_fr, dev_dataset_vn, dev_dataset_en, dev_dataset_cn])

    # if not args.lang_token:
    #     train_df["rerank"]  = train_df.rerank.apply(literal_eval)
    #     dev_df["rerank"]    = dev_df.rerank.apply(literal_eval)

    # truncate passages
    if args.extended_dataset and not bool(args.only_english):
        df_wo_cn    = train_df.head(len(train_df) - len(train_dataset_cn))
        # max_len     = len(max(sum(df_wo_cn["rerank"].tolist(), []), key=len))
        max_len = max(len(string) for lst in df_wo_cn["rerank"] for string in lst)
        train_df["rerank"]  = train_df.rerank.apply(lambda s: [x[:max_len] for x in s])
        dev_df["rerank"]    = dev_df.rerank.apply(lambda s: [x[:max_len] for x in s])


    if args.eval_input_file is None:
        raise Exception("Please specify arg --eval-input-file to read eval dataset from")
    # preprocessing.save_to_json(dev_df, dev_df.columns, fname=args.eval_input_file)
    preprocessing.save_to_json(dev_df, dev_df.columns, fname="test.json", dir=args.cache_dir)

    if args.lang_token:
        freq_df = exploration.get_freq_df(train_df, dev_df)
        exploration.plot_freq(freq_df)

    parent_dir = "all_passages/lang_token" if args.lang_token else "all_passages"
    with open(f'{parent_dir}/id_to_passage.json') as f:
        id_to_passage = json.load(f)

    cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_generation_pretrain', cache_dir=args.cache_dir)
    trainer = DocumentGroundedDialogGenerateTrainer(
        model           =   cache_path,
        train_dataset   =   train_df.to_dict('records'), # train_dataset,
        eval_dataset    =   dev_df.to_dict('records'), # train_dataset[:100],
        lang_token      =   args.lang_token
    )

    # use batch accumulation
    if args.batch_accumulation:
        args.gradient_accumulation_steps = args.batch_size / (args.num_devices * args.per_gpu_batch_size)

    print(f"BATCH SIZE: {args.per_gpu_batch_size}")
    train(trainer, batch_size=args.per_gpu_batch_size, accumulation_steps=args.gradient_accumulation_steps, total_epoches=1, learning_rate=1e-4)
    evaluate(trainer, checkpoint_path=os.path.join(trainer.model.model_dir,
                                                   'finetuned_model.bin'))


if __name__ == '__main__':
    main()