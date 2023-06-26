import os
import re
import string
from collections import Counter
import utils.preprocessing as preprocessing
from utils.preprocessing import get_args, add_translation2trainset
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
from modelscope.trainers.nlp.document_grounded_dialog_generate_trainer import \
    DocumentGroundedDialogGenerateTrainer
from modelscope.utils.logger import get_logger
import sys
import seaborn as sns
import utils.data_exploration as exploration
sns.set(style='whitegrid')
sns.set_palette('pastel')

logger = get_logger()
import sys
sys.path.insert(0, '/path/to/your/local/folder')

from utils.seed import set_seed
set_seed()
SEED = 42

# TODO Fix bug for cn/en dataset: missing rerank column error - breaks at 'context' list comprehension
def collate(batch):
    query = [item['query'] for item in batch]
    context = [item['rerank'] for item in batch]
    label = [item['response'] for item in batch]
    lang = [item['lang'] for item in batch]
    return query, context, label, lang


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

    pattern = r"^<response>\s*|^<[^>]+>\s*"
    hypothesis_list = [re.sub(pattern, '', x)  for x in hypothesis_list] #if len(x) > 10 else 'placeholder'
    reference_list = [
        re.sub(pattern, '', x) for x in result_dict['targets']
    ] # .replace('<response>', '') 
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
          eval_lang:list,
          total_epoches=10,
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
        num_workers=2,
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
            query, context, label,_ = payload
 
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

        if losses:
            logger.info(
                f'epoch: {epoch} \t batch: last \t loss: {sum(losses) / len(losses)}'
            )

        meters = evaluate(trainer, eval_lang=eval_lang,  batch_size=batch_size)
        total_score = sum(sum(meter.values()) for meter in meters.values())
        if total_score >= best_score:
            best_score = total_score
            model_path = os.path.join(trainer.model.model_dir,
                                      'finetuned_model.bin')
            state_dict = trainer.model.model.state_dict()
            torch.save(state_dict, model_path)
            logger.info(
                'epoch %d obtain max score: %.4f, saving model to %s' %
                (epoch, total_score, model_path))


def evaluate(trainer, eval_lang:list, batch_size=16, checkpoint_path=None):
    model = trainer.model.model.generator.generator
    tokenizer = trainer.preprocessor.generation_tokenizer
    device = trainer.preprocessor.device

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)
        trainer.model.model.load_state_dict(state_dict)

    trainer.model.model.eval()
    all_meters= {}
    for lang in eval_lang:
        with torch.no_grad():
            valid_loader = DataLoader(
                dataset=trainer.eval_dataset,
                batch_size=batch_size,
                collate_fn=collate)
            
            results = {'outputs': [], 'targets': []}
            for index, payload in enumerate(tqdm.tqdm(valid_loader)):
                query, context, label, curr_lang = payload
                if bool(set(curr_lang) & set(lang)) == 0:
                    continue
                query, context, label, curr_lang = zip(*[(q, p, n, l) for q, p, n, l in zip(query, context, label, curr_lang) if l in lang])
                query = list(query)
                context = list(context)
                label = list(label)
                curr_lang = list(curr_lang)

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
                                    f"{'_'.join(lang)}_evaluate_result.json")
            logger.info(f"{'_'.join(lang)} - {meters}")
            with open(result_path, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            all_meters["_".join(lang)] = meters
            print(f"{all_meters=}")
    return all_meters


def main(**kwargs):

    train_dataset_fr, train_dataset_vi, train_dataset_en, train_dataset_cn = None, None, None, None
    langs = set(item for sublist in kwargs["eval_lang"] for item in sublist)
    if "en" in langs:
        train_dataset_en = pd.read_json("en_train_dataset_retrieval_generation_in_domain.json", lines=True)
        train_dataset_en["lang"] = "en"
        train_dataset_en = train_dataset_en.rename({"passages": "rerank"},  axis='columns')
        # en_train_dataset = preprocessing.read('DAMO_ConvAI/EnDoc2BotDialogue')
  
    if "cn" in langs:
        cn_train_dataset = pd.read_json("cn_train_dataset_in_domain.json", lines=True)
        cn_train_dataset["lang"] = "cn"
        cn_train_dataset = cn_train_dataset.rename({"passages": "rerank"},  axis='columns')
        # cn_train_dataset = preprocessing.read('DAMO_ConvAI/ZhDoc2BotDialogue')
        
    if "fr" in langs:
        train_dataset_fr = preprocessing.read('DAMO_ConvAI/FrDoc2BotGeneration')
        train_dataset_fr["lang"] = "fr"

    if "vi" in langs:
        train_dataset_vi = preprocessing.read('DAMO_ConvAI/ViDoc2BotGeneration')
        train_dataset_vi["lang"] = "vi"

    train_dataset_vi, dev_dataset_vi = preprocessing.test_split(train_dataset_vi, random_state=SEED)
    train_dataset_fr, dev_dataset_fr = preprocessing.test_split(train_dataset_fr, random_state=SEED)

    train_dataset_en, dev_dataset_en = preprocessing.test_split(train_dataset_en, random_state=SEED)
    train_dataset_cn, dev_dataset_cn = preprocessing.test_split(train_dataset_cn, random_state=SEED)

    # add machine translated en -> fr, vi queries to train set
    if kwargs["translate_mode"] == "train":
        pipeline_step= 'generation'
        train_dataset_vi = add_translation2trainset(train_df=train_dataset_vi, lang='vi', pipeline_step=pipeline_step, dir=kwargs["cache_dir"])
        train_dataset_fr = add_translation2trainset(train_df=train_dataset_fr, lang='fr', pipeline_step=pipeline_step, dir=kwargs["cache_dir"])
        # ttrain_vi = pd.read_json(f"{kwargs['cache_dir']}/ttrain_generation_vi.json", lines=True)
        # ttrain_vi = ttrain_vi.rename({"passages": "rerank"},  axis='columns')
        # ttrain_vi["lang"] = "vi" 

        # old_size = len(train_dataset_vi)
        # train_dataset_vi = pd.concat([train_dataset_vi, ttrain_vi])
        # train_dataset_vi = train_dataset_vi.reset_index(drop=True)
        # new_size = len(train_dataset_vi)

        # print(f"added {new_size - old_size} new datapoints to vi train dataset...")

        # ttrain_fr = pd.read_json(f"{kwargs['cache_dir']}ttrain_generation_fr.json", lines=True)
        # ttrain_fr = ttrain_fr.rename({"passages": "rerank"},  axis='columns')
        # ttrain_fr["lang"] = "fr" 

        # old_size = len(train_dataset_fr)
        # train_dataset_fr = pd.concat([train_dataset_fr, ttrain_fr])
        # train_dataset_fr = train_dataset_fr.reset_index(drop=True)
        # new_size = len(train_dataset_fr)

        # print(f"added {new_size - old_size} new datapoints to fr train dataset...")


    if kwargs["lang_token"]:
        train_dataset_fr      = preprocessing.add_lang_token(train_dataset_fr, "fr", ["query", "rerank"]) 
        train_dataset_vi      = preprocessing.add_lang_token(train_dataset_vi, "vi", ["query", "rerank"]) 
        train_dataset_en      = preprocessing.add_lang_token(train_dataset_en, "en", ["query", "rerank"]) 
        train_dataset_cn      = preprocessing.add_lang_token(train_dataset_cn, "cn", ["query", "rerank"]) 

        dev_dataset_fr = preprocessing.add_lang_token(dev_dataset_fr, "fr", ["query", "rerank"]) 
        dev_dataset_vi = preprocessing.add_lang_token(dev_dataset_vi, "vi", ["query", "rerank"]) 
        dev_dataset_en = preprocessing.add_lang_token(dev_dataset_en, "en", ["query", "rerank"]) 
        dev_dataset_cn = preprocessing.add_lang_token(dev_dataset_cn, "cn", ["query", "rerank"]) 


    lang_dd = {
        "fr": (train_dataset_fr, dev_dataset_fr),
        "vi": (train_dataset_vi, dev_dataset_vi),
        "en": (train_dataset_en, dev_dataset_en),
        "cn": (train_dataset_cn, dev_dataset_cn)
    }

    train_dataset, dev_dataset = [], []
    for lang in langs:
        train_tmp, dev_tmp = lang_dd[lang]
        train_dataset.append(train_tmp)
        dev_dataset.append(dev_tmp)
        
    train_dataset   = pd.concat(train_dataset) 
    dev_dataset     = pd.concat(dev_dataset)

    if not kwargs["lang_token"]:
        train_dataset["rerank"]  = train_dataset.rerank.apply(literal_eval)
        dev_dataset["rerank"]    = dev_dataset.rerank.apply(literal_eval)

    # truncate passages
    # if len(langs - set(["fr", "vi"])) > 1:
    #     df_wo_cn    = train_df.head(len(train_df) - len(train_dataset_cn))
    #     # max_len     = len(max(sum(df_wo_cn["rerank"].tolist(), []), key=len))
    #     max_len = max(len(string) for lst in df_wo_cn["rerank"] for string in lst)
    #     train_df["rerank"]  = train_df.rerank.apply(lambda s: [x[:max_len] for x in s])
    #     dev_df["rerank"]    = dev_df.rerank.apply(lambda s: [x[:max_len] for x in s])

    if kwargs["eval_input_file"] is None:
        raise Exception("Please specify arg --eval-input-file to read eval dataset from")
    preprocessing.save_to_json(dev_dataset, dev_dataset.columns, fname="test.json", pdir=kwargs["cache_dir"])

    if kwargs["lang_token"]:
        freq_df = exploration.get_freq_df(train_dataset, dev_dataset)
        exploration.plot_freq(freq_df, plot_dir=kwargs["cache_dir"])

    parent_dir = "all_passages/lang_token" if kwargs["lang_token"] else "all_passages"
    # with open(f'{parent_dir}/id_to_passage.json') as f:
    #     id_to_passage = json.load(f)

    print(f"{kwargs['lang_token']=}")
    cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_generation_pretrain', cache_dir=kwargs["cache_dir"])
    trainer = DocumentGroundedDialogGenerateTrainer(
        model           =   cache_path,
        train_dataset   =   train_dataset.to_dict('records'), # train_dataset,
        eval_dataset    =   dev_dataset.to_dict('records'), # train_dataset[:100],
        lang_token      =   kwargs["lang_token"]
    )

    # use batch accumulation
    if kwargs["batch_accumulation"]:
        kwargs["gradient_accumulation_steps"] = kwargs["batch_size"] / (kwargs["num_devices"] * kwargs["per_gpu_batch_size"])

    print(f"BATCH SIZE: {kwargs['per_gpu_batch_size']}")

    train(trainer, eval_lang=kwargs["eval_lang"], batch_size=kwargs["per_gpu_batch_size"], accumulation_steps=kwargs["gradient_accumulation_steps"], total_epoches=10, learning_rate=1e-4, loss_log_freq=1)
    evaluate(trainer, eval_lang=kwargs["eval_lang"], checkpoint_path=os.path.join(trainer.model.model_dir,
                                                   'finetuned_model.bin'))


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)