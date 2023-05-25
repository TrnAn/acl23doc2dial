import os
import json
import pandas as pd
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
import utils.preprocessing as preprocessing
import argparse


def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    parser.add_argument("--extended-dataset", help= "Run experiments on English and Chinese dataset", action=argparse.BooleanOptionalAction)
    parser.add_argument("--only-english", help= "Run experiments only on English dataset", type= int, default=0)
    parser.add_argument("--only-chinese", help= "Run experiments only on Chinese dataset", type= int, default=0)
    parser.add_argument("--test-size", help= "Set test split", type= float, default= 0.1)
    parser.add_argument("--lang-token", help= "Add language token <lang> to input", action=argparse.BooleanOptionalAction)
    parser.add_argument("--batch-accumulation", help= "Use batch accumulation to maintain baseline results", action=argparse.BooleanOptionalAction)
    parser.add_argument("--gradient-accumulation-steps", help= "Specifiy cache dir to save model to", type= int, default= 1)
    parser.add_argument("--num-devices", help= "Specifiy number of devices available", type= int, default= 1)
    parser.add_argument("--batch-size", help= "Specifiy batch size", type= int, default= 128)
    parser.add_argument("--per-gpu-batch-size", help= "Specifiy batch size", type= int, default= 1)
    parser.add_argument("--cache-dir", help= "Specifiy cache dir to save model to", type= str, default= ".")
    parser.add_argument("--eval-input-file", help= "File to read eval dataset (query, rerank, response) from", type=str, default=None)

    args = parser.parse_args()

    fr_train_dataset = preprocessing.read('DAMO_ConvAI/FrDoc2BotRetrieval')
    vi_train_dataset = preprocessing.read('DAMO_ConvAI/ViDoc2BotRetrieval')
    fr_train_dataset["lang"] = "fr"
    vi_train_dataset["lang"] = "vi"

    en_train_dataset, cn_train_dataset = None, None
    if args.extended_dataset:
        if not bool(args.only_chinese):
            en_train_dataset = pd.read_json("en_train_dataset_retrieval_generation_in_domain.json", lines=True)
            en_train_dataset["lang"] = "en"
        if not bool(args.only_english):
            cn_train_dataset = pd.read_json("cn_train_dataset_in_domain.json", lines=True)
            cn_train_dataset["lang"] = "cn"



    train_dataset_fr, dev_dataset_fr = preprocessing.test_split(fr_train_dataset)
    train_dataset_vn, dev_dataset_vn = preprocessing.test_split(vi_train_dataset)
    train_dataset_en, dev_dataset_en = preprocessing.test_split(en_train_dataset)
    train_dataset_cn, dev_dataset_cn = preprocessing.test_split(cn_train_dataset)

    if args.lang_token:
        train_dataset_fr    = preprocessing.add_lang_token(train_dataset_fr, "fr", colnames=["query", "positive", "negative"]) 
        train_dataset_vn    = preprocessing.add_lang_token(train_dataset_vn, "vi", colnames=["query", "positive", "negative"])
        train_dataset_en    = preprocessing.add_lang_token(train_dataset_en, "en", colnames=["query", "positive", "negative"])
        train_dataset_cn    = preprocessing.add_lang_token(train_dataset_cn, "cn", colnames=["query", "positive", "negative"])

        dev_dataset_fr = preprocessing.add_lang_token(dev_dataset_fr, "fr", colnames=["query", "positive", "negative"]) 
        dev_dataset_vn = preprocessing.add_lang_token(dev_dataset_vn, "vi", colnames=["query", "positive", "negative"]) 
        dev_dataset_en = preprocessing.add_lang_token(dev_dataset_en, "en", colnames=["query", "positive", "negative"])  
        dev_dataset_cn = preprocessing.add_lang_token(dev_dataset_cn, "cn", colnames=["query", "positive", "negative"])   


    train_dataset   = pd.concat([train_dataset_fr, train_dataset_vn, train_dataset_en, train_dataset_cn]) 
    dev_dataset     = pd.concat([dev_dataset_fr, dev_dataset_vn, dev_dataset_en, dev_dataset_cn])

    # if args.extended_dataset and not bool(args.only_english):
    #     df_wo_cn    = train_dataset.head(len(train_dataset) - len(train_dataset_cn))
    #     max_len     = len(max(sum(df_wo_cn.rerank.tolist(), []), key=len))
    #     train_dataset["rerank"]  = train_dataset.rerank.apply(lambda s: [x[:max_len] for x in s])
    #     dev_dataset["rerank"]    = dev_dataset.rerank.apply(lambda s: [x[:max_len] for x in s])
    print(f"{args.eval_input_file=} {args.cache_dir=}")
    preprocessing.save_to_json(dev_dataset, dev_dataset.columns, fname=args.eval_input_file, dir=args.cache_dir)

    parent_dir = "all_passages/lang_token" if args.lang_token else "all_passages"
    all_passages = []
    languages = ['fr', 'vi']

    if args.extended_dataset:
        if not bool(args.only_chinese):
            languages += ['en']
        if not bool(args.only_english):
            languages += ['cn']

    for file_name in languages:
        with open(f'{parent_dir}/{file_name}.json') as f:
            all_passages += json.load(f)

        # use batch accumulation
    if args.batch_accumulation:
        args.gradient_accumulation_steps = args.batch_size / (args.num_devices * args.per_gpu_batch_size)

    print(f"BATCH SIZE: {args.per_gpu_batch_size}")

    cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_retrieval_pretrain', cache_dir=args.cache_dir)
    trainer = DocumentGroundedDialogRetrievalTrainer(
        model=cache_path,
        train_dataset=train_dataset.to_dict('records'),
        eval_dataset=dev_dataset.to_dict('records'),
        all_passages=all_passages,
        lang_token  =args.lang_token)
    trainer.train(
        batch_size=128,
        total_epoches=50,
        per_gpu_batch_size=args.per_gpu_batch_size,
        accumulation_steps=args.gradient_accumulation_steps
    )
    trainer.evaluate(
        checkpoint_path=os.path.join(trainer.model.model_dir,
                                    'finetuned_model.bin'))



if __name__ == '__main__':
    main()