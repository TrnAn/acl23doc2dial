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

    parser.add_argument("--use-extended-dataset", help= "Run experiments on English and Chinese dataset", type= bool, default= True)
    parser.add_argument("--test-size", help= "Set test split", type= float, default= 0.1)
    parser.add_argument("--use-lang-token", help= "Add language token <lang> to input", type= bool, default= True)
    parser.add_argument("--use-batch-accumulation", help= "Use batch accumulation to maintain baseline results", type= bool, default= False)
    parser.add_argument("--cache-dir", help= "Specifiy cache dir to save model to", type= str, default= "./")
    args = parser.parse_args()

    fr_train_dataset = preprocessing.read('DAMO_ConvAI/FrDoc2BotRetrieval')
    vi_train_dataset = preprocessing.read('DAMO_ConvAI/ViDoc2BotRetrieval')

    # train_dataset = [x for dataset in [fr_train_dataset, vi_train_dataset] for x in dataset]
    train_dataset_fr, dev_dataset_fr = preprocessing.test_split(fr_train_dataset)
    train_dataset_vn, dev_dataset_vn = preprocessing.test_split(vi_train_dataset)
    
    if args.use_lang_token:
        train_dataset_fr    = preprocessing.add_lang_token(train_dataset_fr, "fr", colnames=["query", "positive", "negative"]) 
        train_dataset_vn    = preprocessing.add_lang_token(train_dataset_vn, "vn", colnames=["query", "positive", "negative"])
        dev_dataset_fr = preprocessing.add_lang_token(dev_dataset_fr, "fr", colnames=["query", "positive", "negative"]) 
        dev_dataset_vn = preprocessing.add_lang_token(dev_dataset_vn, "vn", colnames=["query", "positive", "negative"])  

    train_dataset   = pd.concat([train_dataset_fr, train_dataset_vn])
    dev_dataset     = pd.concat([dev_dataset_fr, dev_dataset_vn])

    all_passages = []
    for file_name in ['fr', 'vi']:
        with open(f'all_passages/{file_name}.json') as f:
            tmp = json.load(f)
            
            if args.use_lang_token:
                tmp = [f'{s} <{file_name}>' for s in tmp]
            
            all_passages += tmp

    cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_retrieval_pretrain', cache_dir=args.cache_dir)
    trainer = DocumentGroundedDialogRetrievalTrainer(
        model=cache_path,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        all_passages=all_passages,
        use_lang_token=args.use_lang_token)
    trainer.train(
        batch_size=128,
        total_epoches=50,
    )
    trainer.evaluate(
        checkpoint_path=os.path.join(trainer.model.model_dir,
                                    'finetuned_model.bin'))



if __name__ == '__main__':
    main()