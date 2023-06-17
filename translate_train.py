from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import NllbTokenizer
import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import re
import json
import logging
import sys
from utils import preprocessing
from pathlib import Path

# from utils import preprocessing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer   = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model       = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)

iso_lang = {
    "fr": "fra_Latn",
    "vi": "vie_Latn"
}

# Custom dataset class
class Doc2DialDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]



def translate(df, colnames:list, target_lang:str, batch_size:int=256):
    df_translate = df.copy()
    tokenize    = lambda x: tokenizer(x, return_tensors="pt",  padding=True, truncation=True).to(model.device)
    
    with torch.no_grad():
        for colname in colnames:
            dataset     = Doc2DialDataset(df[colname].values.tolist())
            data_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=True
                )
            
            translated_query = []
            for batch in tqdm(data_loader):           
                inputs = tokenize(batch)

                translated_tokens   = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[iso_lang[target_lang]], max_length=256)
                translated_query    += tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

            df_translate[colname] = [re.sub(r'^<en>\s*', '', q)  for q in translated_query]

    return df_translate


def translate_passage_text(texts, target_lang:str, source_lang:str):
    input_texts = [t["text"] for t in texts]
    encoded_input = tokenizer.batch_encode_plus(input_texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_input["input_ids"].to(model.device)
    attention_mask = encoded_input["attention_mask"].to(model.device)

    with torch.no_grad():
        translated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, forced_bos_token_id=tokenizer.lang_code_to_id[iso_lang[target_lang]], max_length=256)
        translated_texts = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)

    translated_dict = [{"pid": text["pid"], "title": text["title"], "text":  translated_text.replace(f'<{source_lang}>', f'<{target_lang}>')} for text, translated_text in zip(texts, translated_texts)]

    return translated_dict


def translate_rerank(texts, target_lang:str):
    encoded_input = tokenizer(texts, return_tensors="pt",  padding=True, truncation=True).to(model.device)
    input_ids = encoded_input["input_ids"].to(model.device)
    attention_mask = encoded_input["attention_mask"].to(model.device)

    with torch.no_grad():
        translated_ids      = model.generate(input_ids=input_ids, attention_mask=attention_mask, forced_bos_token_id=tokenizer.lang_code_to_id[iso_lang[target_lang]], max_length=256)
        translated_texts    = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)

    return translated_texts


def main():
    print(Path.home())
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument("--target ", help= "Set test split", type= float, default= 0.1)
    parser.add_argument("--lang-token", help= "Add language token <lang> to input", action=argparse.BooleanOptionalAction)
    parser.add_argument("--retrieval-step", help= "Start translation for retrieval step", action=argparse.BooleanOptionalAction)
    parser.add_argument("--rerank-step", help= "Start translation for rerank step", action=argparse.BooleanOptionalAction)
    parser.add_argument("--generation-step", help= "Start translation for generation step", action=argparse.BooleanOptionalAction)
    parser.add_argument("--cache-dir", help= "Specify directory to save translated files to", type=str, default=".")
    parser.add_argument("--target-langs", help= "Specify target languages", action='store', type=str, nargs="+", default=['fr', 'vi']) 
    parser.add_argument("--source-lang", help= "Specify source languages", type=str, default='en') 
    args = parser.parse_args()

    logging.basicConfig(filename=f'{args.cache_dir}/translate.log', level=logging.INFO)

    if args.retrieval_step or args.generation_step:
        en_train_dataset = pd.read_json("en_train_dataset_retrieval_generation_in_domain.json", lines=True)

    if args.rerank_step:
        en_train_dataset = pd.read_json(f"{args.cache_dir}/DAMO_ConvAI/nlp_convai_retrieval_pretrain/en_rerank.json")
    
    en_train_dataset["passages"] = en_train_dataset["passages"].apply(eval)

    for target_lang in args.target_langs:
        logging.info(f'{target_lang=}')

        if args.retrieval_step and not os.path.exists(os.path.join(args.cache_dir, f"retrieval_{target_lang}.json")):
            retrieval_translated_df = translate(df=en_train_dataset, colnames=["query", "positive", "negative"], target_lang=target_lang)

            preprocessing.save_to_json(df=retrieval_translated_df, export_cols=retrieval_translated_df.columns, fname= f"retrieval_{target_lang}.json", dir=args.cache_dir)

        if args.rerank_step and not os.path.exists(os.path.join(args.cache_dir, f"rerank_{target_lang}.json")):    
            rerank_translated_df = translate(df=en_train_dataset, colnames=["input"], target_lang=target_lang)
            rerank_translated_df["passages"] = en_train_dataset["passages"].apply(lambda x: translate_passage_text(x, target_lang=target_lang, source_lang=args.source_lang))
            rerank_translated_df["passages"] = rerank_translated_df["passages"].apply(lambda x: json.dumps(x, ensure_ascii=False))
            rerank_translated_df["lang"] = target_lang

            preprocessing.save_to_json(df=rerank_translated_df, export_cols=rerank_translated_df.columns, fname= f"rerank_{target_lang}.json", dir=args.cache_dir)


        if args.generation_step and not os.path.exists(os.path.join(args.cache_dir, f"generation_{target_lang}.json")):
            generation_translated_df = translate(df=en_train_dataset, colnames=["query", "response"], target_lang=target_lang)
            generation_translated_df["passages"] = generation_translated_df["passages"] .apply(lambda x: translate_rerank(x, target_lang=target_lang))
            generation_translated_df["passages"] = generation_translated_df["passages"].apply(lambda x: json.dumps(x, ensure_ascii=False))
            generation_translated_df["lang"] = target_lang

            preprocessing.save_to_json(df=generation_translated_df, export_cols=generation_translated_df.columns, fname= f"generation_{target_lang}.json", dir=args.cache_dir)
        


if __name__ == '__main__':
    main()