from transformers import AutoModelForSeq2SeqLM
from transformers import NllbTokenizerFast
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import re
import json
import logging
from utils.preprocessing import get_args
from utils import preprocessing
from tqdm import tqdm
tqdm.pandas()

# from utils import preprocessing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")
tokenizer   = NllbTokenizerFast.from_pretrained("facebook/nllb-200-distilled-1.3B") #facebook/nllb-200-distilled-600M
# print(f"{tokenizer.additional_special_tokens=}")
# new_special_tokens = tokenizer.additional_special_tokens + ["<last_turn>", "<user>", "<agent>", "<response>", "<passage>"]
# tokenizer.add_tokens(["<last_turn>", "<user>", "<agent>", "<response>", "<passage>"], special_tokens=True)
# model.resize_token_embeddings(len(tokenizer))
model.to(device)


iso_lang = {
    "fr": "fra_Latn",
    "vi": "vie_Latn",
    "en": "eng_Latn"
}

def translate(df, colnames:list, source_lang:str, target_lang:str, batch_size:int=128):
    tokenize  = lambda x: tokenizer(x, padding=True, truncation=True, return_tensors="pt").to(model.device)

    df_translate = df.copy()
    
    with torch.no_grad():
        for colname in colnames:
            dataset     = df[colname].values.tolist()
            data_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=4, 
                pin_memory=True
                )
            
            translated_query = []
            for batch in tqdm(data_loader, desc=f"translate queries of column '{colname}'"):           
                inputs = tokenize(batch)

                translated_tokens   = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[iso_lang[target_lang]], max_length=256)
                translated_query    += tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

            df_translate[colname] = [re.sub(f'^<\s*{source_lang}\s*>\s*', '', q)  for q in translated_query]

    return df_translate


def translate_passage_text(passage_col, target_lang:str, source_lang:str, batch_size:int=128):
    tokenize  = lambda x: tokenizer(x, return_tensors="pt",  padding=True, truncation=True).to(model.device)
    unique_texts = pd.unique(passage_col.explode().apply(lambda x: x.get('text')))

    with torch.no_grad():
        data_loader = DataLoader(
            dataset=unique_texts, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
            )
        
        passages_dict = {}
        for batch in tqdm(data_loader, desc=f"translate passages' texts..."):           
            inputs = tokenize(batch)

            translated_tokens   = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[iso_lang[target_lang]], max_length=256)
            translated_tmp    = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            passages_dict.update({k: v for k, v in zip(batch, translated_tmp)})

    translated_col = passage_col.progress_apply(lambda x: json.dumps([{"pid": passage_dicts["pid"], "title": passage_dicts["title"], "text": re.sub(f'^<\s*{source_lang}\s*>\s*', '', passages_dict[passage_dicts["text"]])} for passage_dicts in x], ensure_ascii=False))

    return translated_col


def translate_passages(passage_col:pd.Series, all_passages:list, target_lang:str, batch_size:int=128):
    tokenize  = lambda x: tokenizer(x, return_tensors="pt",  padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        data_loader = DataLoader(
            dataset=all_passages, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
            )
        
        passages_dict = {}
        for batch in tqdm(data_loader, desc=f"translate passages..."):    
            inputs = tokenize(batch)
            translated_tokens   = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[iso_lang[target_lang]], max_length=256)
            translated_tmp      = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            passages_dict.update({k: v for k, v in zip(batch, translated_tmp)})
  
        translated_passages = passage_col.apply(lambda x: json.dumps([passages_dict.get(item) for item in x], ensure_ascii=False))

    return translated_passages


def get_dataset(**kwargs):
    if kwargs["retrieval_step"]:
        dataset = pd.read_json("en_train_dataset_retrieval_generation_in_domain.json", lines=True)
        train_dataset, _ = preprocessing.test_split(dataset)
        return train_dataset

    if kwargs["rerank_step"]:
        dataset =  pd.read_json(f"{kwargs['cache_dir']}/DAMO_ConvAI/nlp_convai_retrieval_pretrain/{kwargs['source_langs'][0]}_{kwargs['extended_rerank_dataset_fname']}")
        train_dataset, _ = preprocessing.test_split(dataset)
        return train_dataset
    
    if kwargs["generation_step"]:     
        dataset =  pd.read_json(f"{kwargs['cache_dir']}/DAMO_ConvAI/nlp_convai_ranking_pretrain/{kwargs['source_langs'][0]}_{kwargs['extended_generation_dataset_fname']}")
        train_dataset, _ = preprocessing.test_split(dataset)
        return train_dataset

    return None


def replace_passages(passage_column:pd.Series, all_passages:dict[str]):
    return passage_column.replace(all_passages)


def main(**kwargs):

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers= [
            logging.FileHandler(f'{kwargs["cache_dir"]}/{kwargs["source_langs"][0]}2{"_".join(kwargs["target_langs"])}_translate.log'),  # File handler
            logging.StreamHandler()  # Console handler
        ]
    )
    logger = logging.getLogger()

    logger.info(f"enter translate-{kwargs['translate_mode']} mode on {device=}...")
    train_dataset = get_dataset(**kwargs).head(10)
    preprocessing.save_to_json(df=train_dataset, export_cols=train_dataset.columns, fname= f"ttrain_original_{kwargs['source_langs'][0]}.json", pdir=kwargs["cache_dir"])
    if train_dataset is None: # translate dataset already exists
        return 0
    
    passage_colname = "passages" if "passages" in train_dataset.columns else "rerank"
    train_dataset[passage_colname] = train_dataset[passage_colname].apply(eval)

    if kwargs["retrieval_step"] or kwargs["generation_step"]:
        all_passages = train_dataset[passage_colname].explode().unique()

    source_lang = kwargs["source_langs"][0]
    for target_lang in kwargs["target_langs"]:
        logger.info(f'translate "{source_lang}" dataset to {target_lang=}')

        retrieval_fname = f"ttrain_retrieval_{target_lang}.json"
        retrieval_path  = os.path.join(kwargs["cache_dir"], retrieval_fname)
        rerank_path     = os.path.join(kwargs["cache_dir"], f"ttrain_rerank_{target_lang}.json")
        generation_path = os.path.join(kwargs["cache_dir"], f"ttrain_generation_{target_lang}.json")
        
        if kwargs["retrieval_step"] and not os.path.exists(retrieval_path):
            retrieval_translated_df  = translate(df=train_dataset, 
                                                colnames=["query", "positive", "negative"], 
                                                source_lang=source_lang, 
                                                target_lang=target_lang)
            
            retrieval_translated_df["passages"] = translate_passages(passage_col= retrieval_translated_df["passages"], all_passages=all_passages, target_lang=target_lang)
            retrieval_translated_df["lang"]     = target_lang

            preprocessing.save_to_json(df=retrieval_translated_df, export_cols=retrieval_translated_df.columns, fname= retrieval_fname, pdir=kwargs["cache_dir"])


        if kwargs["rerank_step"] and not os.path.exists(rerank_path):    
            rerank_translated_df = translate(df=train_dataset, colnames=["input"], target_lang=target_lang,  source_lang=source_lang)
            rerank_translated_df["passages"] = translate_passage_text(passage_col=rerank_translated_df["passages"], target_lang=target_lang, source_lang=source_lang)
  
            rerank_translated_df["lang"] = target_lang

            preprocessing.save_to_json(df=rerank_translated_df, export_cols=rerank_translated_df.columns, fname= f"ttrain_rerank_{target_lang}.json", pdir=kwargs["cache_dir"])

        if kwargs["generation_step"] and not os.path.exists(generation_path):
            colnames = ["query", "response"]
            generation_translated_df = translate(df=train_dataset, colnames=colnames, source_lang=source_lang, target_lang=target_lang)
            generation_translated_df["rerank"] = translate_passages(passage_col= generation_translated_df["rerank"], all_passages=all_passages, source_lang=source_lang, target_lang=target_lang)
            generation_translated_df["lang"] = target_lang

            preprocessing.save_to_json(df=generation_translated_df, export_cols=generation_translated_df.columns, fname= f"ttrain_generation_{target_lang}.json", pdir=kwargs["cache_dir"])

        logger.info("done...")


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)
