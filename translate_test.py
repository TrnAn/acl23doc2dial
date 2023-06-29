from transformers import AutoModelForSeq2SeqLM
from transformers import NllbTokenizer
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import re
import json
import logging
from utils.preprocessing import get_args, LANG_TOKENS_DD
from utils import preprocessing
from tqdm import tqdm
tqdm.pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model       = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)

iso_lang = {
    "fr": "fra_Latn",
    "vi": "vie_Latn",
    "en": "eng_Latn"
}

def translate(df, colnames:list, source_lang:str, target_lang:str, batch_size:int=256, lang_token:bool=False):
    tokenizer   = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=iso_lang[source_lang])
    tokenize    = lambda x: tokenizer(x, return_tensors="pt",  padding=True, truncation=True).to(model.device)
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

            df_translate[colname] = [(f"{LANG_TOKENS_DD[target_lang]} " if lang_token else '') + q.replace(f"{LANG_TOKENS_DD[source_lang]} ", '') for q in translated_query]

    return df_translate


def translate_passages(passage_col:pd.Series, all_passages:list,  source_lang:str, target_lang:str, batch_size:int=256, lang_token:bool=False):
    tokenizer   = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=iso_lang[source_lang])
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

        tmp = []
        for item in passage_col:
            translated_passage = passages_dict.get(item)
            translated_passage = (f"{LANG_TOKENS_DD[target_lang]} " if lang_token else '') + translated_passage.replace(f"{LANG_TOKENS_DD[source_lang]} ", '')
            print(f"{translated_passage=}")
            tmp += [translated_passage]

        translated_passages = pd.Series(tmp)
    return translated_passages


def get_dataset(**kwargs):
    if kwargs["retrieval_step"]: 
        df = pd.read_json(f'{kwargs["cache_dir"]}/{kwargs["eval_input_file"]}', lines=True)
        return df 
    return None


def replace_passages(passage_column:pd.Series, all_passages:dict[str]):
    return passage_column.replace(all_passages)

def save_all_passages(source_lang, target_lang, lang_token, cache_dir):
    parent_dir = "all_passages/lang_token" if lang_token else "all_passages"

    with open(f'{parent_dir}/{source_lang}.json') as f:
        lang_passages = json.load(f)
        passages = translate_passages(passage_col=pd.Series(lang_passages, name='passages'), all_passages=lang_passages, source_lang=source_lang, target_lang=target_lang, lang_token=lang_token)
        print(f"{passages.head(2)=}")
        passages = passages.to_list()
        json_data = json.dumps(passages, ensure_ascii=False)

        # Save JSON-formatted string to a file
        fpath = f"{cache_dir}/{parent_dir}"
        os.makedirs(fpath, exist_ok=True)
        with open(f"{fpath}/{source_lang}2{target_lang}.json", 'w', encoding='utf-8-sig') as f:
            print(f"save translated {source_lang} passages to {fpath}...")
            f.write(json_data)
            print(f"DONE.")


def main(**kwargs):

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers= [
            logging.FileHandler(f'{kwargs["cache_dir"]}/{"_".join(kwargs["source_langs"])}2{"_".join(kwargs["target_langs"])}_translate.log'),  # File handler
            logging.StreamHandler()  # Console handler
        ]
    )
    
    logger = logging.getLogger()

    logger.info(f"enter translate-{kwargs['translate_mode']} mode on {device=}...")
    train_dataset = get_dataset(**kwargs)

    if train_dataset is None:
        return 0
    
    target_lang = kwargs["target_langs"][0]
    tmp_dfs = [train_dataset[train_dataset['lang'] == target_lang]]
    for source_lang in kwargs["source_langs"]:
        logger.info(f'translate {source_lang} dataset to {target_lang=}')

        retrieval_fname = f"ttest_{'_'.join(kwargs['source_langs'])}2{target_lang}.json"
        retrieval_path  = os.path.join(kwargs["cache_dir"], retrieval_fname)
        
        if kwargs["retrieval_step"] and not os.path.exists(retrieval_path):
            retrieval_translated_df         = translate(df=train_dataset[train_dataset['lang'] == source_lang], colnames=["query", "positive", "negative"], source_lang=source_lang, target_lang=target_lang, lang_token=kwargs["lang_token"])
            retrieval_translated_df["lang"] = target_lang
            tmp_dfs += [retrieval_translated_df]
            save_all_passages(source_lang=source_lang, target_lang=target_lang, lang_token=kwargs["lang_token"], cache_dir=kwargs["cache_dir"])

    translated_df = pd.concat(tmp_dfs)
    print(f"{translated_df.head(2)=}")
    preprocessing.save_to_json(df=translated_df, export_cols=translated_df.columns, fname= retrieval_fname, pdir=kwargs["cache_dir"])

    logger.info("done...")


if __name__ == '__main__':
    kwargs = get_args()
    main(**kwargs)