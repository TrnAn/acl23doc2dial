from modelscope.utils.constant import DownloadMode
from modelscope.msdatasets import MsDataset
from modelscope.utils.logger import get_logger
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
from typing import Union
import json
import os 
from ast import literal_eval
import argparse
from itertools import chain
from rank_bm25 import BM25Okapi

logger = get_logger()

LANG_TOKENS = {"fr": "<fr>",
               "vi": "<vi>",
               "en": "<en>",
               "cn": "<cn>"}

LANG_TOKENS_DD = defaultdict(lambda: "<unk>", LANG_TOKENS)

def read(dir:str, force_download:bool=False):
    """read in datasets from modelscope hub

    Args:
        dir (str): directory of the given dataset
        mode (str, optional): define download mode. Defaults to DownloadMode.FORCE_REDOWNLOAD.

    Returns:
        pd.DataFrame: dataset as df
    """

    logger.info(f"read in dataset from {dir=}...")

    dataset = MsDataset.load(
        dir,
        download_mode=DownloadMode.FORCE_REDOWNLOAD if force_download else DownloadMode.REUSE_DATASET_IF_EXISTS)

    return pd.DataFrame(list(dataset))


def test_split(dataset:pd.DataFrame, random_state:int=42, test_size:float=0.1):
    logger.info("split dataset...")
    
    return (None, None) if dataset is None else train_test_split(dataset, test_size=test_size, random_state=random_state)


def add_lang_token(dataset:Union[pd.DataFrame, list], lang_key:str, colnames:list=["query"], token_colname:str="lang"):
    """add language token to query

    Args:
        dataset (pd.DataFrame): df containing queries
        lang_key (str): specifies language token
        colnames (list, optional): column names containing the queries that shall be appended by the lang token. Defaults to ["query"].
        token_colname (str, optional): new column name where language token will be stored additionally. Defaults to "lang".

    Returns:
        _type_: _description_
    """
    
    if dataset is None:
        return dataset 
    
    def concat_special_token(col):
        try:
            tmp = col.apply(literal_eval)
            return tmp.apply(lambda l: [ LANG_TOKENS_DD[lang_key] + " " + x for x in l])
        except:
            return   LANG_TOKENS_DD[lang_key] + " " + col

    logger.info(f"adding special language token {lang_key} to input query...")
            
    # dataset[token_colname] = LANG_TOKENS_DD[lang_key]
    dataset[colnames] = dataset[colnames].apply(lambda x: concat_special_token(x))

    return dataset


def save_to_json(df:pd.DataFrame, export_cols:list, fname:str="dev.json", pdir:str=""):
    os.makedirs(pdir, exist_ok=True)
    logger.info(f"save test set: {pdir}/{fname}...")
    df[export_cols].to_json(f"{pdir}/{fname}", orient="records", lines=True, force_ascii=False)
    logger.info("DONE...")


def add_hard_negatives(query, positive, negative, corpus:list, n:int=0):
    if n == 0:
        return [negative]
    
    corpus = list(set(corpus) -  set([positive, negative]))
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split(" ")
    top_n = bm25.get_top_n(tokenized_query, corpus, n=n)

    return [negative] + top_n


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--cache-dir", help= "Specifiy cache dir to save model to", type= str, default= ".")
    parser.add_argument("---extended-rerank-dataset-fname", help= "Specifiy cache dir to save model to", type= str, default= "extended_rerank_dataset.json")
    parser.add_argument("---extended-generation-dataset-fname", help= "Specifiy cache dir to save model to", type= str, default= "extended_generation_dataset.json")
    parser.add_argument("--eval-input-file", help= "File to read eval dataset (query, rerank, response) from", type=str, default=None)
    parser.add_argument("--eval-lang", help= "Specify list of languages to train models on, e.g., [['fr', 'vi'], ['fr'], ['vi']]", type=eval)
    parser.add_argument("--lang-token", help= "Add language token <lang> to input", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--only-inference", help= "Only run inference scripts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--only-train", help= "Only run training scripts", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--test-size", help= "Set test split", type= float, default= 0.1)
    parser.add_argument("--batch-accumulation", help= "Use batch accumulation to maintain baseline results", action=argparse.BooleanOptionalAction)
    parser.add_argument("--gradient-accumulation-steps", help= "Specifiy cache dir to save model to", type= int, default= 1)
    parser.add_argument("--num-devices", help= "Specifiy number of devices available", type= int, default= 1)
    parser.add_argument("--batch-size", help= "Specifiy batch size", type= int, default= 128)
    parser.add_argument("--per-gpu-batch-size", help= "Specifiy batch size", type= int, default= 64)
    parser.add_argument("--retrieval-step", help= "Initiate translation for retrieval step", action=argparse.BooleanOptionalAction)
    parser.add_argument("--rerank-step", help= "Initiate translation for rerank step", action=argparse.BooleanOptionalAction)
    parser.add_argument("--generation-step", help= "Initiate translation for generation step", action=argparse.BooleanOptionalAction)
    parser.add_argument("--target-langs", help= "Specify target languages, e.g. ['fr', 'vi']", type=eval) 
    parser.add_argument("--source-langs", help= "Specify source languages, i.e., ['en']", type=eval) 
    parser.add_argument("--save-output", help= "Save output of current pipeline step", type=int, default=0) 
    parser.add_argument("--translate-mode", help= "Specify source languages", type=str, default="") 
    parser.add_argument("--is-inference", help= "is inference pipeline step", action=argparse.BooleanOptionalAction)
    parser.add_argument("--equal-dataset-size", help= "Set all datasets to comparable dataset sizes", type=int, default=0) 
    parser.add_argument("--add-n-hard-negatives", help= "Set number of hard negatives to add to existing negative passages", type=int, default=0) 

    args, _ = parser.parse_known_args()

    return vars(args)


def get_equal_dataset_size_by_lang(df:pd.DataFrame):
    logger.info("set datasets to equal sizes...")
    grouped = df.groupby('lang')
    # Find the size of each group and get the minimum group size
    min_group_size = grouped.size().astype(int).min()
    # Sample 'n' values from each group
    sampled_df = grouped.apply(lambda x: x.sample(min_group_size))

    return sampled_df.reset_index(drop=True)


def get_unique_langs(arr):
    seen = set()
    unique_flat_list = list(chain.from_iterable(sublist for sublist in arr if not any(item in seen or seen.add(item) for item in sublist)))
    return unique_flat_list


def add_translation2trainset(train_df:pd.DataFrame, lang:str, pipeline_step:str, dir:str):
    logger.info(f"add translated datapoints to train dataset...")

    translated_df = pd.read_json(f"{dir}/ttrain_{pipeline_step}_{lang}.json", lines=True, encoding="utf-8-sig")
    if pipeline_step == "generation":
        translated_df = translated_df.rename({"passages": "rerank"},  axis='columns')

        translated_df = translated_df.dropna(subset=['query'])
    translated_df["lang"] = lang 
    print(f"example of translated datapoints: {translated_df.head(2)}")

    new_train_dataset = pd.concat([train_df, translated_df])
    new_train_dataset = new_train_dataset.reset_index(drop=True)

    print(f"successfully added {len(new_train_dataset) - len(train_df)} new datapoints to train dataset...")

    return new_train_dataset


def get_unique_passages(df:pd.DataFrame, lang:str=None):
    unique_passages = df["positive"].tolist() + df["negative"].tolist()
    if lang is not None:
        unique_passages = [f"{LANG_TOKENS_DD[lang]} {p}" for p in unique_passages]
        
    return unique_passages 

