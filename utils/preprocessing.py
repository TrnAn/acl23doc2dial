from modelscope.utils.constant import DownloadMode
from modelscope.msdatasets import MsDataset
from modelscope.utils.logger import get_logger
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
from torch import nn 
from typing import Union
import json
import os 

logger = get_logger()

LANG_TOKENS = {"fr": "<fr>",
               "vn": "<vn>",
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
            is_list = not isinstance(eval(col.iloc[0]), str)
        except:
            is_list = False
        
        if is_list:
            tmp = pd.Series([[elem + " " + LANG_TOKENS_DD[lang_key] for elem in eval(sublist)] for sublist in col], index=col.index)
            return tmp.apply(lambda s: json.dumps(s))
        else:
            return col + " " + LANG_TOKENS_DD[lang_key]  

    logger.info(f"adding special language token {lang_key} to input query...")
            
    dataset[token_colname] = LANG_TOKENS_DD[lang_key]
    dataset[colnames] = dataset[colnames].apply(lambda x: concat_special_token(x)).astype("str")

    return dataset


def save_to_json(df:pd.DataFrame, export_cols:list, fname:str="dev.json", dir:str=""):
    dir = os.path.join(dir, fname)

    logger.info(f"save test set {fname}...")
    df[export_cols].to_json("dev.json", orient="records", lines=True)
    logger.info("DONE...")


def resize_token_embeddings(model, new_token_size):
    """resize embedding layer of Modelscope models to fit updated number of tokens

    Args:
        model (modelscope.models.Model): model to resize embeddings
        new_token_size (int): new token embedding size

    Returns:
        modelscope.models.Model: model with resized token embeddings
    """

    embedding_layer = model.embeddings.word_embeddings
    old_num_tokens, old_embedding_dim = embedding_layer.weight.shape

    # create new embedding layer with new token emb size
    new_embeddings = nn.Embedding(
            new_token_size, old_embedding_dim
    )

    new_embeddings.to(
        embedding_layer.weight.device,
        dtype=embedding_layer.weight.dtype,
    )

    # copying old entries
    new_embeddings.weight.data[:old_num_tokens, :] = embedding_layer.weight.data[
        :old_num_tokens, :
    ]

    model.embeddings.word_embeddings = new_embeddings

    return model
