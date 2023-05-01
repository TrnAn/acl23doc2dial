from modelscope.utils.constant import DownloadMode
from modelscope.msdatasets import MsDataset
from modelscope.utils.logger import get_logger
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
from typing import Union
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
sns.set_palette('pastel')

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


def test_split(dataset:pd.DataFrame, test_size:float=0.1):
    logger.info("split dataset...")
    
    return ([],[]) if len(list(dataset)) == 0 else train_test_split(dataset, test_size=test_size)

def add_lang_token(dataset:pd.DataFrame, lang_key:str, colnames:list=["query"], token_colname:str="lang"):
    logger.info(f"adding special language token {lang_key} to input query...")

    def concat_special_token(col):
        try:
            is_list = not isinstance(eval(col.iloc[0]), str)
        except:
            is_list = False

        if not is_list:
            return col + " " + LANG_TOKENS_DD[lang_key]
        else:
            return [[elem + " " + LANG_TOKENS_DD[lang_key] for elem in eval(sublist)] for sublist in col]
            
    dataset[token_colname] = LANG_TOKENS_DD[lang_key]
    dataset[colnames] = dataset[colnames].apply(lambda x: concat_special_token(x)).astype("str")

    return dataset




