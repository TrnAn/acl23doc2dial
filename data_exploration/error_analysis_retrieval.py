#%%
import pandas as pd
import numpy as np
from utils import preprocessing 
from ast import literal_eval
import json
from tqdm import tqdm
import random
import os
random.seed(42)

#%%
tst = preprocessing.read('DAMO_ConvAI/ViDoc2BotRetrieval')
gtst = preprocessing.read('DAMO_ConvAI/ViDoc2BotGeneration')

#%%
# vi_train_dataset = preprocessing.read('DAMO_ConvAI/ViDoc2BotRetrieval')
vi_train_dataset = pd.read_json('all_passages/vi.json')
vi_train_dataset["lang"] = "vi" 

fr_train_dataset = pd.read_json('all_passages/fr.json')
fr_train_dataset["lang"] = "fr" 

#%%
vi_domains = vi_train_dataset[0].str.extract(r"//\s*([^//\s*]*)$")
vi_domains = set(vi_domains[0].str.replace("vi-",""))

fr_domains = fr_train_dataset[0].str.extract(r"//\s*([^//\s*]*)$")
fr_domains = set(fr_domains[0].str.replace("fr-",""))
# %%
print(f"domains that are shared: {vi_domains & fr_domains}")
print(f"domains that are not shared: {vi_domains ^ fr_domains}")

# english domains ["dmv", "ssa", "studentaid", "va"]
domain_mapping = {
    "dmv": "DepartmentOfMotorVehicles",
    "ssa": "SocialSecurity",
    "studentaid": "StudentFinancialAidinUSA",
    "va": "VeteransAffairs"
}

print(f"domains that are not shared with en: {set(domain_mapping.values()) ^ (vi_domains & fr_domains)}")

#%%
# TRANSLATION
en_fr   = pd.read_json("ttrain\\ttrain_retrieval_fr.json", lines=True)
en_vi   = pd.read_json("ttrain\\ttrain_retrieval_vi.json", lines=True)
en      = pd.read_json("en_train_dataset_retrieval_generation_in_domain.json", lines=True)


#%%
en_fr   = en_fr[["query", "response", "positive", "negative"]]
en_vi   = en_vi[["query", "response", "positive", "negative"]]
en      = en[["query", "response", "positive", "negative"]]

#%%
merged = en.merge(en_fr, on="response", suffixes=["", "_fr"], how="inner")
merged = merged.merge(en_vi, on="response", suffixes=["", "_vi"], how="inner")

#%%
merged  = merged.drop("response", axis=1)

#%%
merged = merged[sorted(merged.columns)]

#%% 
merged  = merged.replace(r'\|', '\\|', regex=True)
print(merged.sample(n=50).to_markdown(index=False))
