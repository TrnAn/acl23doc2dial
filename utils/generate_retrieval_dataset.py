#%%
import pandas as pd
import numpy as np
from utils import preprocessing 
from ast import literal_eval

#%%
#!/bin/sh
%cd data &&\
%wget http://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip &&\
%wget http://doc2dial.github.io/multidoc2dial/file/multidoc2dial_domain.zip &&\
%unzip multidoc2dial.zip &&\
%unzip multidoc2dial_domain.zip &&\
%rm *.zip &&\
%wget https://huggingface.co/facebook/rag-token-nq/raw/main/question_encoder_tokenizer/tokenizer_config.json && \
%wget https://huggingface.co/facebook/rag-token-nq/raw/main/question_encoder_tokenizer/vocab.txt

#%%
cn_train_dataset = preprocessing.read('DAMO_ConvAI/ZhDoc2BotDialogue')
en_train_dataset = preprocessing.read('DAMO_ConvAI/EnDoc2BotDialogue')

#%%
ref_dataset = preprocessing.read('DAMO_ConvAI/FrDoc2BotRetrieval')
#%%
en_train_dataset["passages"] = en_train_dataset.passages.apply(literal_eval)
cn_train_dataset["passages"] = cn_train_dataset.passages.apply(literal_eval)

#%%
ref_dataset.head(2)
# %%
cn_train_dataset.head(2)

#%%
en_train_dataset.head(2)

#%%
# Add positives
cn_train_dataset["positive"] = cn_train_dataset.passages.str[0]
en_train_dataset["positive"] = en_train_dataset.passages.str[0]

#%%
en_train_dataset["positive"] 