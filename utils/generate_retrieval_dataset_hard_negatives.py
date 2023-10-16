#%%
import pandas as pd
import numpy as np
from utils import preprocessing 
import json
from tqdm import tqdm
import random
import os
from rank_bm25 import BM25Okapi

random.seed(42)
tqdm.pandas()

#%%
cn_train_dataset = preprocessing.read('DAMO_ConvAI/ZhDoc2BotDialogue')
en_train_dataset = preprocessing.read('DAMO_ConvAI/EnDoc2BotDialogue')

#%%
# en_train_dataset["passages"] = en_train_dataset.passages.apply(literal_eval)
# cn_train_dataset["passages"] = cn_train_dataset.passages.apply(literal_eval)

#%%
# ENGLISH SAMPLING
en_train_dataset["positive"] = en_train_dataset["passages"].apply(eval).str[0]
cn_train_dataset["positive"] = cn_train_dataset["passages"].apply(eval).str[0]

#%%
all_en_passages = en_train_dataset["passages"].apply(eval).explode().unique()
all_cn_passages = cn_train_dataset["passages"].apply(eval).explode().unique()

#%%
def _add_hard_negatives(query, positive, corpus:list, n:int=1):   
    corpus = list(set(corpus) -  set([positive]))
   
    # tokenized_corpus = [doc.lower().split(" ") for doc in corpus]

    # bm25 = BM25Okapi(tokenized_corpus)
    bm25 = BM25Okapi(corpus)
    # tokenized_query = query.lower().split(" ")
    top_n = bm25.get_top_n(query, corpus, n=n)

    return top_n
#%%

en_train_dataset["negative"] = en_train_dataset.progress_apply(lambda x: _add_hard_negatives(
            query=x["query"], 
            positive=x["positive"], 
            corpus=all_en_passages
            ), 
        axis=1)

#%%
cn_train_dataset["negative"] = cn_train_dataset.progress_apply(lambda x: _add_hard_negatives(
            query=x["query"], 
            positive=x["positive"], 
            corpus=list(all_cn_passages)
            ), 
        axis=1)

#%%
en_train_dataset["negative"] = en_train_dataset["negative"].str[0]

#%%
cn_train_dataset["negative"] = cn_train_dataset["negative"].str[0]
#%%
en_train_dataset[["query", "positive", "negative", "response"]].to_json("en_train_dataset_retrieval_generation_hn.json", lines=True, orient="records")

#%%
cn_train_dataset[["query", "positive", "negative", "response"]].to_json("cn_train_dataset_retrieval_generation_hn.json", force_ascii=False, lines=True, orient="records")

#%%
from itertools import chain
all_en_passages = en_train_dataset[['positive', 'negative']].values.tolist()
all_en_passages = list(chain(*all_en_passages))


#%%
all_cn_passages = cn_train_dataset[['positive', 'negative']].values.tolist()
all_cn_passages = list(chain(*all_cn_passages))

#%%
# Save the JSON data to a file
with open('all_passages\\en.json', 'w') as file:
    json.dump(all_en_passages, file, indent=4)
    file.write('\n') 

#%%
with open('all_passages\\cn.json', 'w') as file:
    json.dump(all_cn_passages, file, indent=4)
    file.write('\n') 

#%%
# Your existing JSON string
# Read the JSON string from a file
pdir = "all_passages\\id_to_passage.json"
with open(pdir, 'r') as file:
    json_str = file.read()

# Parse the JSON string into a dictionary
data = json.loads(json_str)

# Find the highest existing key and increment it by one
new_key = max(map(int, data.keys())) + 1

# Add new key-value pair
for passage in all_en_passages + all_cn_passages:
    data[f"{new_key}"] = passage
    new_key += 1

# Convert the updated dictionary back to JSON string
updated_json_str = json.dumps(data, indent=4, ensure_ascii=False)
print(updated_json_str)

with open(pdir, 'w') as file:
    file.write(updated_json_str)


#%%
import json
import os
from tqdm import tqdm
directory = "all_passages\\lang_token\\"
if not os.path.exists(directory):
    os.makedirs(directory)


combined_data = {}

languages = ["fr", "vi", "en", "cn"]
# Iterate over the JSON files
index = 0
for lang in languages:
        lang_values = []
        with open(f"all_passages\\{lang}.json", "r") as file:
            json_data = json.load(file)
            print(json_data, flush=True)
            for val in json_data:
                formatted_value = f"<{lang}> {val}"
                combined_data[str(index)] = formatted_value
                lang_values += [formatted_value]
                index += 1
            with open(f"{directory}{lang}.json", "w") as file:
                json.dump(lang_values, file, indent=4, ensure_ascii=False)
#         # Add the array of strings to the combined data dictionary
#             combined_data[str(index)] = json_data

# Save the combined data as a new JSON file
output_file = f"{directory}id_to_passage.json"
with open(output_file, "w") as file:
    json.dump(combined_data, file, indent=4, ensure_ascii=False)

print("Combined data saved to", output_file)