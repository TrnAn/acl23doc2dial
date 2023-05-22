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
cn_train_dataset = preprocessing.read('DAMO_ConvAI/ZhDoc2BotDialogue')
en_train_dataset = preprocessing.read('DAMO_ConvAI/EnDoc2BotDialogue')
#%%
en_train_dataset["passages"] = en_train_dataset.passages.apply(literal_eval)
cn_train_dataset["passages"] = cn_train_dataset.passages.apply(literal_eval)

#%%
def random_first_passage(df, df_row, same_domain:bool=True):
    domain = df_row['domain']
    
    domain_rows = df[df['domain'] == domain] if same_domain else df[df['domain'] != domain]
    if same_domain:
        domain_rows = domain_rows.drop(df_row.name)  # Drop the current row

    if not domain_rows.empty:
        random_index = random.choice(domain_rows.index)
        return domain_rows.loc[random_index, 'passages'][0]
    return None

#%%
# PROCESS CN DATASET
## List to store the combined data
domains     = ["health", "insurance", "technology", "wikihow"]
colnames    = ['turn', 'role', 'utterance', 'document', 'grounding', 'grounding_id']
cn_df       =  pd.DataFrame(columns=colnames)


for domain in tqdm(domains):
    directory   = f"Doc2Bot\\samples\\{domain}\\dialogs"
    combined_data = []
    # Iterate over each JSON file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                # Load the JSON data from the file
                json_data = json.load(file)
                # Append the data to the combined list
                combined_data.extend(json_data)

    # Create a DataFrame from the combined data
    df = pd.json_normalize(combined_data)
    df["domain"] = domain
    cn_df = pd.concat([cn_df, df], ignore_index=True)

#%%
cn_df.tail(1)

#%%
cn_train_dataset.head(3)

#%%
# Apply the function to the DataFrame column
cn_train_dataset[["ql", "qr"]] = cn_train_dataset["query"].str.extract(r"<last_turn>\s*(.*?)\s*<agent>|<last_turn>\s*(.*)", expand=False)
cn_train_dataset["formatted_query"] = cn_train_dataset["ql"].fillna('') + cn_train_dataset["qr"].fillna('')
cn_train_dataset = cn_train_dataset.drop(columns=["ql", "qr"])
#%%
cn_merged = cn_train_dataset.merge(cn_df, left_on="formatted_query", right_on="utterance", how="inner")

#%%
# Output the updated DataFrame
print(f"{len(cn_merged)=}; {len(cn_train_dataset)=}")

#%%
cn_merged_in_domain     = cn_merged.copy()
cn_merged_other_domain  = cn_merged.copy()

#%%
cn_merged_in_domain["negative"]     = cn_merged.apply(lambda x: random_first_passage(cn_merged, x), axis=1)
cn_merged_other_domain["negative"]  = cn_merged.apply(lambda x: random_first_passage(cn_merged, x, same_domain=False), axis=1)

#%%
cn_merged_in_domain["positive"]     = cn_merged_in_domain.passages.str[0]
cn_merged_other_domain["positive"]  = cn_merged_other_domain.passages.str[0]

#%%
cn_merged_in_domain.columns
#%%
cn_merged_in_domain[["query", "passages", "response", "negative", "positive"]].to_json("cn_train_dataset_in_domain.json", orient="records", lines=True, force_ascii=False)

#%%
cn_merged_other_domain[["query", "passages", "response", "negative", "positive"]].to_json("cn_train_dataset_other_domain.json", orient="records", lines=True, force_ascii=False)

#%%
# ref_dataset = preprocessing.read('DAMO_ConvAI/FrDoc2BotRetrieval')

#%%
# ref_dataset.head(1)

#%%
# PROCESS EN DATASET
all_queries = pd.read_json("data\\data\\multidoc2dial\\multidoc2dial_dial_train.json")

#%%
domains     = ["dmv", "ssa", "studentaid", "va"]
colnames    = ['da', 'references', 'role', 'turn_id', 'utterance', 'domain']
queries_df  =  pd.DataFrame(columns=colnames)

for idx, domain in enumerate(tqdm(domains)):
    domain_df = all_queries.loc[domain].explode().apply(pd.Series)["turns"].explode().apply(pd.Series)
    print(f"{domain=}", flush=True)
    id_sps = domain_df.references.apply(lambda x: [item["id_sp"] for item in x])
    doc_ids = domain_df.references.apply(lambda x: [item["doc_id"] for item in x])
    domain_df["id_sp"]  = id_sps
    domain_df["doc_id"] = doc_ids
    domain_df["domain"] = domain
    queries_df = pd.concat([queries_df, domain_df], ignore_index=True)

#%%
# Read JSON data from file
# with open("data\\data\\multidoc2dial\\multidoc2dial_doc.json", 'r') as file:
#     json_data = json.load(file)

#%%
# domains     = ["dmv", "ssa", "studentaid", "va"]
# colnames    = ["id_sp", "text_sec", "title", "doc_id"]
# passages_df =  pd.DataFrame(columns=colnames)
# for idx, domain_name in enumerate(tqdm(domains)):
#     domain_doc = json_data['doc_data'][domain_name]

#     # Extract the desired columns
#     data_list = []
#     for key, value in domain_doc.items():
#         if key == domain_name:
#             continue
#         doc_id = value.get('doc_id') # top-level title
#         spans = value['spans']
#         for span_key, span_value in spans.items():
#             data = {
#                 "doc_id": doc_id,
#                 "id_sp": span_value.get("id_sp"),
#                 "text_sec": span_value.get("text_sec"),
#                 "title": span_value.get("title") # child level title
#             }
#             data_list.append(data)

#         tmp_df              = pd.DataFrame(data_list, columns=colnames)
#         tmp_df["domain"]    = domain_name
#         passages_df = pd.concat([passages_df, tmp_df], ignore_index=True)

#%%
# The 'passages' attribute contains a list of passages arranged according to reply dependencies, followed by a reverse-ordered chain of titles concatenated with "//" as the delimiter. Example:
# ['Sign up or log into MyDMV [6 ]//5. Not Bringing Proper Documentation to DMV Office //Top 5 DMV Mistakes and How to Avoid Them ']
# sep = "//"
# tmp = passages_df.doc_id.str.replace(r'\s*#\d+.*', '', regex=True) # remove #<number> at end of string
# passages_df["passages"] = passages_df.text_sec + sep + passages_df.title + sep + tmp

#%%
# queries_df["id_sp_response"] = queries_df["id_sp"].shift(-1)

#%%
tmp = queries_df[(~(queries_df.da.isin(["respond_solution"]) & (queries_df.role == "user")))].reset_index(drop=True)

#%%
en_merged = tmp.merge(en_train_dataset, left_index=True, right_index=True, how='inner')

#%%
en_merged_not_domain = en_merged.copy()

#%%
# Apply the function to create a new column with randomly selected first passage
en_merged['negative'] = en_merged.apply(lambda x: random_first_passage(en_merged, x), axis=1)

#%%
en_merged_not_domain['negative'] = en_merged_not_domain.apply(lambda x: random_first_passage(en_merged_not_domain, x, same_domain=False), axis=1)

#%%
en_merged['positive']               = en_merged.passages.str[0]
en_merged_not_domain['positive']    = en_merged_not_domain.passages.str[0]

#%%
en_merged[["query", "passages", "response", "positive", "negative"]].to_json("en_train_dataset_retrieval_generation_in_domain.json", lines=True, orient="records")

#%%
en_merged_not_domain[["query", "passages", "response", "positive", "negative"]].to_json("en_train_dataset_retrieval_generation_other_domain.json", lines=True, orient="records")
