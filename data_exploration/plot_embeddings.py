#%%
import pandas as pd
import seaborn as sns
from rouge import Rouge
import sacrebleu
import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.express as px
from sklearn.manifold  import TSNE
pd.set_option('display.max_colwidth', 2)
custom_palette = ['#FFB3E6', '#FFD9B3', '#B3FFE6', '#FFE6B3']
sns.set(style='ticks', palette=custom_palette)
sns.despine(trim=True, right=True, top=True)

#%%
print(os.getcwd())
#%%
df = pd.read_csv("../hard_negatives2_no_fn/DAMO_ConvAI/nlp_convai_retrieval_pretrain/passage_embeddings.csv")
df["embedding"] = df["embedding"].apply(eval)


#%%
df["domain"] = df["passage"].str.extract(r"//(?!.*//).*?-(.*)$")

#%%
vi_passages = pd.read_json("../all_passages/lang_token/vi.json")

#%%
df["lang"] = df["passage"].apply(lambda row: "vi" if row in vi_passages[0].tolist() else "fr")

#%%
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(np.vstack(df["embedding"]))

#%%
df["x"] = embeddings_tsne[:, 0]
df["y"] = embeddings_tsne[:, 1]
fig = px.scatter(df, x="x", y="y", text="passage", hoverinfo="text", symbol="lang", color="domain", opacity=.5, title="t-SNE visualization of domains")

#%%
fig.write_image("scatterblatter.png")
fig.show()