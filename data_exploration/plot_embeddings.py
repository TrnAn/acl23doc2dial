#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.express as px
from sklearn.manifold  import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json

#%%
print(os.getcwd())
#%%
lang_token = True
pdir = "0_baseline_lt"
#%%
df = pd.read_csv(os.path.join(pdir, "passage_embeddings_vi_fr.csv"))
df["embedding"] = df["embedding"].apply(eval)

#%%
# translate_df = pd.read_csv("ttest\\passage_embeddings_vi_fr.csv", quotechar='"', escapechar='\\')
# translate_df["embedding"] = translate_df["embedding"].apply(eval)

# translate_df['lang'] = translate_df['passage'].str.extract(r'^<([^>]+)>')

#%%
df["domain"] = df["passage"].str.extract(r"//(?!.*//).*?-(.*)$")

#%%

vi_passages = pd.read_json(f"all_passages{'/lang_token' if lang_token else ''}/vi.json")

#%%
df["lang"] = df["passage"].apply(lambda row: "vi" if row in vi_passages[0].tolist() else "fr")


#%%
def plot_embeddings(df, title, is_translate:bool=False, langs=["fr", "vi"]):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(np.vstack(df["embedding"]))

    
    # tsne = TSNE(n_components=3, random_state=42)
    # embeddings_tsne3d = tsne.fit_transform(np.vstack(df["embedding"]))

    
    df["passage_short"] = df["passage"].str[:45] + "..."
    
    df["x"] = embeddings_tsne[:, 0]
    df["y"] = embeddings_tsne[:, 1]

    pastel_colors = [
    '#F1ABB9',
    '#FF6961',
    '#004953', 
    '#CCA8E0', 
    '#FF9966',
    '#4B68B8', 
    '#A2E4B8', 
    '#FBDB65', 
    '#9D2235'
    ]
    symbols = []
    if "fr" in langs:
        symbols += ["circle-open"]
    if "vi" in langs:
        symbols += ["x"]

    fig = px.scatter(df, x="x", y="y", symbol_sequence=symbols, 
                    color_discrete_sequence=pastel_colors,
                    hover_data="passage_short", 
                    symbol="lang" if len(langs)>1 else None, 
                    color="domain", 
                    opacity=.5)

            
    plot_dir = os.path.join(pdir,"plots", "embeddings")
    os.makedirs(plot_dir, exist_ok=True)

    fig.update_layout(
    width=1200, height=800,
    xaxis_title_font=dict(size=24),  # Font size for x-axis label
    yaxis_title_font=dict(size=24),  # Font size for y-axis label
    legend_title_font=dict(size=30),   # Font size for legend title
    legend_font=dict(size=30),          # Font size for legend labels
    legend=dict(
    itemsizing='constant',
    traceorder='normal',
    font=dict(size=26))
)
    

    fig.update_traces(marker=dict(size=16))
    fig.write_image(os.path.join(plot_dir, f"t_sne_projection_{'_'.join(langs)}.png"))
    fig.show()

#%%
plot_embeddings(df, title="of Doc2Bot Fr+Vi: passage encodings of baseline retrieval model" , is_translate=False)

#%%
plot_embeddings(df[df["lang"] == "fr"], title="of Doc2Bot Fr+Vi: passage encodings of baseline retrieval model" , is_translate=False, langs=["fr"])
#%%
plot_embeddings(df[df["lang"] == "vi"], title="of Doc2Bot Fr+Vi: passage encodings of baseline retrieval model" , is_translate=False, langs=["vi"])


#%%
# CONFUSION MATRIX
# Create a confusion matrix
import scikitplot as skplt
from matplotlib.colors import LinearSegmentedColormap

def _make_plot(df:pd.DataFrame, normalize:bool=False, mode:str="all", top_n:int=20):
    cm = confusion_matrix(df['target_domain'], df['outputs_domain'])
    labels = sorted(set(df['target_domain']) | (set(df['outputs_domain'])))
    # Define colors
    colors = [
        '#FFDAB9',  # Pastel peach
        '#FFA07A',  # Light salmon
        '#CD5C5C',  # Indian red
        '#8B0000'   # Dark red
    ]

# Create a custom color gradient palette
    n_bins = 400

    # Create a custom color gradient palette
    # colors = [pastel_peach, dark_crimson_red]
    # n_bins = 100  # Number of bins for the colormap
    cmap_name = 'pastel_gradient'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    if normalize:
        row_sums = cm.sum(axis=1)
        # Handle normalization with zero-sum row handling
        cm = np.where(row_sums[:, np.newaxis] == 0, 0, cm.astype('float') / row_sums[:, np.newaxis])

    # Set up seaborn style
    sns.set(style="whitegrid")
    # cmap = "Pastel2"
    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm, annot=True, cmap=cmap, fmt='.2f' if normalize else 'g', cbar=False, xticklabels=labels, yticklabels=labels)
    
    for i in range(len(labels)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='white', lw=1))
        

    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('Actual', fontsize=20)
    plt.xticks(fontsize=16, rotation=-40, ha='left')
    plt.yticks(fontsize=16)

    # title = f'Confusion Matrix for top {top_n} retrieved documents' 
    # if mode == "languages":
    #     title = f"{title} for {df['lang'].unique()[0]}"
    # plt.title(title, fontsize=22, pad=24)

    # Manually create a mappable for the colorbar using imshow
    im = ax.imshow(np.array(cm), cmap=cmap)

    # Add colorbar to the right using the created mappable
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Occurrences', rotation=270, labelpad=20)

    plot_dir = f"{pdir}/plots/cms"
    os.makedirs(plot_dir, exist_ok=True)
    # Adjust figure size and layout
    plt.tight_layout()
    fname = f"{plot_dir}/retrieval_top{top_n}_{'_'.join(df['lang'].unique())}"
    if normalize:
        fname = f"{fname}_nornalized"
    plt.savefig(f"{fname}.png")
    plt.show()

#%%
import scikitplot as skplt
from matplotlib.colors import LinearSegmentedColormap

def _make_cm_lang(df:pd.DataFrame, normalize:bool=False, mode:str="all", top_n:int=20):
    cm = confusion_matrix(df['targets_lang'], df['outputs_lang'])
    labels = sorted(set(df['targets_lang']) | (set(df['outputs_lang'])))
    # Define colors
    colors = [
        '#FFDAB9',  # Pastel peach
        '#FFA07A',  # Light salmon
        '#CD5C5C',  # Indian red
        '#8B0000'   # Dark red
    ]

    #   Create a custom color gradient palette

    n_bins = 400

    # Create a custom color gradient palette
    # colors = [pastel_peach, dark_crimson_red]
    # n_bins = 100  # Number of bins for the colormap
    cmap_name = 'pastel_gradient'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    if normalize:
        row_sums = cm.sum(axis=1)
        # Handle normalization with zero-sum row handling
        cm = np.where(row_sums[:, np.newaxis] == 0, 0, cm.astype('float') / row_sums[:, np.newaxis])

    # Set up seaborn style
    sns.set(style="whitegrid")
    # cmap = "Pastel2"
    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(cm, annot=True, cmap=cmap, fmt='.2f' if normalize else 'g', cbar=False, xticklabels=labels, yticklabels=labels)

    for i in range(len(labels)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='white', lw=1))
        

    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('Actual', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # title = f'Confusion Matrix for top {top_n} retrieved documents' 
    # if mode == "languages":
    #     title = f"{title} for {df['lang'].unique()[0]}"
    # plt.title(title, fontsize=22, pad=24)

    # Manually create a mappable for the colorbar using imshow
    im = ax.imshow(np.array(cm), cmap=cmap)

    # Add colorbar to the right using the created mappable
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Occurrences', labelpad=20)

    plot_dir = f"{pdir}/plots/cms"
    os.makedirs(plot_dir, exist_ok=True)
    # Adjust figure size and layout
    plt.tight_layout()
    fname = f"{plot_dir}/retrieval_lang_top{str(top_n)}"
    if normalize:
        fname = f"{fname}_nornalized"
    plt.savefig(f"{fname}.png")
    plt.show()

#%%
retrieval_results = pd.read_json(os.path.join(pdir, "evaluate_result.json"))
retrieval_results = retrieval_results.explode("outputs")
retrieval_results['outputs_lang'] = retrieval_results["outputs"].apply(lambda row: "vi" if row in vi_passages[0].tolist() else "fr")

#$%%
retrieval_results['targets_lang'] = retrieval_results["targets"].apply(lambda row: "vi" if row in vi_passages[0].tolist() else "fr")
#%%
retrieval_results["targets_domain"] = retrieval_results["targets"].str.extract(r"//(?!.*//).*?-(.*)$")
retrieval_results["outputs_domain"] = retrieval_results["outputs"].str.extract(r"//(?!.*//).*?-(.*)$")

#%%
hn_counts:list=[0]
top_n:int=1
_make_cm_lang(retrieval_results,  top_n=top_n)
_make_cm_lang(retrieval_results, top_n=top_n, normalize=True)
#%%
hn_counts:list=[0]
top_n:int=20

pdir = "domain_clf"
with open(f"{pdir}\\evaluate_result.json", 'r') as f:
    data = json.load(f)

retrieval_df = pd.DataFrame({'outputs': data['outputs'], 'target': data['targets']})
topn_df = retrieval_df.explode("outputs").groupby("target").head(top_n)

topn_df["outputs_domain"] = topn_df["outputs"].str.extract(r"//(?!.*//).*?-(.*)$")
topn_df["target_domain"] = topn_df["target"].str.extract(r"//(?!.*//).*?-(.*)$")

topn_df['lang'] = topn_df["target"].apply(lambda row: "vi" if row in vi_passages[0].tolist() else "fr")

_make_plot(topn_df,  top_n=top_n)
_make_plot(topn_df, normalize=True,top_n=top_n)
# PLOT EACH LANGUAGE
for lang in topn_df["lang"].unique():
    print(lang)
    lang_df = topn_df[topn_df["lang"] == lang]
    _make_plot(lang_df, mode="languages",top_n=top_n)
    _make_plot(lang_df, normalize=True, mode="languages",top_n=top_n)


#%%
vi_passages = pd.read_json('all_passages/vi.json')
vi_passages["lang"] = "vi" 

fr_passages = pd.read_json('all_passages/fr.json')
fr_passages["lang"] = "fr" 

#%%
from utils import preprocessing
vi_train_dataset = preprocessing.read('DAMO_ConvAI/ViDoc2BotRetrieval')
fr_train_dataset = preprocessing.read('DAMO_ConvAI/FrDoc2BotRetrieval')
#%%
# plot freq of lang datasets + passages
import plotly.express as px
import pandas as pd

# Sample data
# data = {
#     'languages': ['French', 'French', 'Vietnamese', 'Vietnamese'],
#     'queries': ['train set', 'test set', 'train set', 'test set'],
#     # 'passages': ['passages'] * 6,
#     'count': [len(fr_train_dataset)*0.9, len(fr_train_dataset)*0.1,
#                 int(len(vi_train_dataset)*0.9), int(len(vi_train_dataset)*0.1),
#                 ]
# }

data = {
    'languages': ['French',  'Vietnamese'],
    'queries': ['passages',  'passages'],
    # 'passages': ['passages'] * 6,
    'count': [len(fr_passages), len(vi_passages)]
}



df = pd.DataFrame(data)

def plot_freq(df):
    # Define pastel peach and pastel lilac colors
    pastel_peach = '#FFDAB9'
    pastel_lilac = '#D5A6E8'
    pastel_red = '#FFA07A'
    # Create a grouped bar plot using Plotly Express
    fig = px.bar(df, x='languages', y='count', color='queries',
                barmode='group',
                labels={'count': 'count', 'language': 'languages', 'passages': 'passages'},
                color_discrete_map={'train set': pastel_peach, 'test set': pastel_red, 'passages': pastel_lilac})


    fig.update_layout(
        legend_font_size=16, # Adjust the legend labels font size
        legend_title_text='',  # Remove legend title
        legend=dict(
            tracegroupgap=0
        ),
        margin=dict(l=20, r=20, t=40, b=40) 
    )
    # Show the plot
    fig.write_image(os.path.join(pdir, "frequencies.png"))
    fig.show()

plot_freq(df)
