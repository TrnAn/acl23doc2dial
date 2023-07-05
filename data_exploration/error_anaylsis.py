#%%
import pandas as pd
import seaborn as sns
from rouge import Rouge
import sacrebleu
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', 2)
custom_palette = ['#FFB3E6', '#FFD9B3', '#B3FFE6', '#FFE6B3']
sns.set(style='ticks', palette=custom_palette)
sns.despine(trim=True, right=True, top=True)

#%%
def boxplot(df, column, by, title, y_label:str="sequence length", x_label:str="scores"):
    plt.figure(figsize=(8, 6))  # Set the figure size

    # Plot the boxplot
    ax = sns.boxplot(x=column, y=by, data=df, linewidth=1.5, width=0.5, fliersize=4, orient='v')
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + (ylim[1] - ylim[0]) * 0.1)  # Increase the y-axis upper limit by 10%

    # Set titles and labels
    plt.title(f'{title} - Boxplot of values by {by}', fontsize=16)
    plt.xlabel(f'Group: {x_label}', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.show()


#%%
def get_dataset(lang:str="en"):
    path = f"{f'{lang}_' if lang == 'en' else ''}no_lang_token\\outputStandardFileBaseline_{lang}.json"
    df  = pd.read_json(path, lines = True)
    df  = df.replace(r'\|', '\\|', regex=True)
    return df.copy()
#%%
df_fr_vi    = get_dataset("fr_vi")
df_en       = get_dataset("en")

#%%
print(df_fr_vi.groupby('lang').apply(lambda x: x.sample(1)).to_markdown(index=False)) # get 1 sample from each language
print(df_en[["query", "gold_response", "response"]].head(2).to_markdown(index=False))

# %%
## Compute Rouge-L F1 score &
rouge = Rouge()
get_rougel_fscore = lambda x: rouge.get_scores([x['response']], [x['gold_response']])[0]['rouge-l']['f']
df_fr_vi["rouge_l"] = df_fr_vi.apply(get_rougel_fscore, axis=1)
df_en["rouge_l"]    = df_en.apply(get_rougel_fscore, axis=1)

#%%
print(f'Rouge-L F1 score on languages fr & vi: {df_fr_vi["rouge_l"].mean() * 100}')
print(f'Rouge-L F1 score on languages fr: {df_fr_vi[df_fr_vi.lang == "fr"]["rouge_l"].mean() * 100}')
print(f'Rouge-L F1 score on languages vi: {df_fr_vi[df_fr_vi.lang == "vi"]["rouge_l"].mean() * 100}')
print(f'Rouge-L F1 score on languages en: {df_en["rouge_l"].mean() * 100}')
#%%
## Compute Bleu score &
bleu = lambda x: sacrebleu.sentence_bleu(x["response"], [x["gold_response"]]).score
df_fr_vi["bleu"] = df_fr_vi.apply(bleu, axis=1)
df_en["bleu"] = df_en.apply(bleu, axis=1)

#%%
print(f'Bleu score on languages fr & vi: {df_fr_vi["bleu"].mean() }')
print(f'Bleu score on languages fr: {df_fr_vi[df_fr_vi.lang == "fr"]["bleu"].mean() }')
print(f'Bleu score on languages vi: {df_fr_vi[df_fr_vi.lang == "vi"]["bleu"].mean() }')
print(f'Bleu score on languages en: {df_en["bleu"].mean()}')

#%%
df_fr_vi["total_score"] = df_fr_vi["rouge_l"] + df_fr_vi["bleu"]
df_en["total_score"]    = df_en["rouge_l"] + df_en["bleu"]

#%%
def get_bottom_top_n(df, n:int=10, languages=["en"]):
    df = df.copy()

    top_ns, bottom_ns = [], []
    for lang in languages:
            bottom_ns.append(df[df.lang == lang].nsmallest(n, "total_score").reset_index(drop=True))
            top_ns.append(df[df.lang == lang].nlargest(n, "total_score").reset_index(drop=True))

    return bottom_ns, top_ns

#%%
def print_bottom_top_n(bottom_ns, top_ns, languages=["fr", "vi"]):
    languages = ", ".join(languages)
    columns = list(set(bottom_ns[0].head(1).columns) - set(["lang"]))
    for i, (bottom, top) in enumerate(zip(bottom_ns, top_ns)):
        print(f'## Bottom 10 on `total_score` for {languages.split(", ")[i]}')
        print(bottom[columns].to_markdown())
        print()
        print(f'## Top 10 on `total_score` for {languages.split(", ")[i]}')
        print(top[columns].to_markdown())
        print()

#%%
n = 10
fr_vi_bottom_ns, fr_vi_top_ns = get_bottom_top_n(df=df_fr_vi, languages=["fr", "vi"])
print_bottom_top_n(fr_vi_bottom_ns, fr_vi_top_ns)

#%%
n = 10
en_bottom_ns, en_top_ns = get_bottom_top_n(df=df_en, languages=["en"])
print_bottom_top_n(en_bottom_ns, en_top_ns, languages=["en"])

#%%
def get_quantiles(tmp):
    df= tmp.copy()
    labels = ["score_q0_25", "score_q25_50", "score_q_50_75", "score_q75_100"]
    df['score_quantile'] = pd.qcut(df['total_score'], q=4, labels=labels)
    df['bleu_quantile'] = pd.qcut(df['bleu'], q=4, labels=labels)
    df['rouge1_quantile'] = pd.qcut(df['rouge_l'], q=4, labels=labels)


    df["pred_length"]      = df.response.str.len()
    df["target_length"]    = df.gold_response.str.len()
    df["length_diff"]      = (df['target_length'] - df['pred_length']).abs()
    return df

#%%
q_all   = get_quantiles(df_fr_vi)
q_en    = get_quantiles(df_en)
q_vi    = get_quantiles(df_fr_vi[df_fr_vi.lang=='vi'])
q_fr    = get_quantiles(df_fr_vi[df_fr_vi.lang=='fr'])

#%%
boxplot(q_all, by='pred_length', column='score_quantile', y_label="sequence length: predicted response", title="lang: all")
boxplot(q_all, by='target_length', column='score_quantile', y_label="sequence length: target response",title="lang: all")
boxplot(q_all, by='length_diff', column='score_quantile', y_label="sequence abs length diff: target vs. predicted response",title="lang: all")

#%%
boxplot(q_en, by='pred_length', column='score_quantile', y_label="sequence length: predicted response", title="lang: en")
boxplot(q_en, by='target_length', column='score_quantile', y_label="sequence length: target response", title="lang: en")
boxplot(q_en, by='length_diff', column='score_quantile', y_label="sequence abs length diff: target vs. predicted response", title="lang: en")


#%%
boxplot(q_vi, by='pred_length', column='score_quantile', y_label="sequence length: predicted response", title="lang: vi")
boxplot(q_vi, by='target_length', column='score_quantile', y_label="sequence length: target response",title="lang: vi")
boxplot(q_vi, by='length_diff', column='score_quantile', y_label="sequence abs length diff: target vs. predicted response", title="lang: vi")

#%%
boxplot(q_fr, by='pred_length', column='score_quantile', y_label="sequence length: predicted response",  title="lang: fr")
boxplot(q_fr, by='target_length', column='score_quantile', y_label="sequence length: target response",title="lang: fr")
boxplot(q_fr, by='length_diff', column='score_quantile', y_label="sequence abs length diff: target vs. predicted response",title="lang: fr")