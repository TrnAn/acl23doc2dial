import os
import pandas as pd
from modelscope.utils.logger import get_logger
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
sns.set_palette('pastel')

logger = get_logger()

def get_freq_df(train_df:pd.DataFrame, dev_df:pd.DataFrame):
    freq_df = pd.DataFrame({"relative_freq_train":  train_df.value_counts("lang", normalize=True), 
                            "relative_freq_dev":    dev_df.value_counts("lang", normalize=True),
                            "absolute_freq_train":  train_df.value_counts("lang"),
                            "absolute_freq_dev":    dev_df.value_counts("lang")
                            })
    
    logger.info(f"{freq_df}")
    logger.info(f"total train set size: {int(freq_df.sum().loc['absolute_freq_train'])}")
    logger.info(f"total dev set size: {int(freq_df.sum().loc['absolute_freq_dev'])}")

    return freq_df


def plot_freq(df:pd.DataFrame, plot_dir:str="./plots", fname:str="lang_frequency.png", save_plot:bool=True):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    _, ax1 = plt.subplots(figsize=(10, 10))

    freq_df = df.reset_index()
    abs_freq_df = freq_df.loc[:,~freq_df.columns.str.startswith('relative')]
    tidy = abs_freq_df.melt(id_vars='lang').rename(columns=str.title)
    ax = sns.barplot(x='Lang', y='Value', hue='Variable', data=tidy, ax=ax1)
    ax.set_ylabel("Frequency")

    _ = [ax.bar_label(c, label_type='edge', rotation=40) for c in ax.containers]
    
    plt.title('Absolute frequency of languages in train vs dev set', fontsize=16, pad=20)
    plt.legend()

    if save_plot: 
        logger.info(f"saving {fname} to {plot_dir}...")
        plt.savefig(os.path.join(plot_dir, fname))
        logger.info("DONE.")

    plt.show()
